from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torchvision.models as tv_models
from torch import nn
from torch.utils.data import DataLoader, Subset

from hyppopipe.pipeline.context import PipelineContext


def _accuracy_from_logits(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[int, int]:
    """Count correct top-1 predictions in a batch.

    Args:
        logits (torch.Tensor): Class logits ``(N, num_classes)``.
        targets (torch.Tensor): Integer labels ``(N,)``.

    Returns:
        tuple[int, int]: ``(num_correct, batch_size)``.
    """
    pred = logits.argmax(dim=1)
    correct = int((pred == targets).sum().item())
    return correct, int(targets.size(0))


def default_adapt_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the last classification layer for ``num_classes`` (torchvision-style).

    Handles common patterns: ``model.fc`` (ResNet-like) or ``model.classifier``
    (Linear or trailing Linear inside ``Sequential``).

    Args:
        model (nn.Module): Pretrained torchvision-style model.
        num_classes (int): Target number of output classes.

    Raises:
        ValueError: If no supported classifier head is found.

    Returns:
        nn.Module: Same ``model`` instance with head replaced in-place.
    """
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    if hasattr(model, "classifier"):
        c = model.classifier
        if isinstance(c, nn.Linear):
            model.classifier = nn.Linear(c.in_features, num_classes)
            return model
        if isinstance(c, nn.Sequential):
            last = c[-1]
            if isinstance(last, nn.Linear):
                in_features = last.in_features
                blocks = list(c.children())[:-1]
                model.classifier = nn.Sequential(
                    *blocks,
                    nn.Linear(in_features, num_classes),
                )
                return model
    raise ValueError(
        "Could not infer classifier head; pass adapt_classifier to FineTuneStep."
    )


class DataLoaderStep:
    """Build ``train_loader`` and optionally ``val_loader`` from context datasets."""

    def __init__(
        self,
        batch_size: int = 32,
        *,
        num_workers: int = 0,
        pin_memory: bool | None = None,
        val_from_test_split: bool = True,
        **loader_kwargs: Any,
    ) -> None:
        """Configure DataLoader defaults for training and validation.

        Args:
            batch_size (int, optional): Batch size. Defaults to ``32``.
            num_workers (int, optional): ``DataLoader`` workers. Defaults to ``0``.
            pin_memory (bool | None, optional): If ``None``, enables when CUDA
                is available. Defaults to ``None``.
            val_from_test_split (bool, optional): If dataset has ``test_indices``,
                build ``val_loader`` as a ``Subset``. Defaults to ``True``.
            **loader_kwargs: Extra arguments forwarded to :class:`DataLoader`.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_from_test_split = val_from_test_split
        self.loader_kwargs = loader_kwargs

    def fit(self, ctx: PipelineContext) -> None:
        """Populate ``ctx.train_loader`` and possibly ``ctx.val_loader``.

        Infers ``ctx.num_classes`` and ``ctx.class_names`` from ``train_src`` when missing.

        Args:
            ctx (PipelineContext): Pipeline state; requires ``ctx.dataset`` or
                ``ctx.train_dataset``.

        Raises:
            ValueError: If neither ``ctx.dataset`` nor ``ctx.train_dataset`` is set.
        """
        train_src = ctx.train_dataset if ctx.train_dataset is not None else ctx.dataset
        if train_src is None:
            msg = "Set ctx.dataset or ctx.train_dataset"
            raise ValueError(msg)

        if ctx.num_classes is None and hasattr(train_src, "num_classes"):
            ctx.num_classes = train_src.num_classes()  # type: ignore[operator]
        if ctx.class_names is None and hasattr(train_src, "classes"):
            ctx.class_names = list(train_src.classes)  # type: ignore[attr-defined]

        pin = self.pin_memory
        if pin is None:
            pin = torch.cuda.is_available()

        ctx.train_loader = DataLoader(
            train_src,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=pin,
            **self.loader_kwargs,
        )

        if ctx.val_loader is not None:
            return
        if ctx.val_dataset is not None:
            ctx.val_loader = DataLoader(
                ctx.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=pin,
                **self.loader_kwargs,
            )
            return
        if self.val_from_test_split and hasattr(train_src, "test_indices"):
            idx = list(train_src.test_indices)  # type: ignore[attr-defined]
            if idx:
                ctx.val_loader = DataLoader(
                    Subset(train_src, idx),
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=pin,
                    **self.loader_kwargs,
                )


class FineTuneStep:
    """Fine-tune a provided ``pretrained_model`` or a torchvision backbone."""

    def __init__(
        self,
        *,
        pretrained_model: nn.Module | None = None,
        backbone: str = "resnet18",
        weights: Any = "DEFAULT",
        epochs: int = 1,
        lr: float = 1e-3,
        adapt_head: bool = True,
        adapt_classifier: Callable[[nn.Module, int], nn.Module] | None = None,
        criterion: nn.Module | None = None,
        verbose: bool = True,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        """Configure fine-tuning hyperparameters and optional custom head adapter.

        Args:
            pretrained_model (nn.Module | None, optional): If set, used instead of
                constructing ``backbone``. Defaults to ``None``.
            backbone (str, optional): ``torchvision.models`` factory name.
                Defaults to ``"resnet18"``.
            weights (Any, optional): Weights argument for the factory.
                Defaults to ``"DEFAULT"``.
            epochs (int, optional): Training epochs. Defaults to ``1``.
            lr (float, optional): Adam learning rate. Defaults to ``1e-3``.
            adapt_head (bool, optional): Whether to replace classifier for
                ``ctx.num_classes``. Defaults to ``True``.
            adapt_classifier (Callable | None, optional): Custom head replacer;
                defaults to :func:`default_adapt_classifier`.
            criterion (nn.Module | None, optional): Loss; defaults to cross-entropy.
            verbose (bool, optional): Log epoch metrics. Defaults to ``True``.
            log_fn (Callable[[str], None] | None, optional): Logger; defaults to
                :func:`print`.
        """
        self.pretrained_model = pretrained_model
        self.backbone = backbone
        self.weights = weights
        self.epochs = epochs
        self.lr = lr
        self.adapt_head = adapt_head
        self.adapt_classifier = adapt_classifier or default_adapt_classifier
        self.criterion = criterion
        self.verbose = verbose
        self.log_fn = log_fn or print

    def _build_model(self, ctx: PipelineContext) -> nn.Module:
        """Instantiate (or take) a model and adapt its classifier head.

        Args:
            ctx (PipelineContext): Must provide ``ctx.num_classes`` when adapting.

        Raises:
            ValueError: If ``ctx.num_classes`` is missing, backbone is unknown,
                or head adaptation fails inside ``adapt_classifier``.

        Returns:
            nn.Module: Model ready for training on device.
        """
        if ctx.num_classes is None:
            msg = (
                "Set ctx.num_classes or run DataLoaderStep with a dataset "
                "that defines num_classes"
            )
            raise ValueError(msg)
        if self.pretrained_model is not None:
            model = self.pretrained_model
        else:
            factory = getattr(tv_models, self.backbone, None)
            if factory is None:
                msg = f"Unknown torchvision backbone: {self.backbone!r}"
                raise ValueError(msg)
            model = factory(weights=self.weights)
        if self.adapt_head:
            model = self.adapt_classifier(model, int(ctx.num_classes))
        return model

    def fit(self, ctx: PipelineContext) -> None:
        """Train ``ctx.model`` on ``ctx.train_loader``; optionally validate.

        Writes lists to ``ctx.history`` under keys ``train_loss``, ``train_acc``,
        and when applicable ``val_loss``, ``val_acc``.

        Args:
            ctx (PipelineContext): Requires ``ctx.train_loader``; builds model
                via :meth:`_build_model` if ``ctx.model`` is ``None``.

        Raises:
            ValueError: If ``ctx.train_loader`` is ``None``.
        """
        if ctx.train_loader is None:
            msg = "Run DataLoaderStep first (ctx.train_loader is required)"
            raise ValueError(msg)

        device = ctx.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        ctx.device = device

        if ctx.model is None:
            ctx.model = self._build_model(ctx)
        ctx.model = ctx.model.to(device)
        ctx.model.train()

        criterion = self.criterion or nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(ctx.model.parameters(), lr=self.lr)

        train_losses: list[float] = []
        train_accs: list[float] = []
        val_losses: list[float] = []
        val_accs: list[float] = []

        for epoch in range(self.epochs):
            running = 0.0
            n = 0
            correct = 0
            total = 0
            for batch_x, batch_y in ctx.train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = ctx.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                running += float(loss.item()) * batch_x.size(0)
                n += batch_x.size(0)
                bc, bt = _accuracy_from_logits(logits, batch_y)
                correct += bc
                total += bt
            train_losses.append(running / max(n, 1))
            train_accs.append(correct / max(total, 1))

            if ctx.val_loader is not None:
                ctx.model.eval()
                vr = 0.0
                vn = 0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for vx, vy in ctx.val_loader:
                        vx = vx.to(device)
                        vy = vy.to(device)
                        logits = ctx.model(vx)
                        vloss = criterion(logits, vy)
                        vr += float(vloss.item()) * vx.size(0)
                        vn += vx.size(0)
                        bc, bt = _accuracy_from_logits(logits, vy)
                        v_correct += bc
                        v_total += bt
                val_losses.append(vr / max(vn, 1))
                val_accs.append(v_correct / max(v_total, 1))
                ctx.model.train()

            if self.verbose:
                msg = (
                    f"epoch {epoch + 1}/{self.epochs} "
                    f"train_loss={train_losses[-1]:.4f} "
                    f"train_acc={train_accs[-1]:.4f}"
                )
                if val_losses:
                    msg += f" val_loss={val_losses[-1]:.4f} val_acc={val_accs[-1]:.4f}"
                self.log_fn(msg)

        ctx.history.setdefault("train_loss", []).extend(train_losses)
        ctx.history.setdefault("train_acc", []).extend(train_accs)
        if val_losses:
            ctx.history.setdefault("val_loss", []).extend(val_losses)
            ctx.history.setdefault("val_acc", []).extend(val_accs)


class ExportCheckpointStep:
    """Save weights and class metadata to a checkpoint file."""

    def __init__(
        self,
        path: str,
        *,
        save_full_model: bool = False,
        use_context_path: bool = True,
    ) -> None:
        """Configure output path and whether to pickle the full module.

        Args:
            path (str): Default checkpoint path when ``ctx.checkpoint_path`` is absent.
            save_full_model (bool, optional): If ``True``, include ``ctx.model`` in
                the payload. Defaults to ``False``.
            use_context_path (bool, optional): Prefer ``ctx.checkpoint_path`` when set.
                Defaults to ``True``.
        """
        self.path = path
        self.save_full_model = save_full_model
        self.use_context_path = use_context_path

    def fit(self, ctx: PipelineContext) -> None:
        """Persist ``state_dict`` and optional metadata to disk.

        Updates ``ctx.checkpoint_path`` to the path written.

        Args:
            ctx (PipelineContext): Must have ``ctx.model`` set.

        Raises:
            ValueError: If ``ctx.model`` is ``None``.
        """
        if ctx.model is None:
            msg = (
                "No model in context; run FineTuneStep or set ctx.model explicitly"
            )
            raise ValueError(msg)
        out = (
            ctx.checkpoint_path
            if self.use_context_path and ctx.checkpoint_path
            else self.path
        )
        payload: dict[str, Any] = {
            "state_dict": ctx.model.state_dict(),
            "class_names": ctx.class_names,
            "num_classes": ctx.num_classes,
        }
        if self.save_full_model:
            payload["model"] = ctx.model
        torch.save(payload, out)
        ctx.checkpoint_path = out
