`Trainer` trains one or more models for a single pipeline step. It iterates over `ModelCandidate` entries (or ready `nn.Module` instances), manages checkpoints, logs progress, and produces a `StepTrainResult`.

Before training starts, `Trainer` reads the step's **action** type and selects the matching implementation from `hyppopipe.train.tasks` — classification, segmentation, and localization differ in dataloaders and loss computation.

### TrainingConfig

Immutable dataclass with hyperparameters:

* `epochs`, `batch_size`, `val_batch_size`, `lr`, `device`
* `optimizer_name` / `optimizer_kwargs` or custom `optimizer_factory`
* `loss` and `monitor` (see `hyppopipe.train.objectives`)
* `early_stopping` (`EarlyStoppingConfig`)

Use `copy_with(epochs=50)` to derive a per-step config from a shared default.

### ModelCandidate

Wraps a torchvision model factory and one or more `WeightsEnum` values so you can compare architectures and pretrained weights in a single `train()` call without duplicating notebooks.

### Export

`TrainResult.export(path)` writes `manifest.json`, weights (`.pth`), and a `reports/` folder with `training_history.json`, `loss_curves.png`, and `monitor_curves.png`.

```python
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)

from hyppopipe.data.dataset import YAMLDataset
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.localization import ImageLocalizer
from hyppopipe.train import ModelCandidate, Trainer, TrainingConfig

ds = YAMLDataset("data/brain_tumor/data.yaml")
splits = ds.as_split_data()

pipeline = Pipeline({"localize": Step(ImageLocalizer())})

result = pipeline.train(
    step_config={
        "localize": Trainer(
            data=splits,
            config=TrainingConfig(epochs=15, batch_size=4, lr=5e-4),
            model_candidates=[
                ModelCandidate(
                    fasterrcnn_mobilenet_v3_large_fpn,
                    FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                ),
            ],
        ),
    },
)
```

## Documentation

::: hyppopipe.train.trainer.Trainer
