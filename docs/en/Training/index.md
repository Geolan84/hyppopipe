Training in Hyppopipe is orchestrated by [`Trainer`](trainer.md) for each pipeline step. The pipeline itself only declares the step sequence; `Pipeline.train()` accepts a mapping of step names to `Trainer` instances.

Key related types:

| Entity | Role |
|--------|------|
| [`Trainer`](trainer.md) | Trains one or more models for a step |
| `TrainingConfig` | Epochs, batch size, device, optimizer, loss, monitor, early stopping |
| `ModelCandidate` | Torchvision factory + pretrained weight variants |
| `TrainResult` | Aggregated results with export to disk |

```python
from torchvision.models.resnet import ResNet18_Weights, resnet18

from hyppopipe.train import ModelCandidate, Trainer, TrainingConfig
from hyppopipe.train.objectives import ClassificationObjectives

common = TrainingConfig(epochs=20, batch_size=32, lr=1e-3, device="cuda")

result = pipeline.train(
    config=common,
    step_config={
        "classify": Trainer(
            data=splits,
            config=common.copy_with(epochs=30),
            model_candidates=[
                ModelCandidate(resnet18, ResNet18_Weights.IMAGENET1K_V1),
            ],
            monitor=ClassificationObjectives.monitor("accuracy"),
        ),
    },
)
result.export("artifacts/run_001")
```

During training, `Trainer` inspects the step's action type (`ImageClassifier`, `ImageSegmentator`, `ImageLocalizer`) and dispatches the appropriate task from `hyppopipe.train.tasks`.

## Documentation

::: hyppopipe.train.Trainer
