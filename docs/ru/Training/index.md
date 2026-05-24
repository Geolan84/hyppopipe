Обучение в Hyppopipe организуется через [`Trainer`](trainer.md) для каждого шага пайплайна. Сам `Pipeline` только описывает последовательность шагов; метод `Pipeline.train()` принимает словарь «имя шага → Trainer».

Связанные сущности:

| Сущность | Назначение |
|----------|------------|
| [`Trainer`](trainer.md) | Обучает одну или несколько моделей для шага |
| `TrainingConfig` | Эпохи, batch size, device, оптимизатор, loss, monitor, early stopping |
| `ModelCandidate` | Фабрика torchvision + варианты предобученных весов |
| `TrainResult` | Сводный результат с экспортом на диск |

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

При обучении `Trainer` определяет тип действия шага (`ImageClassifier`, `ImageSegmentator`, `ImageLocalizer`) и вызывает соответствующую задачу из `hyppopipe.train.tasks`.

## Документация

::: hyppopipe.train.Trainer
