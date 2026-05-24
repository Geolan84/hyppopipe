`Trainer` обучает одну или несколько моделей для отдельного шага пайплайна. Перебирает записи `ModelCandidate` (или готовые экземпляры `nn.Module`), управляет контрольными точками, журналирует прогресс и формирует `StepTrainResult`.

Перед началом обучения `Trainer` определяет тип **action** шага и выбирает реализацию из `hyppopipe.train.tasks` — для классификации, сегментации и локализации отличаются загрузчики данных и расчёт loss.

### TrainingConfig

Неизменяемый dataclass с гиперпараметрами:

* `epochs`, `batch_size`, `val_batch_size`, `lr`, `device`
* `optimizer_name` / `optimizer_kwargs` или пользовательский `optimizer_factory`
* `loss` и `monitor` (см. `hyppopipe.train.objectives`)
* `early_stopping` (`EarlyStoppingConfig`)

Метод `copy_with(epochs=50)` создаёт конфигурацию для отдельного шага на основе общего шаблона.

### ModelCandidate

Оборачивает фабрику модели torchvision и один или несколько `WeightsEnum`, чтобы сравнивать архитектуры и предобученные веса в одном вызове `train()` без дублирования ноутбуков.

### Экспорт

`TrainResult.export(path)` записывает `manifest.json`, веса (`.pth`) и папку `reports/` с `training_history.json`, `loss_curves.png` и `monitor_curves.png`.

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

## Документация

::: hyppopipe.train.trainer.Trainer
