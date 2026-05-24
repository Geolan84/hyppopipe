---
icon: lucide/rocket
---

# Hyppopipe

**Hyppopipe** — Python-фреймворк для построения, обучения и запуска сквозных ML-пайплайнов в медицинской обработке изображений. Он предоставляет декларативный интерфейс для связывания моделей в многошаговые сценарии, автоматически обрабатывая препроцессинг, формы тензоров и пространственное выравнивание.

Типичный медицинский сценарий повторяется из проекта в проект: чтение снимка, локализация ROI, сегментация, классификация или регрессия. Hyppopipe упаковывает эти этапы в переиспользуемые блоки, чтобы пройти путь от загрузки датасета до инференса несколькими командами, а не копированием Jupyter-ноутбуков.

!!! tip "Основные сущности"
    - [`Image`](image.md) — загрузка и отображение медицинских снимков (JPEG, PNG, TIFF, DICOM)
    - [`Pipeline`](pipeline.md) и [`Step`](pipeline.md) — порядок выполнения и зависимости шагов
    - [Шаги пайплайна](Pipeline/Steps/index.md) — `ImageClassifier`, `ImageLocalizer`, `ImageSegmentator`, `ImageTransformer`
    - [Датасеты](Dataset/index.md) — `ImageFolderDataset`, `PairedImageMaskFolderDataset`, `YAMLDataset`
    - [Обучение](Training/index.md) — `Trainer`, `TrainingConfig`, `ModelCandidate`

## Быстрый старт

```python
from torchvision.models.resnet import ResNet18_Weights, resnet18

from hyppopipe.data import TrainVal
from hyppopipe.data.dataset import ImageFolderDataset
from hyppopipe.data.image import Image
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.train import Trainer, TrainingConfig

train_dataset = ImageFolderDataset("data/train")
val_dataset = ImageFolderDataset("data/val")
data = TrainVal(train=train_dataset, val=val_dataset)

pipeline = Pipeline(
    {
        "classify": Step(ImageClassifier(), description="Классификация патологии"),
    },
)

result = pipeline.train(
    step_config={
        "classify": Trainer(
            data=data,
            config=TrainingConfig(epochs=10, batch_size=16, lr=3e-4),
            model_candidates=[resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)],
        )
    },
)

image = Image.from_path("sample.png")
prediction = pipeline.predict(image, bundle_path=result.export_path)
```

## Локальный просмотр

```bash
source .venv/bin/activate
zensical serve
```
