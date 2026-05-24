`ImageLocalizer` — действие пайплайна для детекции объектов и локализации регионов интереса. Обучение выполняется на базе Faster R-CNN из torchvision.

Параметры:

* `num_classes` — число классов **включая фон**; при `None` используется `len(dataset.classes) + 1`

Типичный сценарий: локализовать ROI на одном шаге, затем классифицировать его через `ImageClassifier(source_mode="roi")`.

```python
from hyppopipe.data.dataset import YAMLDataset
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.pipeline.image.localization import ImageLocalizer

pipeline = Pipeline(
    {
        "tumor_localize": Step(
            ImageLocalizer(),
            description="Локализация опухоли",
        ),
        "tumor_classify": Step(
            ImageClassifier(source_mode="roi"),
            inputs=["tumor_localize"],
            description="Классификация типа опухоли в ROI",
        ),
    },
    shift_result=False,
)
```

При инференсе `LocalizationPrediction` предоставляет ограничивающие рамки и опционально обрезанные `Image`.

## Документация

::: hyppopipe.pipeline.image.localization.ImageLocalizer
