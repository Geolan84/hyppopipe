`ImageLocalizer` is a pipeline action for object detection and region localization. Training uses torchvision Faster R-CNN under the hood.

Parameters:

* `num_classes` — total classes **including background**; when `None`, uses `len(dataset.classes) + 1`

Typical use: locate a region of interest before classification with `ImageClassifier(source_mode="roi")`.

```python
from hyppopipe.data.dataset import YAMLDataset
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.pipeline.image.localization import ImageLocalizer

pipeline = Pipeline(
    {
        "tumor_localize": Step(
            ImageLocalizer(),
            description="Locate tumor bounding box",
        ),
        "tumor_classify": Step(
            ImageClassifier(source_mode="roi"),
            inputs=["tumor_localize"],
            description="Classify tumor type inside ROI",
        ),
    },
    shift_result=False,
)
```

At inference time, `LocalizationPrediction` exposes bounding boxes and optional cropped `Image` views.

## Documentation

::: hyppopipe.pipeline.image.localization.ImageLocalizer
