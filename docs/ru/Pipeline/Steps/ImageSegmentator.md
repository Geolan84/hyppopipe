`ImageSegmentator` — действие пайплайна для семантической или instance-сегментации.

Параметры:

* `kind` — `"instance"` (цели Mask R-CNN) или `"semantic"` (карта классов по пикселям)
* `num_classes` — число классов включая фон; при `None` определяется из данных
* `input_channels` — число каналов после подготовки входа (по умолчанию `3`)
* `image_size` — `(H, W)` для semantic-batching; instance-модели масштабируют внутри

Используется с [`PairedImageMaskFolderDataset`](../../Dataset/PairedImageMaskFolder.md) или полигональной разметкой YOLO.

```python
from hyppopipe.data.dataset import PairedImageMaskFolderDataset
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.segmentation import ImageSegmentator

splits = PairedImageMaskFolderDataset("data/nails/").as_split_data()

pipeline = Pipeline(
    {
        "nail_segment": Step(
            ImageSegmentator(kind="semantic"),
            description="Сегментация области ногтя",
        ),
    },
)
```

При инференсе `SegmentationPrediction` предоставляет маски и методы визуализации.

## Документация

::: hyppopipe.pipeline.image.segmentation.ImageSegmentator
