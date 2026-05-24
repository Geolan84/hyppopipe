`ImageSegmentator` is a pipeline action for semantic or instance segmentation.

Parameters:

* `kind` — `"instance"` (Mask R-CNN targets) or `"semantic"` (per-pixel class map)
* `num_classes` — classes including background; inferred from data when `None`
* `input_channels` — channel count after default semantic input preparation (default `3`)
* `image_size` — `(H, W)` for semantic batching; instance models resize internally

Use with [`PairedImageMaskFolderDataset`](../../Dataset/PairedImageMaskFolder.md) or YOLO polygon labels.

```python
from hyppopipe.data.dataset import PairedImageMaskFolderDataset
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.segmentation import ImageSegmentator
from hyppopipe.train import Trainer, TrainingConfig

splits = PairedImageMaskFolderDataset("data/nails/").as_split_data()

pipeline = Pipeline(
    {
        "nail_segment": Step(
            ImageSegmentator(kind="semantic"),
            description="Segment nail region",
        ),
    },
)
```

At inference, `SegmentationPrediction` provides masks and visualization helpers.

## Documentation

::: hyppopipe.pipeline.image.segmentation.ImageSegmentator
