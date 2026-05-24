Datasets in Hyppopipe live in `hyppopipe.data.dataset`. All concrete readers inherit from `ImageDataset` and return data in a task-specific format (image + label, image + mask, and so on).

| Class | Task | Layout |
|-------|------|--------|
| [`ImageFolderDataset`](ImageFolder.md) | Classification | `root/<class>/images` |
| [`PairedImageMaskFolderDataset`](PairedImageMaskFolder.md) | Semantic segmentation | `root/images` + `root/masks` |
| [`YAMLDataset`](YAMLDataset.md) | Classification / detection / segmentation | YOLO `data.yaml` |

## Train / val / test splits

Open datasets often ship with pre-defined `train`, `validation`, and `test` folders. Hyppopipe respects this layout by default (`absorb_folders=False`).

When you need a different ratio, set `absorb_folders=True` and split programmatically:

```python
from hyppopipe.data.dataset import ImageFolderDataset, split_random_fractions

full_ds = ImageFolderDataset("data/", absorb_folders=True)
splits = split_random_fractions(full_ds, (0.7, 0.15, 0.15), seed=42)
```

`TrainVal` and `TrainValTest` (alias `SplitData`) wrap train/val(/test) subsets for [`Trainer`](../Training/trainer.md).

## Task adapters

Some datasets contain labels for multiple tasks. Adapters in `hyppopipe.data.dataset.adapters` expose only the fields needed for a given task: `classification`, `detection`, `segmentation`, `roi_classification`.

## Documentation

::: hyppopipe.data.dataset.base.ImageDataset
