`ImageFolderDataset` reads classification datasets laid out as one subdirectory per class. Each immediate child folder of `root` becomes a class name; supported image files inside are indexed as samples.

When `absorb_folders=True`, an extra top-level tier is allowed (`root/<split>/<class>/...`), which is useful when train/val folders are already separated in the dataset.

Example layout:

```
FundusDataset/
├── Glaucoma/
│   ├── img001.png
│   └── img002.png
└── Non Glaucoma/
    ├── img003.png
    └── img004.png
```

```python
from hyppopipe.data import TrainVal
from hyppopipe.data.dataset import ImageFolderDataset

train_dataset = ImageFolderDataset("data/train")
val_dataset = ImageFolderDataset("data/val")
data = TrainVal(train=train_dataset, val=val_dataset)
```

!!! tip "Re-splitting an existing dataset"
    Set `absorb_folders=True` to ignore pre-defined train/val folders and call `split_random_fractions(dataset, (0.7, 0.15, 0.15))` to create new splits without moving files on disk.

## Documentation

::: hyppopipe.data.dataset.readers.image_folder.ImageFolderDataset
