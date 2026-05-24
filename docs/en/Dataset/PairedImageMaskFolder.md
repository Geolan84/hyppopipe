`PairedImageMaskFolderDataset` reads semantic segmentation datasets where masks are stored as images. The constructor expects a root folder with two subdirectories — images and masks — paired by matching file stems.

Folder names default to `"images"` and `"masks"` but can be overridden.

Example layout:

```
NailsSegmentation/
├── images/
│   ├── 1eecab90-1a92-43a7-b952-0204384e1fae.jpg
│   └── ...
└── labels/
    ├── 1eecab90-1a92-43a7-b952-0204384e1fae.jpg
    └── ...
```

```python
from hyppopipe.data.dataset import PairedImageMaskFolderDataset

nails_split = PairedImageMaskFolderDataset(
    "path/to/dataset/",
    image_folder="images",
    mask_folder="labels",
).as_split_data(fractions=(0.7, 0.15, 0.15))
```

By index, the dataset returns an image tensor and a per-pixel class map suitable for segmentation training.

## Documentation

::: hyppopipe.data.dataset.readers.image_folder.PairedImageMaskFolderDataset
