`YAMLDataset` loads Ultralytics/YOLO-style datasets described by a `data.yaml` file. It exposes `train`, `val`, and optional `test` splits and supports classification, object detection, and instance/semantic segmentation depending on label format.

Label files are parsed in two modes:

1. **Bounding box** — `<class_id> <x_center> <y_center> <width> <height>` (normalized). Suitable for classification and detection.
2. **Polygon mask** — `<class_id> <x1> <y1> ... <xn> <yn>`. For segmentation returns a mask; for detection builds a bounding box from min/max coordinates.

```python
from hyppopipe.data.dataset import YAMLDataset

ds = YAMLDataset("datasets/my_task/data.yaml")
split_data = ds.as_split_data()
train_cls = split_data.train.as_classification_dataset()
```

Like folder-based datasets, `YAMLDataset` supports `absorb_folders` (ignore pre-defined splits) and `strict` (validate file extensions and MIME types at load time).

## Documentation

::: hyppopipe.data.dataset.readers.yaml_dataset.YAMLDataset
