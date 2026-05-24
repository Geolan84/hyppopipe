`YAMLDataset` загружает датасеты в формате Ultralytics/YOLO, описанные файлом `data.yaml`. Предоставляет выборки `train`, `val` и опционально `test`, поддерживает классификацию, детекцию и сегментацию в зависимости от формата разметки.

Файлы меток обрабатываются в двух режимах:

1. **Ограничивающий прямоугольник** — `<class_id> <x_center> <y_center> <width> <height>` (нормализованные координаты). Подходит для классификации и детекции.
2. **Полигональная маска** — `<class_id> <x1> <y1> ... <xn> <yn>`. Для сегментации возвращает маску; для детекции строит bbox по минимальным и максимальным координатам.

```python
from hyppopipe.data.dataset import YAMLDataset

ds = YAMLDataset("datasets/my_task/data.yaml")
split_data = ds.as_split_data()
train_cls = split_data.train.as_classification_dataset()
```

Как и у датасетов на основе директорий, поддерживаются флаги `absorb_folders` (игнорировать готовое разбиение) и `strict` (проверка расширений и MIME-типов при загрузке).

## Документация

::: hyppopipe.data.dataset.readers.yaml_dataset.YAMLDataset
