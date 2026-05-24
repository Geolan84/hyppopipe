Датасеты Hyppopipe находятся в пакете `hyppopipe.data.dataset`. Все конкретные ридеры наследуются от `ImageDataset` и возвращают данные в формате, подходящем для конкретной задачи (изображение + метка класса, изображение + маска и т.д.).

| Класс | Задача | Структура |
|-------|--------|-----------|
| [`ImageFolderDataset`](ImageFolder.md) | Классификация | `root/<class>/снимки` |
| [`PairedImageMaskFolderDataset`](PairedImageMaskFolder.md) | Семантическая сегментация | `root/images` + `root/masks` |
| [`YAMLDataset`](YAMLDataset.md) | Классификация / детекция / сегментация | YOLO `data.yaml` |

## Разбиение на train / val / test

Открытые датасеты часто содержат готовые папки `Train`, `Validation` и `Test`. Hyppopipe учитывает такую структуру по умолчанию (`absorb_folders=False`).

Если нужно другое соотношение, установите `absorb_folders=True` и разделите программно:

```python
from hyppopipe.data.dataset import ImageFolderDataset, split_random_fractions

full_ds = ImageFolderDataset("data/", absorb_folders=True)
splits = split_random_fractions(full_ds, (0.7, 0.15, 0.15), seed=42)
```

`TrainVal` и `TrainValTest` (алиас `SplitData`) оборачивают части выборки для [`Trainer`](../Training/trainer.md).

## Адаптеры задач

Некоторые датасеты содержат разметку сразу для нескольких задач. Адаптеры в `hyppopipe.data.dataset.adapters` отдают только нужные поля: `classification`, `detection`, `segmentation`, `roi_classification`.

## Документация

::: hyppopipe.data.dataset.base.ImageDataset
