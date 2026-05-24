`ImageFolderDataset` - это класс для чтения датасетов классификации, на основе директорий. Каждая директорая содержит снимки из одного класса.

Пример структуры датасета:

```
SomeImageFolder
├── class1
	├── image1.jpg
	├── imageN.jpg
├── class2
	├── image1.jpg
	├── imageM.jpg
```

Пример использования в коде:

```python
from hyppopipe.data import TrainVal
from hyppopipe.data.dataset import ImageFolderDataset

train_dataset = ImageFolderDataset("path_to_train_folder")
val_dataset = ImageFolderDataset("path_to_validation_folder")
data = TrainVal(train=train_dataset, val=val_dataset)
```

!!! tip "Переопределение разбиения"
    Флаг `absorb_folders=True` позволяет игнорировать готовые папки train/val и перераспределить данные через `split_random_fractions(dataset, (0.7, 0.15, 0.15))` без изменения файлов на диске.

## Документация

::: hyppopipe.data.dataset.readers.image_folder.ImageFolderDataset