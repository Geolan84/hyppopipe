`PairedImageMaskFolderDataset` - это класс для чтения датасетов сегментации, на основе директорий. В конструктор передаётся путь к корневой папке датасета, которая содержит ровно две папки: изображения и папка с масками. Названия директорий можно переопределить.

Пользователь может переопределить названия папок.

Пример структуры датасета:

```
NailsSegmentation                                 # Директория датасета
├── images                                        # Фотографии
    ├── 1eecab90-1a92-43a7-b952-0204384e1fae.jpg  # Изображение ногтя
    ├── ...
├── labels                                        # Маски
    ├── 1eecab90-1a92-43a7-b952-0204384e1fae.jpg  # Чёрно белая маска
    ├── ...
```

Пример вызова:

```python
nails_split = PairedImageMaskFolderDataset(
    "path/to/dataset/",
    image_folder="images",
    mask_folder="labels",
).as_split_data(fractions=(0.7, 0.15, 0.15))
```

## Документация

::: hyppopipe.data.dataset.readers.image_folder.PairedImageMaskFolderDataset