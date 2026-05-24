## Описание

`Image` используется для представления изображения внутри фреймворка.

Основные способы создания:

* `Image.from_path(path)` — загрузка с диска
* `Image.from_base64(data)` — загрузка из base64-строки
* `image.show()` — вывод через matplotlib (корректно работает в Jupyter, в отличие от `cv2.imshow`)
* `image.save(path)` — сохранение в файл

При помощи метода `from_path` в память может быть загружено изображение в форматах: `.jpg` / `.jpeg`, `.png`, `.tif` / `.tiff` и `.dcm`. По умолчанию загружаемые файлы проверяются на соответствие допустимым MIME-типам — невалидные файлы отклоняются при `strict=True`.

```python
from hyppopipe.data.image import Image

image = Image.from_path("fundus.tif")
image.show()
image.save("preview.png")
```

## Документация

::: hyppopipe.data.image