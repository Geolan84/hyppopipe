# ImageTransformer

ImageTransformer - специальный тип действия для трансформации изображений.

Этот тип позволяет последовательно описать необходимые изменения для изображения, чтобы при выполнении шага получить на выходе уже изменённое изображение.

Преобразование может быть одиночным или множественным.

`ImageTransformer` не реализует трансформации самостоятельно — внутри используется комбинация `torchvision.transforms` и OpenCV. Для медицинских снимков доступны `circle_crop`, `ellipse_crop` и `min_area_rect_crop` (обрезка по паттерну с фундус-камеры).

```python
from hyppopipe.data.image import Image
from hyppopipe.pipeline.image.transform import ImageTransformer
from hyppopipe.pipeline import Pipeline, Step

from torchvision.transforms import v2 as transforms


image_data_processing = Pipeline(
    {
        "GetResizedImage": Step(
            ImageTransformer().resize(224),
            inputs=["__input__"],
        ),
        "GetSharpenImage": Step(
            ImageTransformer().sharpen(2.0),
            inputs=["GetResizedImage"],
        ),
        "GetBluredImage": Step(
            ImageTransformer().gaussian_blur(5, sigma=(0.1, 2.0)),
            inputs=["GetResizedImage"],
        ),
        # Гибкий интерфейс для множественных трансформаций:
        "FluentTransformations": Step(
            ImageTransformer().resize(224).gaussian_blur(5, sigma=(0.1, 2.0)).rotate(90),
            inputs=["__input__"],
        ),
        # Фабрика из трансформаций torchvision:
        "ComposedTransformations": Step(
            ImageTransformer.from_compose(
                transforms.Compose([
                    transforms.Resize(224),
                    transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
                    transforms.RandomRotation((90, 90)),
                ])
            ),
            inputs=["__input__"],
        )

    },
    shift_result=False,
)

original_image = Image.from_path("images/image74prime.tif")
pipeline_result = image_data_processing.predict(original_image)
```

## Документация

::: hyppopipe.pipeline.image.transform