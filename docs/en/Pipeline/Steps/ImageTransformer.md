# ImageTransformer

ImageTransformer is a special type of action for image transformation.

This type allows you to consistently describe the necessary changes for the image, so that when performing a step, you can get an already modified image at the output.

The conversion can be single or multiple.

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
        # Or we can use fluent transformations:
        "FluentTransformations": Step(
            ImageTransformer().resize(224).gaussian_blur(5, sigma=(0.1, 2.0)).rotate(90),
            inputs=["__input__"],
        ),
        # Or composed transformations:
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

## Documentation

::: hyppopipe.pipeline.image.transform