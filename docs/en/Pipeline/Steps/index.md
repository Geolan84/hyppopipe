Each pipeline stage is represented as an object of the `Step` class. This data structure stores some called action object, as well as additional information about the step: which steps depend on, a text description, and a preprocessor for the input data. It is important that the stored "action" is any called object. This can be either a regular function declared via def or lambda, or an object of any type with support for the **\_\_call\_\_** method.

```python
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.pipeline.image.transform import ImageTransformer
from hyppopipe.pipeline import Pipeline, Step


image_data_processing = Pipeline(
    {
        "GetResizedImage": Step(
            ImageTransformer().resize(224),
        ),
        "Classify image": Step(
            ImageClassifier(),
        )
    },
)
```

Ready-made classes are immediately available to the user: ImageLocalizer, ImageSegmentator, ImageTransformer, ImageClassifier (package hippopipe.pipeline.image).