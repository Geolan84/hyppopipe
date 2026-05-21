Каждый этап пайплайна представляется в виде объекта класса `Step`. Эта структура данных хранит в себе некоторый вызываемый объект action, а также дополнительную информацию о шаге: от каких шагов зависит, текстовое описание, предобработчик для входных данных. Важно, что хранимое «действие» – это любой вызываемый объект. Это может как обычная функция, объявленная через def или lambda, так и объект произвольного типа с поддержкой метода **\_\_call\_\_**.

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

Пользователю сразу доступны готовые классы: ImageLocalizer, ImageSegmentator, ImageTransformer, ImageClassifier (пакет hyppopipe.pipeline.image).