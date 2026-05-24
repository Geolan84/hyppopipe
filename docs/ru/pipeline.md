Центральной сущностью фреймворка является класс `Pipeline`. Его ответственность — хранить последовательность шагов (объекты `Step`), а также предоставлять возможность запуска обучения и предсказания для описанного пайплайна.

Шаги задаются в виде словаря `steps`, где ключ — уникальное имя шага (например, `"nail_localization"`), а значение — объект `Step`. Начиная с Python 3.7, словари сохраняют порядок вставки, поэтому такой способ описания безопасен и гарантирует уникальность имён.

В режиме предсказания пайплайн может работать в двух режимах:

* **`shift_result=True`** (по умолчанию): первый шаг получает вход из `predict()`, каждый следующий — результат предыдущего. Аналогично scikit-learn.
* **`shift_result=False`**: каждый шаг использует свой список `inputs`; пайплайн подставляет результаты именованных шагов перед вызовом. Аналогично Kedro.

Каждый этап представлен объектом `Step`, который хранит вызываемое **action**, опциональные **inputs**, **description** и **preprocessor**. Для специализированных ML-задач используйте готовые функторы:

* [`ImageLocalizer`](Pipeline/Steps/ImageLocalizer.md)
* [`ImageSegmentator`](Pipeline/Steps/ImageSegmentator.md)
* [`ImageClassifier`](Pipeline/Steps/ImageClassifier.md)
* [`ImageTransformer`](Pipeline/Steps/ImageTransformer.md)

```python
from hyppopipe.data.image import Image
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.pipeline.image.localization import ImageLocalizer

brain_pipeline = Pipeline(
    {
        "tumor_localize": Step(
            ImageLocalizer(),
            description="Локализация опухоли",
        ),
        "tumor_classify": Step(
            ImageClassifier(source_mode="roi"),
            inputs=["tumor_localize"],
            description="Классификация типа опухоли",
        ),
    },
    shift_result=False,
)

image = Image.from_path("mri_slice.dcm")
result = brain_pipeline.predict(image, bundle_path="artifacts/")
```

!!! note "Обучение на датасете, а не на выходах шагов"
    Каждая модель обучается на датасете, переданном в `Trainer`, а не на результатах предыдущего шага. Шаги определяют поток данных только при инференсе.

## Документация

::: hyppopipe.pipeline
