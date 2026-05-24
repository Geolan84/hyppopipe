The central entity of the framework is the `Pipeline` class. It stores an ordered sequence of steps (`Step` objects) and exposes methods to train models and run inference on the described workflow.

Steps are passed as a dictionary `steps` where the key is a unique step name (for example, `"nail_localization"`) and the value is a `Step` instance. Since Python 3.7, dict insertion order is preserved, so this layout is safe and guarantees unique names.

During inference, the pipeline supports two execution modes:

* **`shift_result=True`** (default): the first step receives the input passed to `predict()`, and each following step automatically receives the previous step's output. Similar to scikit-learn pipelines.
* **`shift_result=False`**: each step uses its own `inputs` list; the pipeline resolves dependencies and injects results from named steps before calling the action. Similar to Kedro.

Each pipeline stage is represented by a `Step`. It stores a callable **action**, optional **inputs** (step names or `"__input__"`), a human-readable **description**, and an optional **preprocessor** for incoming data.

The action can be any callable: a plain function, a lambda, or an object with `__call__`. For specialized ML tasks, pass dedicated functors — the framework adjusts training and inference accordingly:

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
            description="Locate tumor region",
        ),
        "tumor_classify": Step(
            ImageClassifier(source_mode="roi"),
            inputs=["tumor_localize"],
            description="Classify tumor type",
        ),
    },
    shift_result=False,
)

image = Image.from_path("mri_slice.dcm")
result = brain_pipeline.predict(image, bundle_path="artifacts/")
```

!!! note "Training on datasets, not on step outputs"
    Each model is trained on the dataset you provide to `Trainer`, not on the outputs of the previous pipeline step. Steps only define inference-time data flow.

## Documentation

::: hyppopipe.pipeline
