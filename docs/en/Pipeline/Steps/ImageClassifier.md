`ImageClassifier` is a pipeline action for image classification. It does not run inference by itself at step construction time — instead, its attributes configure `ClassificationTrainingTask` during `Pipeline.train()`.

Parameters:

* `num_classes` — output classes; inferred from the dataset when `None`
* `source_mode="full"` — classify the whole image; `"roi"` expects a crop from a prior localization step

```python
from torchvision.models.resnet import ResNet18_Weights, resnet18

from hyppopipe.data import TrainVal
from hyppopipe.data.dataset import ImageFolderDataset
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.train import Trainer, TrainingConfig

data = TrainVal(
    train=ImageFolderDataset("data/train"),
    val=ImageFolderDataset("data/val"),
)

pipeline = Pipeline(
    {
        "glaucoma_classification": Step(
            ImageClassifier(),
            description="Classify glaucoma",
        ),
    },
)

pipeline.train(
    step_config={
        "glaucoma_classification": Trainer(
            data=data,
            config=TrainingConfig(epochs=10, batch_size=16, lr=3e-4),
            model_candidates=[resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)],
        )
    },
)
```

## Documentation

::: hyppopipe.pipeline.image.classification.ImageClassifier
