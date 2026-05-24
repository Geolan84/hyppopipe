`ImageClassifier` — действие пайплайна для классификации изображений. Сам по себе при создании шага не выполняет инференс — его атрибуты настраивают `ClassificationTrainingTask` во время `Pipeline.train()`.

Параметры:

* `num_classes` — число классов; при `None` определяется из датасета
* `source_mode="full"` — классификация всего изображения; `"roi"` — классификация области, полученной на шаге локализации

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
            description="Классификация глаукомы",
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

## Документация

::: hyppopipe.pipeline.image.classification.ImageClassifier
