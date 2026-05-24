---
icon: lucide/rocket
---

# Hyppopipe

**Hyppopipe** is a Python framework for building, training, and running end-to-end machine learning pipelines in medical image analysis. It provides a declarative interface to chain models into multi-step workflows while handling preprocessing, tensor shapes, and spatial alignment automatically.

Typical medical scenarios follow the same pattern: read an image, localize regions of interest, segment them, and classify or regress. Hyppopipe packages these stages into reusable building blocks so you can go from dataset loading to inference in a few commands instead of copying Jupyter notebooks.

!!! tip "Core entities"
    - [`Image`](image.md) — load and display medical images (JPEG, PNG, TIFF, DICOM)
    - [`Pipeline`](pipeline.md) and [`Step`](pipeline.md) — describe execution order and dependencies
    - [Pipeline steps](Pipeline/Steps/index.md) — `ImageClassifier`, `ImageLocalizer`, `ImageSegmentator`, `ImageTransformer`
    - [Datasets](Dataset/index.md) — `ImageFolderDataset`, `PairedImageMaskFolderDataset`, `YAMLDataset`
    - [Training](Training/index.md) — `Trainer`, `TrainingConfig`, `ModelCandidate`

## Quick start

```python
from torchvision.models.resnet import ResNet18_Weights, resnet18

from hyppopipe.data import TrainVal
from hyppopipe.data.dataset import ImageFolderDataset
from hyppopipe.data.image import Image
from hyppopipe.pipeline import Pipeline, Step
from hyppopipe.pipeline.image.classification import ImageClassifier
from hyppopipe.train import Trainer, TrainingConfig

train_dataset = ImageFolderDataset("data/train")
val_dataset = ImageFolderDataset("data/val")
data = TrainVal(train=train_dataset, val=val_dataset)

pipeline = Pipeline(
    {
        "classify": Step(ImageClassifier(), description="Classify pathology"),
    },
)

result = pipeline.train(
    step_config={
        "classify": Trainer(
            data=data,
            config=TrainingConfig(epochs=10, batch_size=16, lr=3e-4),
            model_candidates=[resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)],
        )
    },
)

image = Image.from_path("sample.png")
prediction = pipeline.predict(image, bundle_path=result.export_path)
```

## Local preview

```bash
source .venv/bin/activate
zensical serve
```
