The `Image` class represents a medical image inside the framework. Although downstream code works with tensors, `Image` simplifies loading, visualization, and saving.

Supported file formats: `.jpg` / `.jpeg`, `.png`, `.tif` / `.tiff`, and `.dcm` (DICOM). By default, files are validated against allowed MIME types — invalid files are rejected when `strict=True`.

Main entry points:

* `Image.from_path(path)` — load from disk
* `Image.from_base64(data)` — load from a base64 string
* `image.show()` — display via matplotlib (works reliably in Jupyter, unlike `cv2.imshow`)
* `image.save(path)` — write to file

```python
from hyppopipe.data.image import Image

image = Image.from_path("fundus.tif")
image.show()
image.save("preview.png")
```

## Documentation

::: hyppopipe.data.image
