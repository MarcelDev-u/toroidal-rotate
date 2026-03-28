# toroidal-rotate

`toroidal-rotate` is a small NumPy utility for periodic raster rotation of toroidally tiling images.

It keeps the same array shape, uses wraparound boundaries, and rotates by inverse-mapped sampling on the periodic image domain.

## What it is

The transform is:

- deterministic
- same output shape as input
- periodic under square tiling
- works on grayscale `(H, W)` and channels-last `(H, W, C)` arrays

## What it is not

This is not an exact Euclidean rotation. It is a periodic raster approximation using wrapped inverse sampling, so inverse rotation is approximate rather than exact and repeated small angles are not numerically exact cumulative rotation.

## Installation

```bash
pip install toroidal-rotate
```

## Showcase

![Toroidal rotate showcase](https://raw.githubusercontent.com/MarcelDev-u/toroidal-rotate/main/docs/showcase.gif)

Direct file: [`docs/showcase.gif`](docs/showcase.gif)

## Usage

### Example 1: grayscale periodic rotation

```python
import numpy as np

from toroidal_rotate import toroidal_rotate

image = np.arange(8 * 8).reshape(8, 8)

rotated = toroidal_rotate(image, 24)
```

### Example 2: RGB image

```python
import numpy as np

from toroidal_rotate import toroidal_rotate

image = np.zeros((64, 64, 3), dtype=np.uint8)
image[..., 0] = 255

rotated = toroidal_rotate(
    image,
    angle_degrees=24,
    rounding="nearest",
    center="pixel_center",
)
```

## API

```python
from toroidal_rotate import (
    ToroidalRotationError,
    ToroidalRotationSpec,
    toroidal_rotate,
    toroidal_rotate_inverse,
    toroidal_rotate_many,
)
```

`toroidal_rotate_inverse` applies the matching wrapped inverse angle, but due to raster quantization it is approximate rather than exact round-trip restoration.

## Build and publish

```bash
python -m pip install --upgrade build twine
python -m build
twine upload --repository testpypi dist/*
# after checking it works
twine upload dist/*
```

To regenerate the demo GIF:

```bash
PYTHONPATH=src python scripts/make_showcase_gif.py
```

## License

MIT
