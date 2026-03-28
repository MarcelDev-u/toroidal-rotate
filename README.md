# toroidal-rotate

`toroidal-rotate` is a small NumPy utility for deterministic, reversible toroidal pseudo-rotation of raster images.

It keeps the same array shape, preserves pixel values exactly, and uses wraparound boundaries. It does not interpolate, blur, or average values.

## What it is

The transform is built from three integer shears on a toroidal grid:

- deterministic
- reversible
- same output shape as input
- exact value preservation
- works on grayscale `(H, W)` and channels-last `(H, W, C)` arrays

## What it is not

This is not a true Euclidean raster rotation. It is a reversible permutation-like transform that looks rotation-like. Repeatedly applying many small angles is not equivalent to exact cumulative rotation.

## Installation

```bash
pip install playfull-toroidal-rotate
```

## Usage

### Example 1: grayscale round-trip

```python
import numpy as np

from toroidal_rotate import toroidal_rotate, toroidal_rotate_inverse

image = np.arange(8 * 8).reshape(8, 8)

rotated = toroidal_rotate(image, 24)
restored = toroidal_rotate_inverse(rotated, 24)

print(np.array_equal(image, restored))  # True
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

## Build and publish

```bash
python -m pip install --upgrade build twine
python -m build
twine upload --repository testpypi dist/*
# after checking it works
twine upload dist/*
```

## License

MIT
