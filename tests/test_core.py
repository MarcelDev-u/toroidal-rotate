import numpy as np
import pytest

from toroidal_rotate import (
    ToroidalRotationError,
    toroidal_rotate,
    toroidal_rotate_inverse,
    toroidal_rotate_many,
)


def test_grayscale_roundtrip():
    image = np.arange(8 * 8).reshape(8, 8)
    rotated = toroidal_rotate(image, 24)
    restored = toroidal_rotate_inverse(rotated, 24)
    assert np.array_equal(image, restored)


def test_rgb_shape_and_dtype_preserved():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 255
    rotated = toroidal_rotate(
        image,
        angle_degrees=24,
        rounding="nearest",
        center="pixel_center",
    )
    assert rotated.shape == image.shape
    assert rotated.dtype == image.dtype
    assert np.array_equal(np.sort(rotated.reshape(-1, 3), axis=0), np.sort(image.reshape(-1, 3), axis=0))


def test_many_matches_manual_sequence():
    image = np.arange(6 * 6).reshape(6, 6)
    out_many = toroidal_rotate_many(image, [10, -5, 17])
    out_manual = toroidal_rotate(toroidal_rotate(toroidal_rotate(image, 10), -5), 17)
    assert np.array_equal(out_many, out_manual)


def test_invalid_rounding_raises():
    image = np.arange(9).reshape(3, 3)
    with pytest.raises(ToroidalRotationError):
        toroidal_rotate(image, 24, rounding="bad")  # type: ignore[arg-type]


def test_non_finite_angle_raises():
    image = np.arange(9).reshape(3, 3)
    with pytest.raises(ToroidalRotationError):
        toroidal_rotate(image, float("nan"))
