from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, overload

import numpy as np
from numpy.typing import NDArray


ArrayLikeImage = NDArray[np.generic]
RoundingMode = Literal["nearest", "floor", "ceil"]
CenterMode = Literal["pixel_center", "origin"]


class ToroidalRotationError(ValueError):
    """Raised when toroidal rotation inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ToroidalRotationSpec:
    """Configuration for deterministic toroidal pseudo-rotation."""

    angle_degrees: float
    rounding: RoundingMode = "nearest"
    center: CenterMode = "pixel_center"


def _validate_image(image: ArrayLikeImage) -> None:
    if not isinstance(image, np.ndarray):
        raise ToroidalRotationError(
            f"Expected 'image' to be a numpy.ndarray, got {type(image)!r}."
        )
    if image.ndim not in (2, 3):
        raise ToroidalRotationError(
            f"Expected image with 2 or 3 dimensions, got shape {image.shape}."
        )
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ToroidalRotationError(
            f"Image spatial dimensions must be positive, got shape {image.shape}."
        )


def _validate_rounding(rounding: str) -> None:
    allowed = {"nearest", "floor", "ceil"}
    if rounding not in allowed:
        raise ToroidalRotationError(
            f"Unsupported rounding mode {rounding!r}. Expected one of {sorted(allowed)!r}."
        )


def _validate_center(center: str) -> None:
    allowed = {"pixel_center", "origin"}
    if center not in allowed:
        raise ToroidalRotationError(
            f"Unsupported center mode {center!r}. Expected one of {sorted(allowed)!r}."
        )


def _centered_coordinates(length: int, center: CenterMode) -> NDArray[np.float64]:
    if center == "pixel_center":
        return np.arange(length, dtype=np.float64) - (length - 1) / 2.0
    if center == "origin":
        return np.arange(length, dtype=np.float64)
    raise ToroidalRotationError(f"Internal error: unsupported center mode {center!r}.")


def _quantize(values: NDArray[np.float64], mode: RoundingMode) -> NDArray[np.int64]:
    if mode == "nearest":
        return np.rint(values).astype(np.int64)
    if mode == "floor":
        return np.floor(values).astype(np.int64)
    if mode == "ceil":
        return np.ceil(values).astype(np.int64)
    raise ToroidalRotationError(f"Internal error: unsupported rounding mode {mode!r}.")


def _compute_wrapped_source_indices(
    shape: tuple[int, int],
    angle_degrees: float,
    rounding: RoundingMode,
    center: CenterMode,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    height, width = shape

    theta = np.deg2rad(angle_degrees)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    y_coords = _centered_coordinates(height, center)[:, None]
    x_coords = _centered_coordinates(width, center)[None, :]

    # Inverse-map from output grid to input grid so the result stays periodic
    # under the same square tiling basis.
    src_x = cos_theta * x_coords + sin_theta * y_coords
    src_y = -sin_theta * x_coords + cos_theta * y_coords

    if center == "pixel_center":
        src_x = src_x + (width - 1) / 2.0
        src_y = src_y + (height - 1) / 2.0

    src_x_idx = _quantize(src_x, rounding) % width
    src_y_idx = _quantize(src_y, rounding) % height
    return src_y_idx, src_x_idx


def _sample_wrapped(image: ArrayLikeImage, y_idx: NDArray[np.int64], x_idx: NDArray[np.int64]) -> ArrayLikeImage:
    if image.ndim == 2:
        return image[y_idx, x_idx]
    return image[y_idx, x_idx, :]


@overload
def toroidal_rotate(
    image: NDArray[np.generic],
    angle_degrees: float,
    *,
    rounding: RoundingMode = "nearest",
    center: CenterMode = "pixel_center",
    copy: bool = True,
) -> NDArray[np.generic]:
    ...


def toroidal_rotate(
    image: ArrayLikeImage,
    angle_degrees: float,
    *,
    rounding: RoundingMode = "nearest",
    center: CenterMode = "pixel_center",
    copy: bool = True,
) -> ArrayLikeImage:
    """
    Apply wrapped periodic raster rotation to an image.

    This performs inverse-mapped sampling on a periodic image domain, so the
    result remains compatible with square tiling under the same wrap basis.
    It is still a raster approximation because coordinates are quantized.
    """
    _validate_image(image)
    _validate_rounding(rounding)
    _validate_center(center)

    if not np.isfinite(angle_degrees):
        raise ToroidalRotationError(
            f"'angle_degrees' must be finite, got {angle_degrees!r}."
        )

    working = image.copy() if copy else image

    y_idx, x_idx = _compute_wrapped_source_indices(
        shape=working.shape[:2],
        angle_degrees=angle_degrees,
        rounding=rounding,
        center=center,
    )
    return _sample_wrapped(working, y_idx, x_idx)


def toroidal_rotate_inverse(
    image: ArrayLikeImage,
    angle_degrees: float,
    *,
    rounding: RoundingMode = "nearest",
    center: CenterMode = "pixel_center",
    copy: bool = True,
) -> ArrayLikeImage:
    """Apply the wrapped inverse rotation approximation of `toroidal_rotate`."""
    _validate_image(image)
    _validate_rounding(rounding)
    _validate_center(center)

    if not np.isfinite(angle_degrees):
        raise ToroidalRotationError(
            f"'angle_degrees' must be finite, got {angle_degrees!r}."
        )

    working = image.copy() if copy else image

    return toroidal_rotate(
        working,
        -angle_degrees,
        rounding=rounding,
        center=center,
        copy=False,
    )


def toroidal_rotate_many(
    image: ArrayLikeImage,
    angles_degrees: Iterable[float],
    *,
    rounding: RoundingMode = "nearest",
    center: CenterMode = "pixel_center",
) -> ArrayLikeImage:
    """
    Apply multiple toroidal pseudo-rotations sequentially.

    This remains periodic on the wrapped square domain, but repeated small
    angles should not be treated as numerically faithful cumulative rotation.
    """
    _validate_image(image)
    _validate_rounding(rounding)
    _validate_center(center)

    out = image.copy()
    for index, angle in enumerate(angles_degrees):
        if not np.isfinite(angle):
            raise ToroidalRotationError(
                f"Encountered non-finite angle at position {index}: {angle!r}."
            )
        out = toroidal_rotate(out, angle, rounding=rounding, center=center, copy=False)
    return out
