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


def _roll_rows(image: ArrayLikeImage, shifts: NDArray[np.int64]) -> ArrayLikeImage:
    height = image.shape[0]
    if shifts.shape != (height,):
        raise ToroidalRotationError(
            f"Expected row shifts of shape {(height,)}, got {shifts.shape}."
        )

    out = np.empty_like(image)
    for row_index in range(height):
        out[row_index] = np.roll(image[row_index], int(shifts[row_index]), axis=0)
    return out


def _roll_columns(image: ArrayLikeImage, shifts: NDArray[np.int64]) -> ArrayLikeImage:
    width = image.shape[1]
    if shifts.shape != (width,):
        raise ToroidalRotationError(
            f"Expected column shifts of shape {(width,)}, got {shifts.shape}."
        )

    out = np.empty_like(image)
    for column_index in range(width):
        out[:, column_index] = np.roll(image[:, column_index], int(shifts[column_index]), axis=0)
    return out


def _compute_shear_shifts(
    shape: tuple[int, int],
    angle_degrees: float,
    rounding: RoundingMode,
    center: CenterMode,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    height, width = shape

    theta = np.deg2rad(angle_degrees)
    a = -np.tan(theta / 2.0)
    b = np.sin(theta)

    y_coords = _centered_coordinates(height, center)
    x_coords = _centered_coordinates(width, center)

    row_shifts_first = _quantize(a * y_coords, rounding)
    column_shifts = _quantize(b * x_coords, rounding)
    row_shifts_second = _quantize(a * y_coords, rounding)

    return row_shifts_first, column_shifts, row_shifts_second


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
    Apply deterministic reversible toroidal pseudo-rotation to an image.

    This is a same-shape, interpolation-free pixel permutation built from
    three integer shears on a toroidal grid. It is reversible, but it is not
    a true Euclidean raster rotation.
    """
    _validate_image(image)
    _validate_rounding(rounding)
    _validate_center(center)

    if not np.isfinite(angle_degrees):
        raise ToroidalRotationError(
            f"'angle_degrees' must be finite, got {angle_degrees!r}."
        )

    working = image.copy() if copy else image

    row_1, col, row_2 = _compute_shear_shifts(
        shape=working.shape[:2],
        angle_degrees=angle_degrees,
        rounding=rounding,
        center=center,
    )

    out = _roll_rows(working, row_1)
    out = _roll_columns(out, col)
    out = _roll_rows(out, row_2)
    return out


def toroidal_rotate_inverse(
    image: ArrayLikeImage,
    angle_degrees: float,
    *,
    rounding: RoundingMode = "nearest",
    center: CenterMode = "pixel_center",
    copy: bool = True,
) -> ArrayLikeImage:
    """Apply the exact inverse of `toroidal_rotate`."""
    _validate_image(image)
    _validate_rounding(rounding)
    _validate_center(center)

    if not np.isfinite(angle_degrees):
        raise ToroidalRotationError(
            f"'angle_degrees' must be finite, got {angle_degrees!r}."
        )

    working = image.copy() if copy else image

    row_1, col, row_2 = _compute_shear_shifts(
        shape=working.shape[:2],
        angle_degrees=angle_degrees,
        rounding=rounding,
        center=center,
    )

    out = _roll_rows(working, -row_2)
    out = _roll_columns(out, -col)
    out = _roll_rows(out, -row_1)
    return out


def toroidal_rotate_many(
    image: ArrayLikeImage,
    angles_degrees: Iterable[float],
    *,
    rounding: RoundingMode = "nearest",
    center: CenterMode = "pixel_center",
) -> ArrayLikeImage:
    """
    Apply multiple toroidal pseudo-rotations sequentially.

    This remains stepwise reversible, but repeated small angles should not be
    treated as numerically faithful cumulative rotation.
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
