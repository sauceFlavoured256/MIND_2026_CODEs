"""Minimal single receding-motion detector extracted from RSNN_individual.

This file intentionally keeps the architecture, not the full implementation.
All tunable parameters are left as ``None`` so that they can be filled from an
experiment configuration later.  The detector is designed for one receptive
field (one 148 x 148 cell in the original experiments), not for a population
or a spatial grid of detectors.

Simplified signal path
----------------------
frame pair -> retina difference -> ON/OFF half-wave channels
           -> reversed T4/T5 inward-motion correlations
           -> directional pooling -> opposite-direction gate
           -> membrane potential -> normalised receding response

The reversed T4/T5 branch is the important selectivity mechanism here:
for a shrinking object, the top edge moves down, the bottom edge moves up,
the left edge moves right, and the right edge moves left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np


Array = np.ndarray
DirectionMaps = Dict[str, Array]
DirectionValues = Dict[str, float]


@dataclass
class RecedingDetectorParameters:
    """Parameters deliberately left undefined for later calibration."""

    sample_interval: Optional[int] = None
    retina_kernel: Optional[Array] = None
    half_wave_threshold: Optional[float] = None
    bias_distance: Optional[int] = None
    correlation_normaliser: Optional[float] = None

    # One mask for each inward edge motion: top, bottom, left, right.
    receptive_field_masks: Optional[Mapping[str, Array]] = None

    on_weight: Optional[float] = None
    off_weight: Optional[float] = None
    directional_threshold: Optional[float] = None
    horizontal_gate_threshold: Optional[float] = None
    vertical_gate_threshold: Optional[float] = None

    membrane_decay: Optional[float] = None
    membrane_threshold: Optional[float] = None
    output_normaliser: Optional[float] = None


@dataclass
class RecedingDetectorState:
    """Temporal state for one detector."""

    frame_index: int = 0
    previous_gray: Optional[Array] = None
    previous_on: Optional[Array] = None
    previous_off: Optional[Array] = None
    membrane_potential: float = 0.0
    output_history: list[float] = field(default_factory=list)


def to_grayscale(frame: Array) -> Array:
    """Convert a grayscale or BGR frame to float32 grayscale."""

    frame = np.asarray(frame)
    if frame.ndim == 2:
        return frame.astype(np.float32, copy=False)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        blue, green, red = frame[..., 0], frame[..., 1], frame[..., 2]
        return (0.114 * blue + 0.587 * green + 0.299 * red).astype(np.float32)
    raise ValueError(f"Expected a grayscale or BGR frame, received {frame.shape}.")


def retina_temporal_difference(current_gray: Array, previous_gray: Array) -> Array:
    """Retina stage: current intensity minus the previous intensity."""

    if current_gray.shape != previous_gray.shape:
        raise ValueError("Current and previous frames must have the same shape.")
    return current_gray.astype(np.float32) - previous_gray.astype(np.float32)


def spatial_filter(signal: Array, kernel: Optional[Array]) -> Array:
    """Placeholder for the original retinal/medulla spatial filtering.

    A ``None`` kernel means that filtering has not yet been specified, so the
    signal is returned unchanged.  Replace this function with the desired
    Gaussian or contrast kernel when the model parameters are fixed.
    """

    if kernel is None:
        return signal
    raise NotImplementedError("Spatial convolution is intentionally omitted here.")


def split_on_off(
    temporal_difference: Array,
    threshold: Optional[float],
) -> Tuple[Array, Array]:
    """Lamina stage: split luminance increments (ON) and decrements (OFF)."""

    if threshold is None:
        raise ValueError("half_wave_threshold is None.")
    on_signal = np.where(temporal_difference > threshold, temporal_difference, 0.0)
    off_signal = np.where(temporal_difference < -threshold, -temporal_difference, 0.0)
    return on_signal.astype(np.float32), off_signal.astype(np.float32)


def shift_for_motion(signal: Array, direction: str, distance: Optional[int]) -> Array:
    """Shift a current signal so it can be correlated with the previous one."""

    if distance is None:
        raise ValueError("bias_distance is None.")
    if distance <= 0:
        raise ValueError("bias_distance must be positive.")

    shifted = np.zeros_like(signal, dtype=np.float32)
    if direction == "up":
        shifted[distance:, :] = signal[:-distance, :]
    elif direction == "down":
        shifted[:-distance, :] = signal[distance:, :]
    elif direction == "left":
        shifted[:, distance:] = signal[:, :-distance]
    elif direction == "right":
        shifted[:, :-distance] = signal[:, distance:]
    else:
        raise ValueError(f"Unknown direction: {direction!r}")
    return shifted


def t4_t5_motion_correlation(
    previous_signal: Array,
    current_signal: Array,
    direction: str,
    distance: Optional[int],
    normaliser: Optional[float],
) -> Array:
    """Simplified unilateral T4/T5 elementary motion correlation.

    This retains the key operation from the source model:
    ``previous_signal * spatially_shifted(current_signal)``.
    """

    shifted_current = shift_for_motion(current_signal, direction, distance)
    response = previous_signal.astype(np.float32) * shifted_current
    if normaliser is not None:
        response = response / normaliser
    return response


def apply_receptive_field_mask(
    response: Array,
    mask: Optional[Array],
) -> Array:
    """Restrict a directional response to its edge-specific receptive field."""

    if mask is None:
        return response
    if mask.shape != response.shape:
        raise ValueError("Receptive-field mask and response must have equal shapes.")
    return response * (mask > 0)


def reversed_t4_t5_branch(
    current_signal: Array,
    previous_signal: Array,
    params: RecedingDetectorParameters,
) -> DirectionMaps:
    """Compute the four inward edge-motion maps used for shrinkage detection."""

    masks = params.receptive_field_masks or {}
    inward_directions = {
        "top_inward": "down",
        "bottom_inward": "up",
        "left_inward": "right",
        "right_inward": "left",
    }
    responses: DirectionMaps = {}
    for name, direction in inward_directions.items():
        response = t4_t5_motion_correlation(
            previous_signal,
            current_signal,
            direction,
            params.bias_distance,
            params.correlation_normaliser,
        )
        responses[name] = apply_receptive_field_mask(response, masks.get(name))
    return responses


def fuse_on_off_receding_maps(
    on_maps: Mapping[str, Array],
    off_maps: Mapping[str, Array],
    on_weight: Optional[float],
    off_weight: Optional[float],
) -> DirectionMaps:
    """Fuse T4-like ON and T5-like OFF responses for each inward direction."""

    if on_weight is None or off_weight is None:
        raise ValueError("on_weight and off_weight must be specified.")
    return {
        name: on_weight * on_maps[name] + off_weight * off_maps[name]
        for name in on_maps
    }


def pool_directional_responses(
    direction_maps: Mapping[str, Array],
    threshold: Optional[float],
) -> DirectionValues:
    """Collapse each directional map to one non-negative detector input."""

    if threshold is None:
        raise ValueError("directional_threshold is None.")
    pooled = {}
    for name, response in direction_maps.items():
        value = float(np.maximum(response, 0.0).sum())
        pooled[name] = value if value >= threshold else 0.0
    return pooled


def opposite_direction_gate(
    directions: Mapping[str, float],
    horizontal_threshold: Optional[float],
    vertical_threshold: Optional[float],
) -> Tuple[float, Dict[str, float]]:
    """LPLC2 stage: require paired inward motion on opposite sides.

    Top+bottom and left+right are admitted only when the corresponding product
    exceeds its gate threshold.  This mirrors the epsilon fusion logic in the
    source model while making the receding geometry explicit.
    """

    if horizontal_threshold is None or vertical_threshold is None:
        raise ValueError("Motion-opponency gate thresholds are None.")

    top = directions["top_inward"]
    bottom = directions["bottom_inward"]
    left = directions["left_inward"]
    right = directions["right_inward"]

    vertical_axis = top + bottom if top * bottom > horizontal_threshold else 0.0
    horizontal_axis = left + right if left * right > vertical_threshold else 0.0
    total = vertical_axis + horizontal_axis
    return total, {
        "vertical_axis": vertical_axis,
        "horizontal_axis": horizontal_axis,
    }


def update_membrane_potential(
    previous_potential: float,
    detector_input: float,
    decay: Optional[float],
    threshold: Optional[float],
    normaliser: Optional[float],
) -> Tuple[float, float]:
    """Update and normalise the single detector's membrane potential."""

    if decay is None or threshold is None or normaliser is None:
        raise ValueError("Membrane parameters are still None.")
    raw_potential = decay * previous_potential + detector_input / normaliser
    normalised = float(np.tanh(raw_potential))
    if normalised < threshold:
        normalised = 0.0
    return raw_potential, normalised


def single_receding_detector_step(
    frame: Array,
    state: RecedingDetectorState,
    params: RecedingDetectorParameters,
) -> Tuple[Optional[float], RecedingDetectorState, dict]:
    """Process one frame for one receding-motion detector.

    Returns ``None`` during initialisation or on unsampled frames.  Once all
    ``None`` parameters have been filled, this function is the only entry
    point required by a video loop.
    """

    gray = to_grayscale(frame)
    diagnostics: dict = {}

    if state.previous_gray is None:
        state.previous_gray = gray.copy()
        state.frame_index += 1
        return None, state, {"status": "initialised"}

    if params.sample_interval is None:
        raise ValueError("sample_interval is None.")
    if state.frame_index % params.sample_interval != 0:
        state.frame_index += 1
        return None, state, {"status": "skipped"}

    temporal = retina_temporal_difference(gray, state.previous_gray)
    temporal = spatial_filter(temporal, params.retina_kernel)
    on_signal, off_signal = split_on_off(temporal, params.half_wave_threshold)

    if state.previous_on is None or state.previous_off is None:
        state.previous_on = on_signal.copy()
        state.previous_off = off_signal.copy()
        state.previous_gray = gray.copy()
        state.frame_index += 1
        return None, state, {"status": "temporal_state_initialised"}

    on_maps = reversed_t4_t5_branch(on_signal, state.previous_on, params)
    off_maps = reversed_t4_t5_branch(off_signal, state.previous_off, params)
    fused_maps = fuse_on_off_receding_maps(
        on_maps,
        off_maps,
        params.on_weight,
        params.off_weight,
    )
    directional_values = pool_directional_responses(
        fused_maps,
        params.directional_threshold,
    )
    detector_input, gated_axes = opposite_direction_gate(
        directional_values,
        params.horizontal_gate_threshold,
        params.vertical_gate_threshold,
    )
    raw_potential, output = update_membrane_potential(
        state.membrane_potential,
        detector_input,
        params.membrane_decay,
        params.membrane_threshold,
        params.output_normaliser,
    )

    state.previous_gray = gray.copy()
    state.previous_on = on_signal.copy()
    state.previous_off = off_signal.copy()
    state.membrane_potential = raw_potential
    state.output_history.append(output)
    state.frame_index += 1

    diagnostics.update(
        {
            "status": "processed",
            "directional_values": directional_values,
            "gated_axes": gated_axes,
            "detector_input": detector_input,
            "raw_potential": raw_potential,
        }
    )
    return output, state, diagnostics


# The intended starting point.  Fill these fields before attempting to run.
PARAMS = RecedingDetectorParameters()

