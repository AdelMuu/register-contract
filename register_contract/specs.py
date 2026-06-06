from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    layout: str


@dataclass(frozen=True)
class GemmSpec:
    A: TensorSpec
    B: TensorSpec
    C: TensorSpec


@dataclass(frozen=True)
class SwigluSpec:
    A: TensorSpec
    B: TensorSpec
    B1: TensorSpec
    C: TensorSpec


@dataclass(frozen=True)
class ComponentConfig:
    tile_shape_mnk: Tuple[int, int, int]
    epi_tile_mn: Tuple[int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    atom_layout_mn: Tuple[int, int]
    ab_stage: int
    epi_stage: int
    reuse_ab: bool
    is_persistent: bool
    gemm_n_prologue: int
    target_arch: str = "sm90"
