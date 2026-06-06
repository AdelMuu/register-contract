from dataclasses import dataclass
from typing import Optional, Tuple

from register_contract.checks import CheckResult
from register_contract.registers import RegisterTile
from register_contract.types import Location, Producer, RegisterRole, TileLayout


@dataclass(frozen=True)
class Sm90MmaConfig:
    tile_shape_mnk: Tuple[int, int, int]
    atom_layout_mn: Tuple[int, int]

    def validate(self) -> CheckResult:
        tile_m, tile_n, tile_k = self.tile_shape_mnk
        atom_m, atom_n = self.atom_layout_mn

        if atom_m not in (1, 2, 3):
            return CheckResult.fail(
                f"invalid atom_layout_mn[0]: expected one of 1, 2, 3, got {atom_m}"
            )

        if atom_n not in (1, 2):
            return CheckResult.fail(
                f"invalid atom_layout_mn[1]: expected one of 1, 2, got {atom_n}"
            )

        if atom_m > 1:
            required_m = atom_m * 64
            if tile_m % required_m != 0:
                return CheckResult.fail(
                    f"invalid tile M: expected tile M divisible by {required_m}, got {tile_m}"
                )

        if atom_n > 1:
            required_n = atom_n * 32
            if tile_n % required_n != 0:
                return CheckResult.fail(
                    f"invalid tile N: expected tile N divisible by {required_n}, got {tile_n}"
                )

        if tile_k % 16 != 0:
            return CheckResult.fail(
                f"invalid tile K: expected tile K divisible by 16, got {tile_k}"
            )

        return CheckResult.pass_("SM90 MMA config is valid")


@dataclass(frozen=True)
class GemmAccumulatorContract:
    rows: int
    cols: int
    left_source: str
    right_source: str
    dtype: str = "fp32"
    layout: TileLayout = TileLayout.ROW
    role: Optional[RegisterRole] = RegisterRole.GEMM_ACCUMULATOR

    def apply(self, accumulator: RegisterTile) -> CheckResult:
        if accumulator.producer is not Producer.GEMM:
            return CheckResult.fail(
                f"expected accumulator producer {Producer.GEMM.value}, got {accumulator.producer}"
            )

        if self.role is not None and accumulator.role is not self.role:
            return CheckResult.fail(
                "expected accumulator role "
                f"{self.role.value}, got {accumulator.role.value}"
            )

        if accumulator.location is not Location.REGISTER:
            return CheckResult.fail(
                f"expected accumulator location {Location.REGISTER.value}, got {accumulator.location.value}"
            )

        if accumulator.rows != self.rows or accumulator.cols != self.cols:
            return CheckResult.fail(
                "accumulator shape mismatch: "
                f"expected ({self.rows}, {self.cols}), "
                f"got ({accumulator.rows}, {accumulator.cols})"
            )

        if accumulator.dtype != self.dtype:
            return CheckResult.fail(
                f"unsupported accumulator dtype: expected {self.dtype}, got {accumulator.dtype}"
            )

        if accumulator.layout is not self.layout:
            return CheckResult.fail(
                "accumulator layout mismatch: "
                f"expected {self.layout.value}, got {accumulator.layout.value}"
            )

        expected_sources = (self.left_source, self.right_source)
        if accumulator.sources != expected_sources:
            return CheckResult.fail(
                f"accumulator source mismatch: expected {expected_sources}, got {accumulator.sources}"
            )

        return CheckResult.pass_("GEMM accumulator contract is valid")


@dataclass(frozen=True)
class SwigluEpilogueContract:
    output_dtype: str = "bf16"

    def apply(self, gate: RegisterTile, value: RegisterTile) -> CheckResult:
        if gate.producer is not Producer.GEMM:
            return CheckResult.fail(
                f"expected gate producer {Producer.GEMM.value}, got {gate.producer}"
            )

        if value.producer is not Producer.GEMM:
            return CheckResult.fail(
                f"expected value producer {Producer.GEMM.value}, got {value.producer}"
            )

        if gate.role is not RegisterRole.SWIGLU_GATE:
            return CheckResult.fail(
                f"expected gate role {RegisterRole.SWIGLU_GATE.value}, got {gate.role.value}"
            )

        if value.role is not RegisterRole.SWIGLU_VALUE:
            return CheckResult.fail(
                f"expected value role {RegisterRole.SWIGLU_VALUE.value}, got {value.role.value}"
            )

        if gate.location is not Location.REGISTER:
            return CheckResult.fail(
                f"expected gate location {Location.REGISTER.value}, got {gate.location.value}"
            )

        if value.location is not Location.REGISTER:
            return CheckResult.fail(
                f"expected value location {Location.REGISTER.value}, got {value.location.value}"
            )

        if gate.rows != value.rows or gate.cols != value.cols:
            return CheckResult.fail(
                f"shape mismatch: gate has shape ({gate.rows}, {gate.cols}), value has shape ({value.rows}, {value.cols})"
            )

        if gate.layout != value.layout:
            return CheckResult.fail(
                f"layout mismatch: gate has layout {gate.layout.value}, value has layout {value.layout.value}"
            )

        if gate.dtype != value.dtype:
            return CheckResult.fail(
                f"dtype mismatch: gate has dtype {gate.dtype}, value has dtype {value.dtype}"
            )

        if gate.layout is not TileLayout.ROW:
            return CheckResult.fail(
                f"unsupported SwiGLU epilogue layout: expected row, got {gate.layout.value}"
            )

        if len(gate.sources) != 2:
            return CheckResult.fail(
                f"expected gate sources (A, B), got {gate.sources}"
            )

        if len(value.sources) != 2:
            return CheckResult.fail(
                f"expected value sources (A, B1), got {value.sources}"
            )

        if gate.sources[0] != value.sources[0]:
            return CheckResult.fail(
                f"source mismatch: gate uses {gate.sources[0]}, value uses {value.sources[0]}"
            )

        if gate.sources[1] == value.sources[1]:
            return CheckResult.fail(
                f"expected distinct gate/value weights, got {gate.sources[1]}"
            )

        return CheckResult.pass_("SwiGLU epilogue can consume GEMM-backed gate and value tiles")
