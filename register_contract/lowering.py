from dataclasses import dataclass
from typing import Optional, Tuple

from register_contract.checks import CheckResult
from register_contract.components import (
    GemmAccumulatorContract,
    Sm90MmaConfig,
    SwigluEpilogueContract,
)
from register_contract.registers import RegisterTile
from register_contract.specs import ComponentConfig, GemmSpec, SwigluSpec, TensorSpec
from register_contract.types import Location, Producer, RegisterRole, TileLayout


@dataclass(frozen=True)
class LoweringPlan:
    component: str
    import_module: str
    import_symbol: str
    config: ComponentConfig
    inputs: Tuple[str, ...]
    output: str


@dataclass(frozen=True)
class LoweringResult:
    ok: bool
    reason: str
    plan: Optional[LoweringPlan] = None

    @staticmethod
    def pass_(plan: LoweringPlan, reason: str = "lowering contract satisfied"):
        return LoweringResult(True, reason, plan)

    @staticmethod
    def fail(reason: str):
        return LoweringResult(False, reason, None)


def lower_gemm_to_gemm4_sm90(spec: GemmSpec, config: ComponentConfig) -> LoweringResult:
    common = _validate_common_sm90_config(config)
    if not common.ok:
        return LoweringResult.fail(common.reason)

    shape_result = _validate_gemm_shapes(spec.A, spec.B, spec.C)
    if not shape_result.ok:
        return LoweringResult.fail(shape_result.reason)

    dtype_result = _validate_bf16_gemm_dtypes(spec.A, spec.B, spec.C)
    if not dtype_result.ok:
        return LoweringResult.fail(dtype_result.reason)

    layout_result = _validate_gemm_tensor_layouts(spec.A, spec.B, spec.C)
    if not layout_result.ok:
        return LoweringResult.fail(layout_result.reason)

    accumulator = _make_gemm_accumulator(spec, config)
    accumulator_result = GemmAccumulatorContract(
        rows=config.tile_shape_mnk[0],
        cols=config.tile_shape_mnk[1],
        left_source=spec.A.name,
        right_source=spec.B.name,
    ).apply(accumulator)
    if not accumulator_result.ok:
        return LoweringResult.fail(accumulator_result.reason)

    plan = LoweringPlan(
        component="Gemm4SM90",
        import_module="gemm",
        import_symbol="Gemm4SM90",
        config=config,
        inputs=(spec.A.name, spec.B.name),
        output=spec.C.name,
    )
    return LoweringResult.pass_(plan, "GemmSpec can lower to Gemm4SM90")


def lower_swiglu_to_swiglu3_sm90(spec: SwigluSpec, config: ComponentConfig) -> LoweringResult:
    common = _validate_common_sm90_config(config)
    if not common.ok:
        return LoweringResult.fail(common.reason)

    shape_result = _validate_swiglu_shapes(spec)
    if not shape_result.ok:
        return LoweringResult.fail(shape_result.reason)

    dtype_result = _validate_bf16_swiglu_dtypes(spec)
    if not dtype_result.ok:
        return LoweringResult.fail(dtype_result.reason)

    layout_result = _validate_swiglu_tensor_layouts(spec)
    if not layout_result.ok:
        return LoweringResult.fail(layout_result.reason)

    if spec.B.name == spec.B1.name:
        return LoweringResult.fail(
            f"B and B1 must be distinct tensors, got {spec.B.name}"
        )

    gate, value = _make_swiglu_internal_accumulators(spec, config)
    gate_accumulator_result = GemmAccumulatorContract(
        rows=config.tile_shape_mnk[0],
        cols=config.tile_shape_mnk[1],
        left_source=spec.A.name,
        right_source=spec.B.name,
        role=RegisterRole.SWIGLU_GATE,
    ).apply(gate)
    if not gate_accumulator_result.ok:
        return LoweringResult.fail(gate_accumulator_result.reason)

    value_accumulator_result = GemmAccumulatorContract(
        rows=config.tile_shape_mnk[0],
        cols=config.tile_shape_mnk[1],
        left_source=spec.A.name,
        right_source=spec.B1.name,
        role=RegisterRole.SWIGLU_VALUE,
    ).apply(value)
    if not value_accumulator_result.ok:
        return LoweringResult.fail(value_accumulator_result.reason)

    epilogue_result = SwigluEpilogueContract().apply(gate=gate, value=value)
    if not epilogue_result.ok:
        return LoweringResult.fail(epilogue_result.reason)

    plan = LoweringPlan(
        component="Swiglu3SM90",
        import_module="swiglu",
        import_symbol="Swiglu3SM90",
        config=config,
        inputs=(spec.A.name, spec.B.name, spec.B1.name),
        output=spec.C.name,
    )
    return LoweringResult.pass_(plan, "SwigluSpec can lower to Swiglu3SM90")


def _validate_common_sm90_config(config: ComponentConfig) -> CheckResult:
    if config.target_arch != "sm90":
        return CheckResult.fail(
            f"unsupported target_arch: expected sm90, got {config.target_arch}"
        )

    mma_result = Sm90MmaConfig(
        tile_shape_mn=config.tile_shape_mnk[:2],
        atom_layout_mn=config.atom_layout_mn,
    ).validate()
    if not mma_result.ok:
        return mma_result

    tile_k = config.tile_shape_mnk[2]
    if tile_k % 16 != 0:
        return CheckResult.fail(
            f"invalid tile K: expected tile_shape_mnk[2] divisible by 16, got {tile_k}"
        )

    tile_m, tile_n, _ = config.tile_shape_mnk
    epi_m, epi_n = config.epi_tile_mn
    if epi_m <= 0 or epi_n <= 0:
        return CheckResult.fail(
            f"invalid epilogue tile: expected positive dimensions, got {config.epi_tile_mn}"
        )

    if epi_m > tile_m or epi_n > tile_n:
        return CheckResult.fail(
            "invalid epilogue tile: expected epi_tile_mn to fit inside "
            f"tile_shape_mnk[:2] ({tile_m}, {tile_n}), got {config.epi_tile_mn}"
        )

    if tile_m % epi_m != 0 or tile_n % epi_n != 0:
        return CheckResult.fail(
            "invalid epilogue tile: expected epi_tile_mn to evenly divide "
            f"tile_shape_mnk[:2] ({tile_m}, {tile_n}), got {config.epi_tile_mn}"
        )

    atom_n = config.atom_layout_mn[1]
    if atom_n > 1 and config.epi_tile_mn[1] != tile_n:
        return CheckResult.fail(
            "invalid epilogue tile N: when atom_layout_mn[1] > 1, "
            f"expected epi_tile_mn[1] == tile_shape_mnk[1] ({tile_n}), "
            f"got {config.epi_tile_mn[1]}"
        )

    if config.reuse_ab and config.is_persistent:
        return CheckResult.fail("reuse_ab and is_persistent cannot both be true")

    if config.gemm_n_prologue != 0:
        return CheckResult.fail(
            f"unsupported gemm_n_prologue: expected 0, got {config.gemm_n_prologue}"
        )

    return CheckResult.pass_("common SM90 component config is valid")


def _validate_gemm_shapes(A: TensorSpec, B: TensorSpec, C: TensorSpec) -> CheckResult:
    rank_result = _require_rank2(A, B, C)
    if not rank_result.ok:
        return rank_result

    m, k = A.shape
    c_m, n = C.shape
    if C.shape != (m, n) or c_m != m:
        return CheckResult.fail(
            f"invalid C shape: expected ({m}, N) matching A rows, got {C.shape}"
        )

    expected_b = (n, k)
    if B.shape != expected_b:
        return CheckResult.fail(
            f"invalid B shape for A @ B.T: expected {expected_b}, got {B.shape}"
        )

    return CheckResult.pass_("GEMM shapes are valid")


def _validate_swiglu_shapes(spec: SwigluSpec) -> CheckResult:
    rank_result = _require_rank2(spec.A, spec.B, spec.B1, spec.C)
    if not rank_result.ok:
        return rank_result

    m, k = spec.A.shape
    c_m, n = spec.C.shape
    if c_m != m:
        return CheckResult.fail(
            f"invalid C shape: expected first dimension {m}, got {spec.C.shape}"
        )

    expected_weight = (n, k)
    if spec.B.shape != expected_weight:
        return CheckResult.fail(
            f"invalid B shape for gate = A @ B.T: expected {expected_weight}, got {spec.B.shape}"
        )

    if spec.B1.shape != expected_weight:
        return CheckResult.fail(
            f"invalid B1 shape for value = A @ B1.T: expected {expected_weight}, got {spec.B1.shape}"
        )

    return CheckResult.pass_("SwiGLU shapes are valid")


def _validate_bf16_gemm_dtypes(A: TensorSpec, B: TensorSpec, C: TensorSpec) -> CheckResult:
    if _dtype(A) != _dtype(B):
        return CheckResult.fail(f"A/B element types must match: got {A.dtype} and {B.dtype}")

    if _dtype(A) != "bf16":
        return CheckResult.fail(f"supported first pass dtype is bf16, got {A.dtype}")

    if _dtype(C) != "bf16":
        return CheckResult.fail(f"supported output dtype is bf16, got {C.dtype}")

    return CheckResult.pass_("GEMM dtypes are valid")


def _validate_gemm_tensor_layouts(A: TensorSpec, B: TensorSpec, C: TensorSpec) -> CheckResult:
    for tensor in (A, B, C):
        if tensor.layout != "row_major":
            return CheckResult.fail(
                f"unsupported {tensor.name} layout: expected row_major, got {tensor.layout}"
            )
    return CheckResult.pass_("GEMM tensor layouts are valid")


def _validate_bf16_swiglu_dtypes(spec: SwigluSpec) -> CheckResult:
    dtypes = {_dtype(spec.A), _dtype(spec.B), _dtype(spec.B1)}
    if len(dtypes) != 1:
        return CheckResult.fail(
            "A, B, and B1 element types must match: "
            f"got A={spec.A.dtype}, B={spec.B.dtype}, B1={spec.B1.dtype}"
        )

    dtype = dtypes.pop()
    if dtype != "bf16":
        return CheckResult.fail(f"supported SwiGLU input dtype is bf16, got {dtype}")

    if _dtype(spec.C) != "bf16":
        return CheckResult.fail(f"supported output dtype is bf16, got {spec.C.dtype}")

    return CheckResult.pass_("SwiGLU dtypes are valid")


def _validate_swiglu_tensor_layouts(spec: SwigluSpec) -> CheckResult:
    for tensor in (spec.A, spec.B, spec.B1, spec.C):
        if tensor.layout != "row_major":
            return CheckResult.fail(
                f"unsupported {tensor.name} layout: expected row_major, got {tensor.layout}"
            )
    return CheckResult.pass_("SwiGLU tensor layouts are valid")


def _make_gemm_accumulator(spec: GemmSpec, config: ComponentConfig) -> RegisterTile:
    rows, cols, _ = config.tile_shape_mnk
    return RegisterTile(
        name=f"{spec.C.name}_accum",
        dtype="fp32",
        rows=rows,
        cols=cols,
        layout=TileLayout.ROW,
        role=RegisterRole.GEMM_ACCUMULATOR,
        producer=Producer.GEMM,
        sources=(spec.A.name, spec.B.name),
        location=Location.REGISTER,
    )


def _make_swiglu_internal_accumulators(
    spec: SwigluSpec,
    config: ComponentConfig,
) -> Tuple[RegisterTile, RegisterTile]:
    rows, cols, _ = config.tile_shape_mnk
    gate = RegisterTile(
        name="gate_accum",
        dtype="fp32",
        rows=rows,
        cols=cols,
        layout=TileLayout.ROW,
        role=RegisterRole.SWIGLU_GATE,
        producer=Producer.GEMM,
        sources=(spec.A.name, spec.B.name),
        location=Location.REGISTER,
    )
    value = RegisterTile(
        name="value_accum",
        dtype="fp32",
        rows=rows,
        cols=cols,
        layout=TileLayout.ROW,
        role=RegisterRole.SWIGLU_VALUE,
        producer=Producer.GEMM,
        sources=(spec.A.name, spec.B1.name),
        location=Location.REGISTER,
    )
    return gate, value


def _require_rank2(*tensors: TensorSpec) -> CheckResult:
    for tensor in tensors:
        if len(tensor.shape) != 2:
            return CheckResult.fail(
                f"{tensor.name} must be rank-2, got shape {tensor.shape}"
            )
    return CheckResult.pass_("all tensors are rank-2")


def _dtype(tensor: TensorSpec) -> str:
    return tensor.dtype.lower()


lower_gemm_to_gemm1_sm90 = lower_gemm_to_gemm4_sm90
lower_swiglu_to_swiglu_sm90 = lower_swiglu_to_swiglu3_sm90
