from register_contract.components import (
    GemmAccumulatorContract,
    Sm90MmaConfig,
    SwigluEpilogueContract,
)
from register_contract.registers import RegisterTile
from register_contract.types import Location, Producer, RegisterRole, TileLayout


def make_accumulator(
    name: str,
    role: RegisterRole,
    sources: tuple[str, str],
    layout: TileLayout = TileLayout.ROW,
):
    return RegisterTile(
        name=name,
        dtype="fp32",
        rows=128,
        cols=256,
        layout=layout,
        role=role,
        producer=Producer.GEMM,
        sources=sources,
        location=Location.REGISTER,
    )


def test_swiglu_epilogue_accepts_gemm_backed_gate_and_value_tiles():
    gate = make_accumulator(
        name="gate",
        role=RegisterRole.SWIGLU_GATE,
        sources=("A", "B"),
    )
    value = make_accumulator(
        name="value",
        role=RegisterRole.SWIGLU_VALUE,
        sources=("A", "B1"),
    )

    result = SwigluEpilogueContract().apply(gate=gate, value=value)

    assert result.ok


def test_swiglu_epilogue_rejects_layout_mismatch():
    gate = make_accumulator(
        name="gate",
        role=RegisterRole.SWIGLU_GATE,
        sources=("A", "B"),
        layout=TileLayout.ROW,
    )
    value = make_accumulator(
        name="value",
        role=RegisterRole.SWIGLU_VALUE,
        sources=("A", "B1"),
        layout=TileLayout.COL,
    )

    result = SwigluEpilogueContract().apply(gate=gate, value=value)

    assert not result.ok
    assert "layout mismatch" in result.reason


def test_swiglu_epilogue_rejects_swapped_gate_and_value_roles():
    gate = make_accumulator(
        name="gate",
        role=RegisterRole.SWIGLU_VALUE,
        sources=("A", "B"),
    )
    value = make_accumulator(
        name="value",
        role=RegisterRole.SWIGLU_GATE,
        sources=("A", "B1"),
    )

    result = SwigluEpilogueContract().apply(gate=gate, value=value)

    assert not result.ok
    assert "expected gate role" in result.reason


def test_swiglu_epilogue_rejects_different_activation_sources():
    gate = make_accumulator(
        name="gate",
        role=RegisterRole.SWIGLU_GATE,
        sources=("A", "B"),
    )
    value = make_accumulator(
        name="value",
        role=RegisterRole.SWIGLU_VALUE,
        sources=("A2", "B1"),
    )

    result = SwigluEpilogueContract().apply(gate=gate, value=value)

    assert not result.ok
    assert "source mismatch" in result.reason


def test_sm90_mma_config_rejects_invalid_tile_m_for_atom_layout():
    config = Sm90MmaConfig(
        tile_shape_mnk=(96, 256, 64),
        atom_layout_mn=(2, 1),
    )

    result = config.validate()

    assert not result.ok
    assert "tile M" in result.reason
    assert "divisible by 128" in result.reason


def test_sm90_mma_config_rejects_invalid_tile_k():
    config = Sm90MmaConfig(
        tile_shape_mnk=(128, 128, 40),
        atom_layout_mn=(2, 1),
    )

    result = config.validate()

    assert not result.ok
    assert "tile K" in result.reason
    assert "divisible by 16" in result.reason


def test_gemm_accumulator_contract_accepts_gemm_backed_output_tile():
    accumulator = RegisterTile(
        name="C_accum",
        dtype="fp32",
        rows=128,
        cols=128,
        layout=TileLayout.ROW,
        role=RegisterRole.GEMM_ACCUMULATOR,
        producer=Producer.GEMM,
        sources=("A", "B"),
        location=Location.REGISTER,
    )

    result = GemmAccumulatorContract(
        rows=128,
        cols=128,
        left_source="A",
        right_source="B",
    ).apply(accumulator)

    assert result.ok


def test_gemm_accumulator_contract_rejects_wrong_sources():
    accumulator = RegisterTile(
        name="C_accum",
        dtype="fp32",
        rows=128,
        cols=128,
        layout=TileLayout.ROW,
        role=RegisterRole.GEMM_ACCUMULATOR,
        producer=Producer.GEMM,
        sources=("A", "B1"),
        location=Location.REGISTER,
    )

    result = GemmAccumulatorContract(
        rows=128,
        cols=128,
        left_source="A",
        right_source="B",
    ).apply(accumulator)

    assert not result.ok
    assert "source mismatch" in result.reason


def test_gemm_accumulator_contract_rejects_wrong_layout():
    accumulator = RegisterTile(
        name="C_accum",
        dtype="fp32",
        rows=128,
        cols=128,
        layout=TileLayout.COL,
        role=RegisterRole.GEMM_ACCUMULATOR,
        producer=Producer.GEMM,
        sources=("A", "B"),
        location=Location.REGISTER,
    )

    result = GemmAccumulatorContract(
        rows=128,
        cols=128,
        left_source="A",
        right_source="B",
    ).apply(accumulator)

    assert not result.ok
    assert "layout mismatch" in result.reason


def test_gemm_accumulator_contract_accepts_swiglu_gate_role():
    gate = RegisterTile(
        name="gate_accum",
        dtype="fp32",
        rows=128,
        cols=128,
        layout=TileLayout.ROW,
        role=RegisterRole.SWIGLU_GATE,
        producer=Producer.GEMM,
        sources=("A", "B"),
        location=Location.REGISTER,
    )

    result = GemmAccumulatorContract(
        rows=128,
        cols=128,
        left_source="A",
        right_source="B",
        role=RegisterRole.SWIGLU_GATE,
    ).apply(gate)

    assert result.ok


def test_gemm_accumulator_contract_rejects_swiglu_gate_with_value_role():
    gate = RegisterTile(
        name="gate_accum",
        dtype="fp32",
        rows=128,
        cols=128,
        layout=TileLayout.ROW,
        role=RegisterRole.SWIGLU_VALUE,
        producer=Producer.GEMM,
        sources=("A", "B"),
        location=Location.REGISTER,
    )

    result = GemmAccumulatorContract(
        rows=128,
        cols=128,
        left_source="A",
        right_source="B",
        role=RegisterRole.SWIGLU_GATE,
    ).apply(gate)

    assert not result.ok
    assert "expected accumulator role" in result.reason
