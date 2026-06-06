import pytest

from register_contract.codegen import generate_python_wrapper
from register_contract.lowering import (
    lower_gemm_to_gemm4_sm90,
    lower_swiglu_to_swiglu3_sm90,
)
from register_contract.specs import ComponentConfig, GemmSpec, SwigluSpec, TensorSpec


def tensor(name, shape, dtype="bf16", layout="row_major"):
    return TensorSpec(name=name, shape=shape, dtype=dtype, layout=layout)


def valid_config():
    return ComponentConfig(
        tile_shape_mnk=(128, 128, 64),
        epi_tile_mn=(128, 32),
        cluster_shape_mnk=(2, 1, 1),
        atom_layout_mn=(2, 1),
        ab_stage=3,
        epi_stage=2,
        reuse_ab=False,
        is_persistent=True,
        gemm_n_prologue=0,
        target_arch="sm90",
    )


def valid_gemm_spec():
    return GemmSpec(
        A=tensor("A", (4096, 7168)),
        B=tensor("B", (1536, 7168)),
        C=tensor("C", (4096, 1536)),
    )


def valid_swiglu_spec():
    return SwigluSpec(
        A=tensor("A", (4096, 7168)),
        B=tensor("B", (1536, 7168)),
        B1=tensor("B1", (1536, 7168)),
        C=tensor("C", (4096, 1536)),
    )


def require_plan(result):
    assert result.ok, result.reason
    assert result.plan is not None
    return result.plan


def test_valid_gemm_spec_generates_gemm4_wrapper():
    plan = require_plan(lower_gemm_to_gemm4_sm90(valid_gemm_spec(), valid_config()))

    generated = generate_python_wrapper(plan)

    assert generated.filename == "generated/gemm_gemm4sm90.py"
    assert "from gemm import Gemm4SM90" in generated.source
    assert "tile_shape_mnk=(128, 128, 64)" in generated.source
    assert "epi_tile_mn=(128, 32)" in generated.source
    assert "cluster_shape_mnk=(2, 1, 1)" in generated.source
    assert "atom_layout_mn=(2, 1)" in generated.source
    assert "ab_stage=3" in generated.source
    assert "epi_stage=2" in generated.source
    assert "reuse_ab=" not in generated.source
    assert "is_persistent=True" in generated.source
    assert "gemm_n_prologue=0" in generated.source


def test_valid_swiglu_spec_generates_swiglu3_wrapper():
    plan = require_plan(lower_swiglu_to_swiglu3_sm90(valid_swiglu_spec(), valid_config()))

    generated = generate_python_wrapper(plan)

    assert generated.filename == "generated/swiglu_swiglu3sm90.py"
    assert "from swiglu import Swiglu3SM90" in generated.source
    assert "INPUTS = ('A', 'B', 'B1')" in generated.source
    assert "OUTPUT = 'C'" in generated.source
    assert "reuse_ab=False" in generated.source


def test_invalid_gemm_with_transposed_b_fails_before_codegen():
    spec = GemmSpec(
        A=tensor("A", (4096, 7168)),
        B=tensor("B", (7168, 1536)),
        C=tensor("C", (4096, 1536)),
    )

    result = lower_gemm_to_gemm4_sm90(spec, valid_config())

    assert not result.ok
    assert result.plan is None
    assert "invalid B shape for A @ B.T" in result.reason


def test_invalid_swiglu_with_mismatched_b1_shape_fails_before_codegen():
    spec = SwigluSpec(
        A=tensor("A", (4096, 7168)),
        B=tensor("B", (1536, 7168)),
        B1=tensor("B1", (2048, 7168)),
        C=tensor("C", (4096, 1536)),
    )

    result = lower_swiglu_to_swiglu3_sm90(spec, valid_config())

    assert not result.ok
    assert result.plan is None
    assert "invalid B1 shape" in result.reason


def test_invalid_swiglu_with_same_b_and_b1_tensor_fails_before_codegen():
    spec = SwigluSpec(
        A=tensor("A", (4096, 7168)),
        B=tensor("B", (1536, 7168)),
        B1=tensor("B", (1536, 7168)),
        C=tensor("C", (4096, 1536)),
    )

    result = lower_swiglu_to_swiglu3_sm90(spec, valid_config())

    assert not result.ok
    assert result.plan is None
    assert "B and B1 must be distinct" in result.reason


@pytest.mark.parametrize(
    "lowering,spec",
    [
        (lower_gemm_to_gemm4_sm90, valid_gemm_spec()),
        (lower_swiglu_to_swiglu3_sm90, valid_swiglu_spec()),
    ],
)
def test_invalid_component_config_fails_before_codegen(lowering, spec):
    config = ComponentConfig(
        tile_shape_mnk=(96, 256, 64),
        epi_tile_mn=(96, 32),
        cluster_shape_mnk=(2, 1, 1),
        atom_layout_mn=(2, 1),
        ab_stage=3,
        epi_stage=2,
        reuse_ab=False,
        is_persistent=True,
        gemm_n_prologue=0,
        target_arch="sm90",
    )

    result = lowering(spec, config)

    assert not result.ok
    assert result.plan is None
    assert "invalid tile M" in result.reason


def test_invalid_gemm_input_layout_fails_before_codegen():
    spec = GemmSpec(
        A=tensor("A", (4096, 7168), layout="col_major"),
        B=tensor("B", (1536, 7168)),
        C=tensor("C", (4096, 1536)),
    )

    result = lower_gemm_to_gemm4_sm90(spec, valid_config())

    assert not result.ok
    assert result.plan is None
    assert "unsupported A layout" in result.reason


def test_invalid_epilogue_tile_fails_before_codegen():
    config = ComponentConfig(
        tile_shape_mnk=(128, 128, 64),
        epi_tile_mn=(96, 32),
        cluster_shape_mnk=(2, 1, 1),
        atom_layout_mn=(2, 1),
        ab_stage=3,
        epi_stage=2,
        reuse_ab=False,
        is_persistent=True,
        gemm_n_prologue=0,
        target_arch="sm90",
    )

    result = lower_gemm_to_gemm4_sm90(valid_gemm_spec(), config)

    assert not result.ok
    assert result.plan is None
    assert "epilogue tile" in result.reason
    assert "evenly divide" in result.reason
