from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from register_contract.codegen import generate_python_wrapper
from register_contract.lowering import (
    lower_gemm_to_gemm4_sm90,
    lower_swiglu_to_swiglu3_sm90,
)
from register_contract.specs import ComponentConfig, GemmSpec, SwigluSpec, TensorSpec


def tensor(name, shape, dtype="bf16", layout="row_major"):
    return TensorSpec(name=name, shape=shape, dtype=dtype, layout=layout)


def checked_source(result):
    if not result.ok:
        raise ValueError(result.reason)
    return generate_python_wrapper(result.plan)


def write_generated(generated):
    output_path = REPO_ROOT / generated.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated.source)
    return output_path


def main():
    config = ComponentConfig(
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

    gemm = GemmSpec(
        A=tensor("A", (4096, 7168)),
        B=tensor("B", (1536, 7168)),
        C=tensor("C", (4096, 1536)),
    )
    swiglu = SwigluSpec(
        A=tensor("A", (4096, 7168)),
        B=tensor("B", (1536, 7168)),
        B1=tensor("B1", (1536, 7168)),
        C=tensor("C", (4096, 1536)),
    )

    generated = [
        checked_source(lower_gemm_to_gemm4_sm90(gemm, config)),
        checked_source(lower_swiglu_to_swiglu3_sm90(swiglu, config)),
    ]

    for source in generated:
        print(write_generated(source))


if __name__ == "__main__":
    main()
