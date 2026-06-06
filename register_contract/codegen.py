from dataclasses import dataclass

from register_contract.lowering import LoweringPlan


@dataclass(frozen=True)
class GeneratedSource:
    filename: str
    source: str


def generate_python_wrapper(plan: LoweringPlan) -> GeneratedSource:
    filename = f"generated/{plan.import_module}_{plan.import_symbol.lower()}.py"
    config_args = _format_config_args(plan)
    source = f'''"""Generated wrapper for a checked {plan.component} lowering plan."""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FUSED_KERNELS_PYTHON = REPO_ROOT / "third_party" / "fused_kernels_cutedsl_lib" / "python"
FUSED_KERNELS_COMPONENTS = FUSED_KERNELS_PYTHON / "cutedsl_kernels"

sys.path.insert(0, str(FUSED_KERNELS_PYTHON))
sys.path.insert(0, str(FUSED_KERNELS_COMPONENTS))

from {plan.import_module} import {plan.import_symbol}


INPUTS = {plan.inputs!r}
OUTPUT = {plan.output!r}


def build_component():
    return {plan.import_symbol}(
{config_args}
    )


COMPONENT = build_component()
'''
    return GeneratedSource(filename=filename, source=source)


def _format_config_args(plan: LoweringPlan) -> str:
    config = plan.config
    fields = (
        "tile_shape_mnk",
        "epi_tile_mn",
        "cluster_shape_mnk",
        "atom_layout_mn",
        "ab_stage",
        "epi_stage",
    )
    if plan.import_symbol == "Swiglu3SM90":
        fields = fields + ("reuse_ab",)
    fields = fields + (
        "is_persistent",
        "gemm_n_prologue",
    )
    return "\n".join(
        f"        {field}={getattr(config, field)!r}," for field in fields
    )
