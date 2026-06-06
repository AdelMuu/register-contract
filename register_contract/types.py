from enum import Enum

class TileLayout(str, Enum):
    ROW = "row"
    COL = "col"


class RegisterRole(str, Enum):
    NONE = "none"
    GEMM_ACCUMULATOR = "gemm_accumulator"
    SWIGLU_GATE = "swiglu_gate"
    SWIGLU_VALUE = "swiglu_value"
    SWIGLU_OUTPUT = "swiglu_output"


class Producer(str, Enum):
    GEMM = "gemm"


class Location(str, Enum):
    REGISTER = "register"
