from dataclasses import dataclass
from typing import Optional, Tuple

from register_contract.types import Location, Producer, RegisterRole, TileLayout

@dataclass(frozen=True)
class RegisterTile:
    name: str
    dtype: str
    rows: int
    cols: int
    layout: TileLayout
    role: RegisterRole
    producer: Optional[Producer] = None
    sources: Tuple[str, ...] = ()
    location: Location = Location.REGISTER
