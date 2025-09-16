# board_model.py
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class BoardModel:
    r_inner_bull: float = 6.35
    r_outer_bull: float = 15.9
    r_treble_in:  float = 107.0
    ring_width:   float = 8.0
    r_double_out: float = 170.0
    seg20_center_deg: float = 90.0
    sector_order: Tuple[int, ...] = (
        20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
        3, 19, 7, 16, 8, 11, 14, 9, 12, 5
    )
    @property
    def r_treble_out(self) -> float: return self.r_treble_in + self.ring_width
    @property
    def r_double_in(self) -> float:  return self.r_double_out - self.ring_width
    @property
    def r_board_score_outer(self) -> float: return self.r_double_out
    def polar_to_sector(self, theta_deg: float) -> int:
        a = theta_deg - self.seg20_center_deg
        a = (-a) % 360.0
        idx = int((a + 9.0) // 18.0) % 20
        return self.sector_order[idx]
    def ring_from_radius(self, r_mm: float) -> str:
        if r_mm <= self.r_inner_bull:  return "inner_bull"
        if r_mm <= self.r_outer_bull:  return "outer_bull"
        if self.r_treble_in < r_mm <= self.r_treble_out: return "treble"
        if self.r_double_in < r_mm <= self.r_double_out: return "double"
        if r_mm <= self.r_board_score_outer: return "single"
        return "miss"
    def score(self, r_mm: float, theta_deg: float) -> tuple[int, str, int]:
        ring = self.ring_from_radius(r_mm)
        if ring == "inner_bull": return 50, ring, 0
        if ring == "outer_bull": return 25, ring, 0
        if ring == "miss":       return 0, ring, 0
        sector = self.polar_to_sector(theta_deg)
        mult = {"single":1,"double":2,"treble":3}[ring]
        return sector * mult, ring, sector
