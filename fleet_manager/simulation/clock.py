"""
Minimal deterministic clock for the fixed-timestep simulation loop.
Not event-driven (a true event-queue simulator would jump directly between
state changes); this fixed-dt approach is simpler, fully deterministic
across runs, and precise enough at dt=0.1s for this arena's speeds/scales.
"""
from dataclasses import dataclass


@dataclass
class SimClock:
    t: float = 0.0
    dt: float = 0.1

    def tick(self) -> float:
        self.t += self.dt
        return self.t
