from typing import Protocol, runtime_checkable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx


class TimeStep(eqx.Module):
    observation: jax.Array
    rewards: jax.Array
    done: jax.Array = eqx.field(default_factory=lambda: jnp.array(False, dtype=bool))

class JaxEnvironment(Protocol):
    @property
    def num_players(self) -> int:
        ...

    @property
    def num_actions(self) -> int:
        ...

    def reset(self, key: jax.Array) -> jax.Array:
        ...

    def step(self, state: jax.Array, actions: jax.Array) -> tuple[jax.Array, TimeStep]:
        ...

    def observe(self, state: jax.Array) -> jax.Array:
        ...