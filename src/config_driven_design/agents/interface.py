from dataclasses import dataclass
from typing import Protocol

import jax
import equinox as eqx

class AgentInput(eqx.Module):
    observation: jax.Array
    reward: jax.Array
    done: jax.Array


class AgentOutput(eqx.Module):
    action: jax.Array
    scores: jax.Array


class AgentTransition(eqx.Module):
    action: jax.Array
    reward: jax.Array


class JaxAgent(Protocol):
    @property
    def num_actions(self) -> int:
        ...

    def reset(self, key: jax.Array) -> jax.Array:
        ...

    def act(
        self,
        state: jax.Array,
        agent_input: AgentInput,
        key: jax.Array,
    ) -> tuple[jax.Array, AgentOutput]:
        ...

    def update(self, state: jax.Array, transition: AgentTransition) -> jax.Array:
        ...