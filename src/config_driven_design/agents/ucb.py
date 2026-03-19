from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx

from config_driven_design.agents.interface import AgentInput, AgentOutput, AgentTransition


class UCBState(eqx.Module):
    counts: jax.Array
    value_estimates: jax.Array
    total_steps: jax.Array


class UCBAgent(eqx.Module):
    num_actions: int
    exploration_coefficient: float = 2.0

    def reset(self, key: jax.Array) -> UCBState:
        del key
        return UCBState(
            counts=jnp.zeros((self.num_actions,), dtype=jnp.int32),
            value_estimates=jnp.zeros((self.num_actions,), dtype=jnp.float32),
            total_steps=jnp.array(0, dtype=jnp.int32),
        )

    def act(
        self,
        state: UCBState,
        agent_input: AgentInput,
        key: jax.Array,
    ) -> tuple[UCBState, AgentOutput]:
        del agent_input
        del key

        safe_total_steps = jnp.maximum(state.total_steps, 1)
        explored = state.counts > 0
        exploration_bonus = self.exploration_coefficient * jnp.sqrt(
            jnp.log(safe_total_steps.astype(jnp.float32) + 1.0)
            / jnp.maximum(state.counts.astype(jnp.float32), 1.0)
        )
        scores = jnp.where(explored, state.value_estimates + exploration_bonus, jnp.inf)

        action = jnp.argmax(scores).astype(jnp.int32)
        output = AgentOutput(action=action, scores=scores)
        return state, output

    def update(self, state: UCBState, transition: AgentTransition) -> UCBState:
        action = jnp.asarray(transition.action, dtype=jnp.int32)
        reward = jnp.asarray(transition.reward, dtype=jnp.float32)

        action_count = state.counts[action]
        new_count = action_count + 1
        prev_value = state.value_estimates[action]
        new_value = prev_value + (reward - prev_value) / new_count.astype(jnp.float32)

        return UCBState(
            counts=state.counts.at[action].set(new_count),
            value_estimates=state.value_estimates.at[action].set(new_value),
            total_steps=state.total_steps + 1,
        )