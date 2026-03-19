from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx

from config_driven_design.agents.interface import AgentInput, AgentOutput, AgentTransition


class ThomsonSamplingState(eqx.Module):
    alpha: jax.Array
    beta: jax.Array


class ThomsonSamplingAgent(eqx.Module):
    num_actions: int
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def reset(self, key: jax.Array) -> ThomsonSamplingState:
        del key
        return ThomsonSamplingState(
            alpha=jnp.full((self.num_actions,), self.prior_alpha, dtype=jnp.float32),
            beta=jnp.full((self.num_actions,), self.prior_beta, dtype=jnp.float32),
        )

    def act(
        self,
        state: ThomsonSamplingState,
        agent_input: AgentInput,
        key: jax.Array,
    ) -> tuple[ThomsonSamplingState, AgentOutput]:
        del agent_input
        sampled_scores = jax.random.beta(key, state.alpha, state.beta)
        action = jnp.argmax(sampled_scores).astype(jnp.int32)
        output = AgentOutput(action=action, scores=sampled_scores)
        return state, output

    def update(
        self,
        state: ThomsonSamplingState,
        transition: AgentTransition,
    ) -> ThomsonSamplingState:
        action = jnp.asarray(transition.action, dtype=jnp.int32)
        reward = jnp.asarray(transition.reward, dtype=jnp.float32)
        bounded_reward = jnp.clip(reward, 0.0, 1.0)

        return ThomsonSamplingState(
            alpha=state.alpha.at[action].add(bounded_reward),
            beta=state.beta.at[action].add(1.0 - bounded_reward),
        )