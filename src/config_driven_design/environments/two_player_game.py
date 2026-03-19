import jax
import jax.numpy as jnp
import equinox as eqx

from config_driven_design.environments.interface import TimeStep


class TwoPlayerGame(eqx.Module):
    payoff_matrix: jax.Array

    @property
    def num_players(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return 2

    def reset(self, key: jax.Array) -> jax.Array:
        del key
        return jnp.array(0, dtype=jnp.int32)

    def observe(self, state: jax.Array) -> jax.Array:
        return jnp.asarray(state, dtype=jnp.int32)[None]

    def step(self, state: jax.Array, actions: jax.Array) -> tuple[jax.Array, TimeStep]:
        clipped_actions = jnp.clip(jnp.asarray(actions, dtype=jnp.int32), 0, self.num_actions - 1)
        action_0 = clipped_actions[0]
        action_1 = clipped_actions[1]

        rewards = self.payoff_matrix[action_0, action_1]
        next_state = jnp.asarray(state, dtype=jnp.int32) + 1

        timestep = TimeStep(observation=self.observe(next_state), rewards=rewards)
        return next_state, timestep