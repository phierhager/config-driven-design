import hydra
from omegaconf import DictConfig, OmegaConf
import dacite
import jax
import jax.numpy as jnp
import equinox as eqx

from config_driven_design.config import AppConfig
from config_driven_design.agents.factory import create_agent
from config_driven_design.environments.factory import create_environment
from config_driven_design.agents.interface import AgentInput, AgentTransition


def run_compiled_episode(agent1, agent2, env, seed_key: jax.Array, max_steps: int):
    """Runs a single episode using a compiled jax.lax.scan loop."""
    rng = seed_key
    rng, env_key, a1_key, a2_key = jax.random.split(rng, 4)

    # 1. Initial State
    env_state = env.reset(env_key)
    a1_state = agent1.reset(a1_key)
    a2_state = agent2.reset(a2_key)

    obs = env.observe(env_state)
    input1 = AgentInput(observation=obs, reward=jnp.array(0.0), done=jnp.array(False))
    input2 = AgentInput(observation=obs, reward=jnp.array(0.0), done=jnp.array(False))

    carry = (env_state, a1_state, a2_state, input1, input2, rng)

    def scan_step(carry, _):
        env_state, a1_state, a2_state, input1, input2, rng = carry
        rng, k1, k2 = jax.random.split(rng, 3)

        # Check if the episode was ALREADY done before this step started
        was_done = input1.done

        # 1. Agents Act
        a1_state, a1_out = agent1.act(a1_state, input1, k1)
        a2_state, a2_out = agent2.act(a2_state, input2, k2)

        # 2. Environment Steps
        actions = jnp.array([a1_out.action, a2_out.action])
        env_state, timestep = env.step(env_state, actions)
        
        # 3. Apply Done Masking
        # If we are done, valid_mask is False. We zero out rewards for ghost steps.
        valid_mask = jnp.logical_not(was_done)
        masked_r1 = jnp.where(valid_mask, timestep.rewards[0], 0.0)
        masked_r2 = jnp.where(valid_mask, timestep.rewards[1], 0.0)

        # 4. Conditionally Update Agents
        # We calculate the new state, but only apply it if the episode is still active
        proposed_a1_state = agent1.update(a1_state, AgentTransition(action=a1_out.action, reward=masked_r1))
        proposed_a2_state = agent2.update(a2_state, AgentTransition(action=a2_out.action, reward=masked_r2))

        # jax.tree_util.tree_map applies the jnp.where to every array inside the dataclasses
        a1_state = jax.tree_util.tree_map(lambda new, old: jnp.where(valid_mask, new, old), proposed_a1_state, a1_state)
        a2_state = jax.tree_util.tree_map(lambda new, old: jnp.where(valid_mask, new, old), proposed_a2_state, a2_state)

        # 5. Determine if we are done going into the next step
        is_now_done = jnp.logical_or(was_done, timestep.done)

        next_input1 = AgentInput(observation=timestep.observation, reward=masked_r1, done=is_now_done)
        next_input2 = AgentInput(observation=timestep.observation, reward=masked_r2, done=is_now_done)

        next_carry = (env_state, a1_state, a2_state, next_input1, next_input2, rng)
        
        metrics = {
            "a1_actions": a1_out.action,
            "a2_actions": a2_out.action,
            "a1_rewards": masked_r1,
            "a2_rewards": masked_r2,
            "done": is_now_done
        }
        
        return next_carry, metrics

    final_carry, history = jax.lax.scan(scan_step, carry, None, length=max_steps)
    return history


@eqx.filter_jit
def run_multiple_episodes(agent1, agent2, env, base_seed: int, num_episodes: int, max_steps: int):
    """Vectorizes the episode loop to run all episodes in parallel."""
    rng = jax.random.PRNGKey(base_seed)
    
    # Generate a unique seed for each episode
    episode_keys = jax.random.split(rng, num_episodes)

    # vmap over the seed dimension (in_axes=0 for episode_keys, None for the rest)
    mapped_run = jax.vmap(run_compiled_episode, in_axes=(None, None, None, 0, None))
    
    return mapped_run(agent1, agent2, env, episode_keys, max_steps)