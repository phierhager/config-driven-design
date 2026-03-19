
import hydra
from omegaconf import DictConfig, OmegaConf
import dacite
import jax
import jax.numpy as jnp
import equinox as eqx

from config_driven_design.config import AppConfig
from config_driven_design.agents.factory import create_agent
from config_driven_design.environments.factory import create_environment
from config_driven_design.run import run_multiple_episodes

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    app_config = dacite.from_dict(AppConfig, config_dict, dacite.Config(strict=True))

    print(f"--- Configuration Loaded: {app_config.experiment.name} ---")

    env = create_environment(app_config.environment)
    agent1 = create_agent(app_config.agent)
    agent2 = create_agent(app_config.agent)

    print(f"Simulating {app_config.experiment.num_episodes} episodes of {app_config.experiment.max_steps_per_episode} steps...")

    # Run the fully vectorized, compiled experiment!
    history = run_multiple_episodes(
        agent1, 
        agent2, 
        env, 
        app_config.experiment.seed, 
        app_config.experiment.num_episodes,
        app_config.experiment.max_steps_per_episode
    )

    # history contains arrays with shape: (num_episodes, max_steps)
    # Let's aggregate the rewards across all steps for each episode, then average them
    avg_total_reward_p1 = jnp.mean(jnp.sum(history["a1_rewards"], axis=1))
    avg_total_reward_p2 = jnp.mean(jnp.sum(history["a2_rewards"], axis=1))

    print(f"Average Total Reward P1: {avg_total_reward_p1:.2f}")
    print(f"Average Total Reward P2: {avg_total_reward_p2:.2f}")

if __name__ == "__main__":
    main()