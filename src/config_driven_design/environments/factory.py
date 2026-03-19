import jax.numpy as jnp

from config_driven_design.environments.two_player_game import TwoPlayerGame
from config_driven_design.environments.config import EnvironmentConfig, TwoPlayerGameConfig
from config_driven_design.environments.interface import JaxEnvironment

def create_environment(config: EnvironmentConfig) -> JaxEnvironment:
    if isinstance(config, TwoPlayerGameConfig):
        payoff_matrix = jnp.asarray(config.payoff_matrix, dtype=jnp.float32)
        return TwoPlayerGame(payoff_matrix=payoff_matrix)
    else:
        raise ValueError(f"Unsupported environment config type: {type(config)}")