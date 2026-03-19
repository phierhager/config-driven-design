from dataclasses import dataclass
from typing import Union, Any

from config_driven_design.agents.config import UCBConfig, ThomsonSamplingConfig
from config_driven_design.environments.config import TwoPlayerGameConfig

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    num_episodes: int
    max_steps_per_episode: int

@dataclass(frozen=True)
class AppConfig:
    experiment: ExperimentConfig
    # Dacite will automatically try these types in order and pick the one 
    # whose fields match the incoming dictionary.
    agent: Union[UCBConfig, ThomsonSamplingConfig]
    environment: Union[TwoPlayerGameConfig]