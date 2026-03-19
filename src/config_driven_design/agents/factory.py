from config_driven_design.agents.thomson_sampling import ThomsonSamplingAgent
from config_driven_design.agents.ucb import UCBAgent
from config_driven_design.agents.interface import JaxAgent
from config_driven_design.agents.config import AgentConfig, UCBConfig, ThomsonSamplingConfig

def create_agent(config: AgentConfig) -> JaxAgent:
    if isinstance(config, UCBConfig):
        return UCBAgent(
            num_actions=config.num_actions,
            exploration_coefficient=config.exploration_coefficient,
        )
    elif isinstance(config, ThomsonSamplingConfig):
        return ThomsonSamplingAgent(
            num_actions=config.num_actions,
            prior_alpha=config.prior_alpha,
            prior_beta=config.prior_beta,
        )
    else:
        raise ValueError(f"Unsupported agent config type: {type(config)}")