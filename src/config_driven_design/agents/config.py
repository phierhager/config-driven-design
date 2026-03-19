from dataclasses import dataclass

@dataclass(kw_only=True, frozen=True)
class AgentConfig:
    num_actions: int

@dataclass(kw_only=True, frozen=True)
class UCBConfig(AgentConfig):
    exploration_coefficient: float

    type: str = "ucb"  
    """This field is used to help Dacite determine which AgentConfig subclass to instantiate.
    It is not necessary if no other config classes share the same fields, but it is a good practice to include it for clarity and future extensibility.
    """

@dataclass(kw_only=True, frozen=True)
class ThomsonSamplingConfig(AgentConfig):
    prior_alpha: float 
    prior_beta: float

    type: str = "thomson_sampling"
    """This field is used to help Dacite determine which AgentConfig subclass to instantiate.
    It is not necessary if no other config classes share the same fields, but it is a good practice to include it for clarity and future extensibility.
    """
