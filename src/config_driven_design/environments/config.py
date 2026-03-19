from dataclasses import dataclass

@dataclass(kw_only=True, frozen=True)
class EnvironmentConfig:
    num_actions: int

@dataclass(kw_only=True, frozen=True)
class TwoPlayerGameConfig(EnvironmentConfig):
    payoff_matrix: list[list[list[float]]]

    type: str = "two_player_game"
    """This field is used to help Dacite determine which EnvironmentConfig subclass to instantiate.
    It is not necessary if no other config classes share the same fields, but it is a good practice to include it for clarity and future extensibility.
    """
