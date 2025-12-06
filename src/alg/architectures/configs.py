from alg.architectures.cnn import BaseCnnActorCritic
from alg.architectures.resnet import BaseResNetActorCritic
from alg.architectures.transformer import BaseTransformerActorCritic


# --- TRANSFORMER ---
class TransformerSActorCritic(BaseTransformerActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, embed_dim=56, num_layers=2, num_heads=4, head_hidden_dim=128)
        self._architecture_name = "transformer_b_s"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


class TransformerLActorCritic(BaseTransformerActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, embed_dim=96, num_layers=5, num_heads=8)
        self._architecture_name = "transformer_b_l"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


# --- RESNET ---
class ResNetSActorCritic(BaseResNetActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, channels=32, num_blocks=4, head_hidden_dim=128)
        self._architecture_name = "resnet_b_s"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


class ResNetLActorCritic(BaseResNetActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, channels=80, num_blocks=5)
        self._architecture_name = "resnet_b_l"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


# --- CNN ---
class CnnSActorCritic(BaseCnnActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, channels=[56, 56, 56, 56], head_hidden_dim=128)
        self._architecture_name = "cnn_b_s"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }


class CnnLActorCritic(BaseCnnActorCritic):
    def __init__(self, obs_shape, action_dim):
        super().__init__(obs_shape, action_dim, channels=[96] * 8)
        self._architecture_name = "cnn_b_l"
        self._architecture_params = {
            "obs_shape": [int(x) for x in obs_shape],
            "action_dim": int(action_dim),
        }