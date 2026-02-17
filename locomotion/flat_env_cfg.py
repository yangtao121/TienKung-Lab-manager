from isaaclab.utils import configclass

from .rough_env_cfg import TienKung2LiteRoughEnvCfg


@configclass
class TienKung2LiteFlatEnvCfg(TienKung2LiteRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
