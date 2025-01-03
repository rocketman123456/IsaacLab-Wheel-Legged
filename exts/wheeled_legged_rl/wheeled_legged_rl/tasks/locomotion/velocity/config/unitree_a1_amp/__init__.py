import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Velocity-Flat-Amp-Unitree-A1-v0",
    entry_point=("wheeled_legged_rl.tasks.locomotion.velocity.config.unitree_a1_amp.env.manager_based_rl_amp_env:ManagerBasedRLAmpEnv"),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeA1AmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1AmpFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Rough-Amp-Unitree-A1-v0",
    entry_point=("wheeled_legged_rl.tasks.locomotion.velocity.config.unitree_a1_amp.env.manager_based_rl_amp_env:ManagerBasedRLAmpEnv"),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeA1AmpRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeA1AmpRoughPPORunnerCfg",
    },
)
