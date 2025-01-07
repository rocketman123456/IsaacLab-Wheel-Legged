import math

from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

import wheeled_legged_rl.tasks.locomotion.velocity.mdp as mdp
from wheeled_legged_rl.tasks.locomotion.velocity.config.unitree_go2w_handstand.env import rewards
from wheeled_legged_rl.tasks.locomotion.velocity.velocity_env_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
# use cloud assets
# from omni.isaac.lab_assets.unitree import UNITREE_GO2W_CFG  # isort: skip

# use local assets
from wheeled_legged_rl.assets.unitree import UNITREE_GO2W_CFG  # isort: skip


@configclass
class UnitreeGo2WActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None)
    joint_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None)


@configclass
class UnitreeGo2WHandStandRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")})
    joint_acc_wheel_l2 = RewTerm(func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")})
    joint_torques_wheel_l2 = RewTerm(func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")})

    handstand_feet_height_exp = RewTerm(
        func=rewards.handstand_feet_height_exp,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.0, "std": math.sqrt(0.25)},
    )
    handstand_feet_on_air = RewTerm(
        func=rewards.handstand_feet_on_air,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
        },
    )
    handstand_feet_air_time = RewTerm(
        func=rewards.handstand_feet_air_time,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 5.0,
        },
    )
    handstand_orientation_l2 = RewTerm(
        func=rewards.handstand_orientation_l2,
        weight=0.0,
        params={
            "target_gravity": [],
        },
    )


@configclass
class UnitreeGo2WHandStandRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: UnitreeGo2WHandStandRewardsCfg = UnitreeGo2WHandStandRewardsCfg()
    actions: UnitreeGo2WActionsCfg = UnitreeGo2WActionsCfg()

    base_link_name = "base"
    foot_link_name = ".*_foot"
    wheel_joint_name = ".*_foot_joint"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 10.0
        self.viewer.resolution = (1920, 1080)

        # ------------------------------Sence------------------------------
        # switch robot to unitree-go2w
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg("robot", joint_names=[self.wheel_joint_name])
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg("robot", joint_names=[self.wheel_joint_name])
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        # self.actions.joint_pos.scale = 0.25
        # self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.actions.joint_vel.joint_names = [self.wheel_joint_name]

        # ------------------------------Events------------------------------
        # self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.randomize_actuator_gains = None
        # self.events.randomize_joint_parameters = None

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.35
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -0.0002
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.1, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.05
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*_thigh"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [".*_foot"]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still_when_zero_command.weight = -0.01

        # HandStand
        handstand_type = "back"  # which leg on air, can be "front", "back", "left", "right"
        if handstand_type == "front":
            air_foot_name = "F.*_foot"
            self.rewards.handstand_orientation_l2.weight = -1.0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [-1.0, 0.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.5
        elif handstand_type == "back":
            air_foot_name = "R.*_foot"
            self.rewards.handstand_orientation_l2.weight = -1.0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [1.0, 0.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.5
        elif handstand_type == "left":
            air_foot_name = ".*L_foot"
            self.rewards.handstand_orientation_l2.weight = 0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [0.0, -1.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.3
        elif handstand_type == "right":
            air_foot_name = ".*R_foot"
            self.rewards.handstand_orientation_l2.weight = 0
            self.rewards.handstand_orientation_l2.params["target_gravity"] = [0.0, 1.0, 0.0]
            self.rewards.handstand_feet_height_exp.params["target_height"] = 0.3
        self.rewards.handstand_feet_height_exp.weight = 10
        self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_on_air.weight = 1.0
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_air_time.weight = 1.0
        self.rewards.handstand_feet_air_time.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WHandStandRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
            ".*_hip",
            ".*_thigh",
            ".*_calf",
        ]

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = False
