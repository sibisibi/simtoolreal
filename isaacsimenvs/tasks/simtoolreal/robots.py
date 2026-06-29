from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class RobotCfg:
    urdf: str = MISSING
    arm_joint_regex: str = MISSING
    hand_joint_regex: str = MISSING
    joint_order: tuple[str, ...] = MISSING
    palm_body: str = MISSING
    fingertip_body_regex: str = MISSING
    fingertip_bodies: tuple[str, ...] = MISSING
    arm_stiffness: dict[str, float] = MISSING
    arm_damping: dict[str, float] = MISSING
    hand_stiffness: dict[str, float] = MISSING
    hand_damping: dict[str, float] = MISSING
    hand_armature: dict[str, float] = MISSING
    arm_default_pos: dict[str, float] = MISSING
    palm_center_offset: tuple[float, float, float] = MISSING
    fingertip_offset_by_body: dict[str, tuple[float, float, float]] = MISSING
    raycast_link_exprs: tuple[str, ...] = MISSING
    self_collision: bool = True


_FR3_XHAND_HAND_JOINTS = (
    "thumb_joint0", "thumb_joint1", "thumb_joint2",
    "index_joint0", "index_joint1", "index_joint2",
    "middle_joint0", "middle_joint1",
    "ring_joint0", "ring_joint1",
    "pinky_joint0", "pinky_joint1",
)

_FR3_XHAND_FINGERTIP_BODIES = (
    "index_rota_link2", "mid_link2", "ring_link2",
    "thumb_rota_link2", "pinky_link2",
)

_FR3_XHAND_SHARED = dict(
    self_collision=True,
    arm_joint_regex="fr3_joint.*",
    hand_joint_regex="(thumb|index|middle|ring|pinky)_joint.*",
    joint_order=(
        "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
        "fr3_joint5", "fr3_joint6", "fr3_joint7",
    ) + _FR3_XHAND_HAND_JOINTS,
    palm_body="fr3_link7",
    fingertip_body_regex="(thumb_rota_link2|index_rota_link2|mid_link2|ring_link2|pinky_link2)",
    fingertip_bodies=_FR3_XHAND_FINGERTIP_BODIES,
    arm_stiffness={f"fr3_joint{i}": 400.0 for i in range(1, 8)},
    arm_damping={f"fr3_joint{i}": 80.0 for i in range(1, 8)},
    hand_stiffness={j: 3.0 for j in _FR3_XHAND_HAND_JOINTS},
    hand_damping={j: 0.1 for j in _FR3_XHAND_HAND_JOINTS},
    hand_armature={j: 0.001 for j in _FR3_XHAND_HAND_JOINTS},
    arm_default_pos={
        "fr3_joint1": -1.571, "fr3_joint2": -0.6, "fr3_joint3": 0.0,
        "fr3_joint4": -2.2, "fr3_joint5": 0.0, "fr3_joint6": 1.9,
        "fr3_joint7": -0.785,
    },
    palm_center_offset=(0.0, 0.0, 0.16),
    fingertip_offset_by_body={
        "thumb_rota_link2": (0.0, 0.0502276499414863, 0.0),
        "index_rota_link2": (0.0, 0.0, 0.0422482924089424),
        "mid_link2": (0.0, 0.0, 0.042248),
        "ring_link2": (0.0, 0.0, 0.0422482924089404),
        "pinky_link2": (0.0, 0.0, 0.0422482924089405),
    },
    raycast_link_exprs=(
        "/World/envs/env_.*/Robot/fr3_link.*/visuals",
        "/World/envs/env_.*/Robot/palm/visuals",
        "/World/envs/env_.*/Robot/thumb_.*/visuals",
        "/World/envs/env_.*/Robot/index_.*/visuals",
        "/World/envs/env_.*/Robot/mid_.*/visuals",
        "/World/envs/env_.*/Robot/ring_.*/visuals",
        "/World/envs/env_.*/Robot/pinky_.*/visuals",
    ),
)

_SHARPA_HAND_JOINTS = (
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)

_SHARPA_FINGERTIP_BODIES = (
    "left_index_DP", "left_middle_DP", "left_ring_DP",
    "left_thumb_DP", "left_pinky_DP",
)

ROBOT_PROFILES: dict[str, RobotCfg] = {
    "kuka-sharpa": RobotCfg(
        urdf="assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf",
        arm_joint_regex="iiwa14_joint_.*",
        hand_joint_regex="left_.*",
        joint_order=(
            "iiwa14_joint_1", "iiwa14_joint_2", "iiwa14_joint_3", "iiwa14_joint_4",
            "iiwa14_joint_5", "iiwa14_joint_6", "iiwa14_joint_7",
        ) + _SHARPA_HAND_JOINTS,
        palm_body="iiwa14_link_7",
        fingertip_body_regex="left_(index|middle|ring|thumb|pinky)_DP",
        fingertip_bodies=_SHARPA_FINGERTIP_BODIES,
        arm_stiffness={
            "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
            "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
            "iiwa14_joint_7": 200.0,
        },
        arm_damping={
            "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
            "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
            "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
            "iiwa14_joint_7": 9.147747263670984,
        },
        hand_stiffness={
            "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
            "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
            "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
            "left_index_PIP": 0.9, "left_index_DIP": 0.9,
            "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
            "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
            "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
            "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
            "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
            "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
        },
        hand_damping={
            "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
            "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
            "left_thumb_IP": 0.04190723,
            "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
            "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
            "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
            "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
            "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
            "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
            "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
            "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
            "left_pinky_DIP": 0.0350446,
        },
        hand_armature={
            "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
            "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
            "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
            "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
            "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
            "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
            "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
            "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
            "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
            "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
        },
        arm_default_pos={
            "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
            "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
            "iiwa14_joint_7": 1.308,
        },
        palm_center_offset=(-0.0, -0.02, 0.16),
        fingertip_offset_by_body={b: (0.02, 0.002, 0.0) for b in _SHARPA_FINGERTIP_BODIES},
        raycast_link_exprs=(
            "/World/envs/env_.*/Robot/iiwa14_link_.*/visuals",
            "/World/envs/env_.*/Robot/left_.*/visuals",
        ),
    ),
    "fr3-xhand-adapter": RobotCfg(
        urdf="assets/urdf/fr3_xhand_description/fr3_xhand/fr3_xhand.urdf",
        **_FR3_XHAND_SHARED,
    ),
    "fr3-xhand": RobotCfg(
        urdf="assets/urdf/fr3_xhand_description/fr3_xhand/fr3_xhand_noadapter.urdf",
        **_FR3_XHAND_SHARED,
    ),
}
