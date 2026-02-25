import os

import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, MujocoPyFetchEnv
from gymnasium_robotics.utils.rotations import euler2quat, quat2euler, quat_mul

MODEL_XML_PATH = os.path.join("fetch", "stir.xml")

# Bowl and ball layout constants
_BOWL_CENTER_X = 1.525
_BOWL_CENTER_Y = 0.825
_BALL_RADIUS_LAYOUT = 0.055  # radial distance of ball centers from bowl axis

# Layer Z positions (world frame) — all blues below all reds
_BLUE_Z1 = 0.440   # blue layer 1  (base_top=0.420 + ball_r=0.020)
_BLUE_Z2 = 0.485   # blue layer 2
_BLUE_Z3 = 0.532   # blue layer 3
_BLUE_Z4 = 0.577   # blue layer 4
_RED_Z1  = 0.622   # red layer 1
_RED_Z2  = 0.667   # red layer 2
_RED_Z3  = 0.712   # red layer 3
_RED_Z4  = 0.757   # red layer 4  (top=0.777 < wall_top=0.780)

_ANGLES_ODD  = [0, 72, 144, 216, 288]   # degrees, layers 1 & 3
_ANGLES_EVEN = [36, 108, 180, 252, 324] # degrees, layers 2 & 4


def _layer_positions(z, angles_deg):
    """Return (N, 3) world positions for balls in one layer."""
    pos = []
    for a in angles_deg:
        rad = np.radians(a)
        x = _BOWL_CENTER_X + _BALL_RADIUS_LAYOUT * np.cos(rad)
        y = _BOWL_CENTER_Y + _BALL_RADIUS_LAYOUT * np.sin(rad)
        pos.append([x, y, z])
    return pos


# Pre-computed ball positions: blues (obj1-10, 21-30) at bottom, reds (obj11-20, 31-40) at top
_BALL_INIT_POSITIONS = (
    _layer_positions(_BLUE_Z1, _ANGLES_ODD)    # obj1-5   (blue, Z=0.455)
    + _layer_positions(_BLUE_Z2, _ANGLES_EVEN) # obj6-10  (blue, Z=0.500)
    + _layer_positions(_RED_Z1,  _ANGLES_ODD)  # obj11-15 (red,  Z=0.637)
    + _layer_positions(_RED_Z2,  _ANGLES_EVEN) # obj16-20 (red,  Z=0.682)
    + _layer_positions(_BLUE_Z3, _ANGLES_ODD)  # obj21-25 (blue, Z=0.547)
    + _layer_positions(_BLUE_Z4, _ANGLES_EVEN) # obj26-30 (blue, Z=0.592)
    + _layer_positions(_RED_Z3,  _ANGLES_ODD)  # obj31-35 (red,  Z=0.727)
    + _layer_positions(_RED_Z4,  _ANGLES_EVEN) # obj36-40 (red,  Z=0.772)
)


def _build_initial_qpos():
    qpos = {
        "robot0:slide0": 0.4049,
        "robot0:slide1": 0.48,
        "robot0:slide2": 0.0,
        # Spoon starts near gripper initial position
        "object0:joint": [1.3419, 0.7491, 0.560, 1.0, 0.0, 0.0, 0.0],
        # Bowl at its rest position
        "stir_bowl:joint": [1.525, 0.825, 0.410, 1.0, 0.0, 0.0, 0.0],
    }
    for i, pos in enumerate(_BALL_INIT_POSITIONS):
        qpos[f"object{i + 1}:joint"] = [pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0]
    return qpos


class MujocoFetchStirEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description

    FetchStir environment: robot holds a spoon (pre-held via weld constraint) and
    must stir 20 balls in a bowl (10 blue on bottom, 10 red on top) until they mix,
    without tipping the bowl.

    ## Action Space
    Box(-1, 1, (4,)): [dx, dy, dz, dtheta_z]

    ## Observation Space (GoalEnv dict, FlattenObservation -> 31 dim)
    - observation  (25,): robot + spoon kinematics
    - achieved_goal (3,): [z_sep, xy_sep, bowl_tilt]
    - desired_goal  (3,): [0, 0, 0]  (fixed)

    ## Reward (Dense)
    r = w_v*(1-tanh(8*z_sep)) + w_l*(1-tanh(6*xy_sep)) + w_s*cos(bowl_tilt)
    w_v=1.0, w_l=0.5, w_s=1.0

    ## Reward (Sparse)
    0 if z_sep < 0.02 and bowl_tilt < 0.1, else -1
    """

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = _build_initial_qpos()

        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.0,
            target_range=0.0,
            distance_threshold=0.02,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

        self.target_gripper_pos = 0.0  # fully closed

        # Dense reward scaling
        self.k_z       = 8.0   # tanh scale for vertical separation
        self.k_xy      = 6.0   # tanh scale for horizontal separation
        self.k_approach = 4.0  # tanh scale for spoon→bowl approach
        self.w_v       = 3.0   # weight: vertical mixing  ← dominant signal
        self.w_l       = 0.5   # weight: lateral mixing
        self.w_s       = 0.4   # weight: bowl stability  ← allows slight tilt
        self.w_approach = 0.2  # weight: spoon→bowl approach ← already learned

        # Cache weld constraint ID
        self._weld_eq_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_EQUALITY, "spoon_grip"
        )
        assert self._weld_eq_id >= 0, "Weld constraint 'spoon_grip' not found in model"

        # Ball body IDs are cached lazily in _init_ball_body_ids()
        # (base class calls _get_obs() during __init__ before we can cache them here)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_reward(self, achieved_goal, goal, info):
        z_sep      = achieved_goal[..., 0]
        xy_sep     = achieved_goal[..., 1]
        bowl_tilt  = achieved_goal[..., 2]
        spoon_dist = achieved_goal[..., 3]

        if self.reward_type == "sparse":
            success = (z_sep < self.distance_threshold) & (bowl_tilt < 0.1)
            return -(1.0 - success.astype(np.float32))

        r_vertical  = 1.0 - np.tanh(self.k_z       * z_sep)
        r_lateral   = 1.0 - np.tanh(self.k_xy      * xy_sep)
        r_stability = np.cos(bowl_tilt)
        r_approach  = 1.0 - np.tanh(self.k_approach * spoon_dist)

        return (self.w_v       * r_vertical
                + self.w_l    * r_lateral
                + self.w_s    * r_stability
                + self.w_approach * r_approach)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _init_ball_body_ids(self):
        """Lazily cache ball body IDs (needed because _get_obs is called during __init__)."""
        ids = lambda rng: [
            self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_BODY, f"object{i}"
            ) for i in rng
        ]
        self._blue_body_ids = ids(range(1, 11)) + ids(range(21, 31))
        self._red_body_ids  = ids(range(11, 21)) + ids(range(31, 41))
        assert all(b >= 0 for b in self._blue_body_ids), "Blue ball bodies not found"
        assert all(b >= 0 for b in self._red_body_ids),  "Red ball bodies not found"

    def _get_ball_stats(self):
        """Return (z_sep, xy_sep) between red and blue ball centroids."""
        if not hasattr(self, "_red_body_ids"):
            self._init_ball_body_ids()
        red_pos  = np.stack([self.data.xpos[i] for i in self._red_body_ids])
        blue_pos = np.stack([self.data.xpos[i] for i in self._blue_body_ids])
        z_sep  = float(abs(red_pos[:, 2].mean() - blue_pos[:, 2].mean()))
        xy_sep = float(np.linalg.norm(red_pos[:, :2].mean(0) - blue_pos[:, :2].mean(0)))
        return z_sep, xy_sep

    def _get_spoon_dist_to_bowl(self):
        """XY distance from spoon (object0) to bowl center."""
        if not hasattr(self, "_spoon_body_id"):
            self._spoon_body_id = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_BODY, "object0"
            )
        spoon_xy = self.data.xpos[self._spoon_body_id, :2]
        bowl_xy = np.array([_BOWL_CENTER_X, _BOWL_CENTER_Y])
        return float(np.linalg.norm(spoon_xy - bowl_xy))

    def _get_bowl_tilt(self):
        """Return bowl tilt angle (sqrt of roll^2 + pitch^2) in radians."""
        qpos = self._utils.get_joint_qpos(self.model, self.data, "stir_bowl:joint")
        euler = quat2euler(qpos[3:7])
        return float(np.sqrt(euler[0] ** 2 + euler[1] ** 2))

    def _get_obs(self):
        obs_dict = super()._get_obs()
        z_sep, xy_sep = self._get_ball_stats()
        bowl_tilt = self._get_bowl_tilt()
        spoon_dist = self._get_spoon_dist_to_bowl()
        achieved_goal = np.array([z_sep, xy_sep, bowl_tilt, spoon_dist], dtype=np.float64)
        desired_goal  = np.zeros(4, dtype=np.float64)
        return {
            "observation":   obs_dict["observation"],
            "achieved_goal": achieved_goal,
            "desired_goal":  desired_goal,
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        obs, _reward, terminated, truncated, info = super().step(action)

        z_sep      = float(obs["achieved_goal"][0])
        xy_sep     = float(obs["achieved_goal"][1])
        bowl_tilt  = float(obs["achieved_goal"][2])
        spoon_dist = float(obs["achieved_goal"][3])

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        info["z_sep"]        = z_sep
        info["xy_sep"]       = xy_sep
        info["bowl_tilt"]    = bowl_tilt
        info["spoon_dist"]   = spoon_dist
        info["mixing_score"] = float(1.0 - np.tanh(self.k_z * z_sep))

        if self.reward_type != "sparse":
            info["reward_vertical"]  = float(1.0 - np.tanh(self.k_z       * z_sep))
            info["reward_lateral"]   = float(1.0 - np.tanh(self.k_xy      * xy_sep))
            info["reward_stability"] = float(np.cos(bowl_tilt))
            info["reward_approach"]  = float(1.0 - np.tanh(self.k_approach * spoon_dist))

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------------

    def _sample_goal(self):
        return np.zeros(4)

    def _render_callback(self):
        pass

    # ------------------------------------------------------------------
    # Action (identical to peg_in_hole_pre_held)
    # ------------------------------------------------------------------

    def _reset_mocap2body_xpos(self):
        """Reset mocap positions, skipping non-mocap welds (spoon_grip)."""
        import mujoco as mj
        for i in range(self.model.neq):
            if self.model.eq_type[i] != mj.mjtEq.mjEQ_WELD:
                continue
            obj1_id = self.model.eq_obj1id[i]
            obj2_id = self.model.eq_obj2id[i]
            mocap_id = self.model.body_mocapid[obj1_id]
            if mocap_id != -1:
                body_idx = obj2_id
            else:
                mocap_id = self.model.body_mocapid[obj2_id]
                body_idx = obj1_id
            if mocap_id == -1:
                continue  # non-mocap weld (spoon_grip)
            self.data.mocap_pos[mocap_id][:] = self.data.xpos[body_idx]
            self.data.mocap_quat[mocap_id][:] = self.data.xquat[body_idx]

    def _update_weld_constraint(self):
        """Compute and set weld relpose from current gripper–spoon positions."""
        grip_body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link"
        )
        spoon_body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "object0"
        )

        grip_pos  = self.data.xpos[grip_body_id]
        grip_mat  = self.data.xmat[grip_body_id].reshape(3, 3)
        grip_quat = self.data.xquat[grip_body_id]
        spoon_pos = self.data.xpos[spoon_body_id]
        spoon_quat = self.data.xquat[spoon_body_id]

        rel_pos = grip_mat.T @ (spoon_pos - grip_pos)

        grip_quat_inv = np.array([
            grip_quat[0], -grip_quat[1], -grip_quat[2], -grip_quat[3]
        ])
        rel_quat = quat_mul(grip_quat_inv, spoon_quat)

        self.model.eq_data[self._weld_eq_id, 3:6] = rel_pos
        self.model.eq_data[self._weld_eq_id, 6:10] = rel_quat

    def _step_callback(self):
        """Keep gripper closed."""
        if hasattr(self, "data") and hasattr(self.data, "ctrl"):
            try:
                l_idx = self._model_names.actuator_name2id["robot0:l_gripper_finger_joint"]
                r_idx = self._model_names.actuator_name2id["robot0:r_gripper_finger_joint"]
                self.data.ctrl[l_idx] = self.target_gripper_pos
                self.data.ctrl[r_idx] = self.target_gripper_pos
            except (KeyError, AttributeError):
                pass

    def _set_action(self, action):
        """[dx, dy, dz, dtheta_z] — spoon welded, gripper stays closed."""
        assert action.shape == (4,)
        action = action.copy()

        pos_ctrl  = action[:3] * 0.05
        rot_z_ctrl = action[3] * 0.1

        self._reset_mocap2body_xpos()
        current_quat = self.data.mocap_quat[0].copy()

        rot_quat = euler2quat(np.array([rot_z_ctrl, 0.0, 0.0]))
        new_quat  = quat_mul(current_quat, rot_quat)

        self.data.mocap_pos[0] = self.data.mocap_pos[0] + pos_ctrl
        self.data.mocap_quat[0] = new_quat

        gripper_ctrl = np.array([0.0, 0.0])
        gripper_action = np.concatenate([pos_ctrl, [0, 0, 0, 0], gripper_ctrl])
        self._utils.ctrl_set_action(self.model, self.data, gripper_action)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_sim(self):
        """Reset: gripper to start pos, bowl upright, balls in layers, 300 settle steps."""
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Disable weld during gripper positioning (both default and runtime)
        self.model.eq_active0[self._weld_eq_id] = 0
        self.data.eq_active[self._weld_eq_id] = 0

        # Close gripper
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint", self.target_gripper_pos
        )
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:r_gripper_finger_joint", self.target_gripper_pos
        )

        self._mujoco.mj_forward(self.model, self.data)

        # Move gripper to a starting position away from the bowl
        self._reset_mocap2body_xpos()
        self.data.mocap_pos[0] = np.array([1.15, 0.60, 0.65])
        self.data.mocap_quat[0] = np.array([0.7071068, 0.0, 0.7071068, 0.0])

        # Step until gripper reaches target position
        for _ in range(100):
            self._mujoco.mj_step(self.model, self.data)

        # Place spoon at gripper (2 cm below grip site), handle pointing up
        gripper_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")
        spoon_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
        assert spoon_qpos.shape == (7,)
        spoon_qpos[:3] = gripper_pos + np.array([0.0, 0.0, -0.02])
        spoon_qpos[3:] = [1.0, 0.0, 0.0, 0.0]
        self._utils.set_joint_qpos(self.model, self.data, "object0:joint", spoon_qpos)

        self._mujoco.mj_forward(self.model, self.data)

        # Lock spoon to gripper BEFORE settle so it cannot fall freely into the bowl
        self._update_weld_constraint()
        self.model.eq_active0[self._weld_eq_id] = 1
        self.data.eq_active[self._weld_eq_id] = 1

        # Reset bowl upright at its rest position
        bowl_qpos = np.array([1.525, 0.825, 0.410, 1.0, 0.0, 0.0, 0.0])
        self._utils.set_joint_qpos(self.model, self.data, "stir_bowl:joint", bowl_qpos)

        # Place all 40 balls in their initial layer positions
        for i, pos in enumerate(_BALL_INIT_POSITIONS):
            joint_name = f"object{i + 1}:joint"
            ball_qpos = np.array([pos[0], pos[1], pos[2], 1.0, 0.0, 0.0, 0.0])
            self._utils.set_joint_qpos(self.model, self.data, joint_name, ball_qpos)

        self._mujoco.mj_forward(self.model, self.data)

        # Settle: balls stack naturally, bowl stabilises; spoon held rigidly by weld
        for _ in range(300):
            self._mujoco.mj_step(self.model, self.data)

        return True
