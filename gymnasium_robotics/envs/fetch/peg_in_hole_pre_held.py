import os

import gymnasium as gym
import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, MujocoPyFetchEnv
from gymnasium_robotics.utils.rotations import euler2quat, quat2euler, quat_mul

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "peg_in_hole_pre_held.xml")


class MujocoFetchPegInHolePreHeldEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description

    "PegInHolePreHeld" environment for the Fetch robot. This is a simplified version of the PegInHole
    task where the robot starts with the peg already grasped. The task focuses purely on the insertion
    phase of the assembly, requiring precise position control and orientation alignment without the
    complexity of grasping.

    The robot is a 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) with a two-fingered
    parallel gripper. The robot is controlled by small displacements of the gripper in Cartesian
    coordinates and the inverse kinematics are computed internally by the MuJoCo framework.

    The control frequency of the robot is of `f = 25 Hz`. This is achieved by applying the same
    action in 20 subsequent simulator step (with a time step of `dt = 0.002 s`) before returning
    the control to the robot.

    ## Key Differences from FetchPegInHole

    - **Pre-grasped peg**: The peg is welded to the gripper at the start via an equality constraint
    - **No gripper control**: The gripper stays closed automatically
    - **4-DoF control**: Translational movements (dx, dy, dz) + Z-axis rotation (wrist roll)
    - **Simplified task**: Focus purely on insertion, no pick-and-place required
    - **Shorter episodes**: Default max_episode_steps=50 (vs 100 for full PegInHole)

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (4,), float32)`. Actions control position and Z-axis rotation.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | ------------------------------------------------------------------ | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Displacement of the end effector in the x direction dx             | -1          | 1           | robot0:mocap                     | hinge | position (m) |
    | 1   | Displacement of the end effector in the y direction dy             | -1          | 1           | robot0:mocap                     | hinge | position (m) |
    | 2   | Displacement of the end effector in the z direction dz             | -1          | 1           | robot0:mocap                     | hinge | position (m) |
    | 3   | Rotation of end effector around Z-axis (wrist roll) dtheta         | -1          | 1           | robot0:mocap                     | hinge | angle (rad)  |

    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with
    information about the robot's end effector state, peg state, and goal. The dictionary
    consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(49,)`. It consists of kinematic
      information of the peg object, gripper, and four overhead obstacle positions:

    | Num | Observation                                              | Min    | Max    | Unit                     |
    |-----|----------------------------------------------------------|--------|--------|--------------------------|
    | 0   | End effector x position in global coordinates            | -Inf   | Inf    | position (m)             |
    | 1   | End effector y position in global coordinates            | -Inf   | Inf    | position (m)             |
    | 2   | End effector z position in global coordinates            | -Inf   | Inf    | position (m)             |
    | 3   | Peg x position in global coordinates                     | -Inf   | Inf    | position (m)             |
    | 4   | Peg y position in global coordinates                     | -Inf   | Inf    | position (m)             |
    | 5   | Peg z position in global coordinates                     | -Inf   | Inf    | position (m)             |
    | 6   | Relative peg x position w.r.t. gripper                   | -Inf   | Inf    | position (m)             |
    | 7   | Relative peg y position w.r.t. gripper                   | -Inf   | Inf    | position (m)             |
    | 8   | Relative peg z position w.r.t. gripper                   | -Inf   | Inf    | position (m)             |
    | 9   | Joint displacement of the right gripper finger           | -Inf   | Inf    | position (m)             |
    | 10  | Joint displacement of the left gripper finger            | -Inf   | Inf    | position (m)             |
    | 11  | Global x rotation of the peg (Euler)                     | -Inf   | Inf    | angle (rad)              |
    | 12  | Global y rotation of the peg (Euler)                     | -Inf   | Inf    | angle (rad)              |
    | 13  | Global z rotation of the peg (Euler)                     | -Inf   | Inf    | angle (rad)              |
    | 14  | Relative peg linear velocity in x direction              | -Inf   | Inf    | velocity (m/s)           |
    | 15  | Relative peg linear velocity in y direction              | -Inf   | Inf    | velocity (m/s)           |
    | 16  | Relative peg linear velocity in z direction              | -Inf   | Inf    | velocity (m/s)           |
    | 17  | Peg angular velocity along x axis                        | -Inf   | Inf    | angular velocity (rad/s) |
    | 18  | Peg angular velocity along y axis                        | -Inf   | Inf    | angular velocity (rad/s) |
    | 19  | Peg angular velocity along z axis                        | -Inf   | Inf    | angular velocity (rad/s) |
    | 20  | End effector linear velocity x direction                 | -Inf   | Inf    | velocity (m/s)           |
    | 21  | End effector linear velocity y direction                 | -Inf   | Inf    | velocity (m/s)           |
    | 22  | End effector linear velocity z direction                 | -Inf   | Inf    | velocity (m/s)           |
    | 23  | Right gripper finger linear velocity                     | -Inf   | Inf    | velocity (m/s)           |
    | 24  | Left gripper finger linear velocity                      | -Inf   | Inf    | velocity (m/s)           |
    | 25-27 | Right obstacle (x,y,z) absolute position              | -Inf   | Inf    | position (m)             |
    | 28-30 | Left obstacle (x,y,z) absolute position               | -Inf   | Inf    | position (m)             |
    | 31-33 | Front obstacle (x,y,z) absolute position              | -Inf   | Inf    | position (m)             |
    | 34-36 | Back obstacle (x,y,z) absolute position               | -Inf   | Inf    | position (m)             |
    | 37-39 | Right obstacle (x,y,z) relative to gripper            | -Inf   | Inf    | position (m)             |
    | 40-42 | Left obstacle (x,y,z) relative to gripper             | -Inf   | Inf    | position (m)             |
    | 43-45 | Front obstacle (x,y,z) relative to gripper            | -Inf   | Inf    | position (m)             |
    | 46-48 | Back obstacle (x,y,z) relative to gripper             | -Inf   | Inf    | position (m)             |

    * `desired_goal`: this key represents the final goal to be achieved. In this environment
      it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates
      of the target position at the hole center `[x,y,z]`. This is a fixed position.

    * `achieved_goal`: this key represents the current state of the peg, as if it would have
      achieved a goal. The value is an `ndarray` with shape `(3,)` representing the current
      peg position `[x,y,z]`.

    ## Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the peg hasn't reached its
      final target position inside the hole AND is properly aligned (vertical), and `0` if the peg
      is in the final target position with proper orientation (the peg is considered successful if the
      Euclidean distance between the peg and the goal is lower than 0.02 m and orientation error is lower than 0.1 rad).
    - *dense*: the returned reward is the negative Euclidean distance between the achieved goal
      position (peg position) and the desired goal (hole center), with an additional penalty for
      orientation misalignment. No grip bonus is provided since the peg is always grasped.

    To initialize this environment with one of the mentioned reward functions the type of reward
    must be specified in the id string when the environment is initialized. For `sparse` reward
    the id is the default of the environment, `FetchPegInHolePreHeld-v1`. However, for `dense` reward
    the id must be modified to `FetchPegInHolePreHeldDense-v1` and initialized as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchPegInHolePreHeldDense-v1')
    ```

    ## Starting State

    When the environment is reset the gripper is placed in the following global cartesian
    coordinates `(x,y,z) = [1.3419 0.7491 0.555] m`, and its orientation in quaternions is
    `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`.

    The peg is welded to the gripper at a fixed relative position. The peg starts in a vertical
    (upright) orientation, aligned with the gripper.

    The hole plate with the square hole is fixed at position `(x,y,z) = [1.525, 0.825, 0.42] m`.
    The target position is at the hole center at `(x,y,z) = [1.525, 0.825, 0.45] m`.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps`
    which by default is set to 50 timesteps. The episode is never `terminated` since the task
    is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated`
    the `max_episode_steps` argument can be set at initialization:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchPegInHolePreHeld-v1', max_episode_steps=100)
    ```

    ## Version History

    * v1: Initial version.
    """

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            # Peg starts at gripper position, held by closed fingers
            "object0:joint": [1.3419, 0.7491, 0.515, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,  # Gripper NOT blocked, but we'll keep it closed
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.0,  # No randomization - peg starts in grasp
            target_range=0.0,  # No randomization for target - fixed hole position
            distance_threshold=0.02,  # 20mm - relaxed for RL learnability
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

        # Target gripper position (closed to hold peg)
        # Peg is 2cm wide (1cm half-size = 0.01m), gripper must compress into peg
        # For physics-based grip, fingers need to squeeze peg tightly
        self.target_gripper_pos = 0.0  # Fully closed - maximum grip force

        # Peg-in-hole specific parameters (no grip reward since always grasped)
        self.orientation_weight = 0.0  # Disabled: weld constraint handles alignment, penalty discourages insertion
        self.alignment_threshold = 0.1  # Orientation success threshold (radians ~6°)

        # Random spawn area (entire table including near hole at Y=0.825)
        self.spawn_range_x = (1.475, 1.575)  # X bounds (±5cm around hole)
        self.spawn_range_y = (0.775, 0.875)  # Y bounds (±5cm around hole)
        self.spawn_height = 0.555        # Default Z height (above table)
        self.spawn_height_above_plate = 0.62  # Higher Z when above hole plate (above walls)

        # Hole plate zone (plate center at X=1.525, Y=0.825, size 30x30cm)
        self.hole_plate_x = 1.525
        self.hole_plate_y = 0.825
        self.hole_plate_half_size = 0.15  # 15cm half-size

        # Decomposed reward scaling factors (Robosuite-inspired)
        self.k_lateral = 10.0    # tanh scaling for XY distance
        self.k_depth = 5.0       # tanh scaling for Z distance
        self.w_lateral = 1.0     # weight for lateral centering
        self.w_depth = 1.0       # weight for insertion depth
        self.w_alignment = 0.5   # weight for peg tilt alignment

        # Cache weld constraint ID for peg-gripper attachment
        self._weld_eq_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_EQUALITY, "peg_grip"
        )
        assert self._weld_eq_id >= 0, "Weld constraint 'peg_grip' not found in model"

        # Cache obstacle body IDs (lazy-init flag for _get_obs during super().__init__)
        self._obstacles_initialized = False
        self._init_obstacle_ids()

        # Override observation space: 25 base + 24 obstacle (4 obs × 3 abs + 4 obs × 3 rel) = 49
        obs_shape = 49
        self.observation_space = gym.spaces.Dict(
            dict(
                desired_goal=gym.spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype="float64"
                ),
                achieved_goal=gym.spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype="float64"
                ),
                observation=gym.spaces.Box(
                    -np.inf, np.inf, shape=(obs_shape,), dtype="float64"
                ),
            )
        )

    def _init_obstacle_ids(self):
        """Cache obstacle body IDs from the MuJoCo model."""
        self._obstacle_body_ids = {}
        for name in ["obstacle_right", "obstacle_left", "obstacle_front", "obstacle_back"]:
            bid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_BODY, name
            )
            assert bid >= 0, f"Body '{name}' not found"
            self._obstacle_body_ids[name] = bid
        self._obstacles_initialized = True

    def compute_reward(self, achieved_goal, goal, info):
        """Decomposed reward: lateral + depth + alignment.

        Robosuite-inspired: each axis provides a separate signal so the agent
        can distinguish lateral misalignment from insufficient insertion depth.
        """
        diff = achieved_goal - goal

        if self.reward_type == "sparse":
            d = np.linalg.norm(diff, axis=-1)
            position_ok = d < self.distance_threshold
            orientation_ok = info.get("orientation_error", 1.0) < self.alignment_threshold
            reward = -((~(position_ok & orientation_ok)).astype(np.float32))
        else:
            # Lateral: XY distance (centering over hole)
            d_xy = np.linalg.norm(diff[..., :2], axis=-1)
            lateral = 1.0 - np.tanh(self.k_lateral * d_xy)

            # Depth: Z distance (insertion depth)
            d_z = np.abs(diff[..., 2])
            depth = 1.0 - np.tanh(self.k_depth * d_z)

            # Alignment: peg tilt error (from info dict)
            tilt_error = info.get("orientation_error", 0.0)
            alignment = np.cos(tilt_error)

            reward = (self.w_lateral * lateral
                    + self.w_depth * depth
                    + self.w_alignment * alignment)

        return reward

    def _get_peg_orientation_error(self):
        """Calculate peg orientation error (should be vertical).

        Reads peg quaternion directly from MuJoCo joint data instead of
        calling _get_obs() again (avoids redundant full observation computation).

        For a vertical peg, X and Y rotations (tilt) should be near 0.
        Z rotation doesn't matter as it's rotation around the peg's own axis.
        """
        peg_quat = self._utils.get_joint_qpos(
            self.model, self.data, "object0:joint"
        )[3:7]
        peg_euler = quat2euler(peg_quat)

        # For vertical peg, X and Y rotations should be near 0
        # Z rotation doesn't matter (rotation around peg axis)
        tilt_error = np.sqrt(peg_euler[0]**2 + peg_euler[1]**2)

        return tilt_error

    def _reset_mocap2body_xpos(self):
        """Reset mocap positions, skipping non-mocap welds (e.g., peg-grip)."""
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
                continue  # Skip non-mocap welds (peg-grip)
            self.data.mocap_pos[mocap_id][:] = self.data.xpos[body_idx]
            self.data.mocap_quat[mocap_id][:] = self.data.xquat[body_idx]

    def _update_weld_constraint(self):
        """Compute and set weld relpose from current gripper-peg body positions."""
        grip_body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link"
        )
        peg_body_id = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, "object0"
        )

        grip_pos = self.data.xpos[grip_body_id]
        grip_mat = self.data.xmat[grip_body_id].reshape(3, 3)
        grip_quat = self.data.xquat[grip_body_id]
        peg_pos = self.data.xpos[peg_body_id]
        peg_quat = self.data.xquat[peg_body_id]

        # Relative position: peg in gripper's local frame
        rel_pos = grip_mat.T @ (peg_pos - grip_pos)

        # Relative quaternion: q_rel = q_grip^{-1} * q_peg
        grip_quat_inv = np.array([
            grip_quat[0], -grip_quat[1], -grip_quat[2], -grip_quat[3]
        ])
        rel_quat = quat_mul(grip_quat_inv, peg_quat)

        # eq_data layout: [anchor(3), relpose_pos(3), relpose_quat(4), torquescale(1)]
        self.model.eq_data[self._weld_eq_id, 3:6] = rel_pos
        self.model.eq_data[self._weld_eq_id, 6:10] = rel_quat

    def _step_callback(self):
        """Keep gripper closed. Peg held by weld constraint + finger contact."""
        if hasattr(self, 'data') and hasattr(self.data, 'ctrl'):
            try:
                l_finger_idx = self._model_names.actuator_name2id["robot0:l_gripper_finger_joint"]
                r_finger_idx = self._model_names.actuator_name2id["robot0:r_gripper_finger_joint"]
                self.data.ctrl[l_finger_idx] = self.target_gripper_pos
                self.data.ctrl[r_finger_idx] = self.target_gripper_pos
            except (KeyError, AttributeError):
                pass

    def _set_action(self, action):
        """Override to add Z-axis rotation control instead of gripper.

        action[0:3]: Position delta (dx, dy, dz)
        action[3]: Z-axis rotation delta (wrist roll)

        The gripper stays closed via _step_callback().
        """
        assert action.shape == (4,)
        action = action.copy()

        pos_ctrl = action[:3]
        rot_z_ctrl = action[3]  # Z-axis rotation delta

        pos_ctrl *= 0.05  # Position limit (same as base class)
        rot_z_ctrl *= 0.1  # Rotation limit (~5.7 degrees/step max)

        # Get current mocap quaternion (aligned with gripper body)
        self._reset_mocap2body_xpos()
        current_quat = self.data.mocap_quat[0].copy()

        # Create rotation for wrist roll (rotating peg around its vertical axis)
        # The gripper is rotated 90 degrees around Y, so gripper's local X = world Z (peg axis)
        # Using euler [angle, 0, 0] rotates around local X axis = peg's vertical axis
        rot_quat = euler2quat(np.array([rot_z_ctrl, 0.0, 0.0]))

        # Apply rotation in gripper's local frame: current * local_rotation
        new_quat = quat_mul(current_quat, rot_quat)

        # Apply position delta
        self.data.mocap_pos[0] = self.data.mocap_pos[0] + pos_ctrl

        # Apply new quaternion directly (not as delta)
        self.data.mocap_quat[0] = new_quat

        # Gripper stays closed - set gripper control
        gripper_ctrl = np.array([0.0, 0.0])
        gripper_action = np.concatenate([pos_ctrl, [0, 0, 0, 0], gripper_ctrl])
        self._utils.ctrl_set_action(self.model, self.data, gripper_action)

    def step(self, action):
        """Override step to include orientation information in info dict.

        action[3] controls Z-axis rotation (wrist roll). Gripper stays closed.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        info["is_grasped"] = True

        orientation_error = self._get_peg_orientation_error()
        info["orientation_error"] = orientation_error

        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        # Obstacle positions in info
        for name, bid in self._obstacle_body_ids.items():
            info[f"{name}_pos"] = self.data.xpos[bid].copy()

        # Log decomposed reward components (dense mode only)
        if self.reward_type != "sparse":
            diff = obs["achieved_goal"] - self.goal
            d_xy = np.linalg.norm(diff[:2])
            d_z = np.abs(diff[2])
            info["reward_lateral"] = float(1.0 - np.tanh(self.k_lateral * d_xy))
            info["reward_depth"] = float(1.0 - np.tanh(self.k_depth * d_z))
            info["reward_alignment"] = float(np.cos(orientation_error))
            info["d_xy"] = float(d_xy)
            info["d_z"] = float(d_z)

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Override to use peg TIP position as achieved_goal and add obstacle observations.

        The peg is 8cm tall (half-size 4cm). The peg center is positioned 2cm below
        gripper. So peg tip is 4cm below peg center = 6cm below gripper.
        This ensures the policy controls the peg tip position for precise insertion.

        Observation is extended from 25D to 49D:
        - [0:25]  base observation (gripper, peg, velocities)
        - [25:28] obstacle_right absolute position
        - [28:31] obstacle_left absolute position
        - [31:34] obstacle_front absolute position
        - [34:37] obstacle_back absolute position
        - [37:40] obstacle_right position relative to gripper
        - [40:43] obstacle_left position relative to gripper
        - [43:46] obstacle_front position relative to gripper
        - [46:49] obstacle_back position relative to gripper
        """
        # Get base observation (uses peg CENTER as achieved_goal)
        obs_dict = super()._get_obs()

        # Replace achieved_goal with peg TIP position instead of peg CENTER
        peg_center = obs_dict["achieved_goal"]
        peg_tip = peg_center.copy()
        peg_tip[2] -= 0.04  # Subtract half-height to get tip position
        obs_dict["achieved_goal"] = peg_tip

        # Obstacle positions (may not be initialized during base class __init__)
        if getattr(self, "_obstacles_initialized", False):
            grip_pos = obs_dict["observation"][:3]
            abs_positions = []
            rel_positions = []
            for bid in self._obstacle_body_ids.values():
                pos = self.data.xpos[bid].copy()
                abs_positions.append(pos)
                rel_positions.append(pos - grip_pos)

            obs_dict["observation"] = np.concatenate([
                obs_dict["observation"],  # 25D base
                *abs_positions,           # 4 × 3D absolute
                *rel_positions,           # 4 × 3D relative to grip
            ])

        return obs_dict

    def _sample_goal(self):
        """Goal position inside the hole with small XY perturbation.

        The hole plate is at (1.525, 0.825, 0.42), hole is 10cm deep.
        Goal is for peg TIP to reach 5.5cm inside the hole.
        Hole top at Z=0.505, goal at Z=0.45 (5.5cm inside).

        Small XY randomization (±1mm) within 2mm clearance for valid HER relabeling.
        """
        goal = np.array([1.525, 0.825, 0.45])
        goal[0] += self.np_random.uniform(-0.001, 0.001)
        goal[1] += self.np_random.uniform(-0.001, 0.001)
        return goal.copy()

    def _render_callback(self):
        """Render callback - target position is fixed, no need to update dynamically."""
        # Do nothing - target0 site is already at the correct position in XML
        pass

    def _reset_sim(self):
        """Reset simulation and position peg in gripper's grasp at random location."""
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Disable weld during setup to avoid interference while positioning
        self.model.eq_active0[self._weld_eq_id] = 0

        # Close gripper fingers to hold the peg
        gripper_target = self.target_gripper_pos
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint", gripper_target
        )
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:r_gripper_finger_joint", gripper_target
        )

        self._mujoco.mj_forward(self.model, self.data)

        # Randomize gripper starting position within spawn area
        random_x = self.np_random.uniform(*self.spawn_range_x)
        random_y = self.np_random.uniform(*self.spawn_range_y)

        # Check if above hole plate - if so, spawn higher to clear walls
        above_plate_x = abs(random_x - self.hole_plate_x) < self.hole_plate_half_size
        above_plate_y = abs(random_y - self.hole_plate_y) < self.hole_plate_half_size
        if above_plate_x and above_plate_y:
            spawn_z = self.spawn_height_above_plate
        else:
            spawn_z = self.spawn_height

        random_pos = np.array([random_x, random_y, spawn_z])

        # Move mocap (gripper) to random position
        self._reset_mocap2body_xpos()
        self.data.mocap_pos[0] = random_pos

        # Step simulation to move gripper to new position
        for _ in range(300):
            self._mujoco.mj_step(self.model, self.data)

        # Position peg in gripper
        if self.has_object:
            gripper_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

            peg_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
            assert peg_qpos.shape == (7,)

            # Place peg center 2cm below grip site
            peg_qpos[:3] = gripper_pos + np.array([0.0, 0.0, -0.02])
            peg_qpos[3:] = [1.0, 0.0, 0.0, 0.0]

            self._utils.set_joint_qpos(self.model, self.data, "object0:joint", peg_qpos)

        self._mujoco.mj_forward(self.model, self.data)

        # Enable weld with relpose computed from current configuration
        self._update_weld_constraint()
        self.model.eq_active0[self._weld_eq_id] = 1

        return True


class MujocoPyFetchPegInHolePreHeldEnv(MujocoPyFetchEnv, EzPickle):
    """MujocoPy version of the FetchPegInHolePreHeld environment. See MujocoFetchPegInHolePreHeldEnv for documentation."""

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.3419, 0.7491, 0.515, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoPyFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.0,
            target_range=0.0,
            distance_threshold=0.01,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def _sample_goal(self):
        """Fixed goal position at hole center."""
        goal = np.array([1.525, 0.825, 0.45])
        return goal.copy()

    def _render_callback(self):
        """Render callback - target position is fixed."""
        pass
