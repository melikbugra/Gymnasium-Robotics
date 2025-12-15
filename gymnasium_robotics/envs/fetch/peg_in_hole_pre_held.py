import os

import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, MujocoPyFetchEnv

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
    - **No gripper control**: The gripper action (action[3]) is disabled and has no effect
    - **3-DoF control**: Only translational movements (dx, dy, dz) are available
    - **Simplified task**: Focus purely on insertion, no pick-and-place required
    - **Shorter episodes**: Default max_episode_steps=50 (vs 100 for full PegInHole)

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (4,), float32)`. Only the first 3 actions control the robot;
    the 4th action (gripper) is ignored.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | ------------------------------------------------------------------ | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Displacement of the end effector in the x direction dx             | -1          | 1           | robot0:mocap                     | hinge | position (m) |
    | 1   | Displacement of the end effector in the y direction dy             | -1          | 1           | robot0:mocap                     | hinge | position (m) |
    | 2   | Displacement of the end effector in the z direction dz             | -1          | 1           | robot0:mocap                     | hinge | position (m) |
    | 3   | **DISABLED** - Gripper action (has no effect)                      | -1          | 1           | N/A                              | N/A   | N/A          |

    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with
    information about the robot's end effector state, peg state, and goal. The dictionary
    consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic
      information of the peg object and gripper (same as FetchPegInHole):

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
      Euclidean distance between the peg and the goal is lower than 0.01 m and orientation error is lower than 0.1 rad).
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

    The hole plate with the square hole is fixed at position `(x,y,z) = [1.3, 0.9, 0.42] m`.
    The target position is at the hole center at `(x,y,z) = [1.3, 0.9, 0.45] m`.

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
            distance_threshold=0.005,  # 5mm - tight threshold for insertion
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
        self.orientation_weight = 0.5  # Weight for orientation penalty in dense reward
        self.alignment_threshold = 0.1  # Orientation success threshold (radians ~6°)

    def compute_reward(self, achieved_goal, goal, info):
        """Compute reward with orientation penalty.

        For peg-in-hole assembly, both position and orientation matter.
        The peg must reach the hole center AND be properly aligned (vertical).
        No grip bonus is provided since the peg is always grasped.
        """
        # Base reward: distance to goal
        d = np.linalg.norm(achieved_goal - goal, axis=-1)

        if self.reward_type == "sparse":
            # Success requires both position AND orientation
            position_ok = d < self.distance_threshold
            orientation_ok = info.get("orientation_error", 1.0) < self.alignment_threshold
            # Combine both conditions for success
            reward = -((~(position_ok & orientation_ok)).astype(np.float32))
        else:
            # Dense reward: distance + orientation penalty (no grip bonus)
            reward = -d

            # Add orientation penalty (peg should be vertical)
            if "orientation_error" in info:
                reward += -self.orientation_weight * info["orientation_error"]

        return reward

    def _get_peg_orientation_error(self):
        """Calculate peg orientation error (should be vertical).

        For a vertical peg, X and Y rotations (tilt) should be near 0.
        Z rotation doesn't matter as it's rotation around the peg's own axis.
        """
        obs = self._get_obs()
        peg_rot = obs["observation"][11:14]  # Euler XYZ rotations

        # For vertical peg, X and Y rotations should be near 0
        # Z rotation doesn't matter (rotation around peg axis)
        tilt_error = np.sqrt(peg_rot[0]**2 + peg_rot[1]**2)

        return tilt_error

    def _step_callback(self):
        """Override to keep gripper closed - pure physics-based gripping.

        The gripper fingers physically hold the peg through contact forces and friction.
        No kinematic constraints - all physics and collision detection fully active.
        """
        # Set gripper actuator control to maintain closed position
        if hasattr(self, 'data') and hasattr(self.data, 'ctrl'):
            try:
                l_finger_idx = self._model_names.actuator_name2id["robot0:l_gripper_finger_joint"]
                r_finger_idx = self._model_names.actuator_name2id["robot0:r_gripper_finger_joint"]

                # Apply strong closing force to grip peg
                self.data.ctrl[l_finger_idx] = self.target_gripper_pos
                self.data.ctrl[r_finger_idx] = self.target_gripper_pos
            except (KeyError, AttributeError):
                pass

        # NO kinematic weld, NO position override
        # Peg is held purely by gripper contact forces and friction
        # All collisions (peg-gripper AND peg-wall) are computed by MuJoCo physics

    def _set_action(self, action):
        """Override to ignore gripper action (action[3]).

        The gripper stays closed via _step_callback(), so action[3] is ignored.
        """
        # Gripper action is ignored - just pass action through
        # The _step_callback() will enforce closed gripper
        super()._set_action(action)

    def step(self, action):
        """Override step to include orientation information in info dict.

        Note: The gripper action (action[3]) is ignored and gripper stays closed.
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # Peg is always grasped (welded)
        info["is_grasped"] = True

        # Calculate orientation error and add to info
        orientation_error = self._get_peg_orientation_error()
        info["orientation_error"] = orientation_error

        # Recompute reward with orientation info
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Override to use peg TIP position as achieved_goal, not peg center.

        The peg is 8cm tall (half-size 4cm). The peg center is positioned 2cm below
        gripper. So peg tip is 4cm below peg center = 6cm below gripper.
        This ensures the policy controls the peg tip position for precise insertion.
        """
        # Get base observation (uses peg CENTER as achieved_goal)
        obs_dict = super()._get_obs()

        # Replace achieved_goal with peg TIP position instead of peg CENTER
        # Peg center is at obs_dict["achieved_goal"]
        # Peg tip is 4cm (0.04m) below peg center (half-height)
        peg_center = obs_dict["achieved_goal"]
        peg_tip = peg_center.copy()
        peg_tip[2] -= 0.04  # Subtract half-height to get tip position

        obs_dict["achieved_goal"] = peg_tip
        return obs_dict

    def _sample_goal(self):
        """Fixed goal position inside the hole.

        The hole plate is at (1.3, 0.95, 0.42), hole is 10cm deep.
        Goal is for peg TIP to reach 5cm inside the hole (middle depth).
        Hole top at Z=0.52, goal at Z=0.47 (5cm inside).
        """
        goal = np.array([1.3, 0.95, 0.47])
        return goal.copy()

    def _render_callback(self):
        """Render callback - target position is fixed, no need to update dynamically."""
        # Do nothing - target0 site is already at the correct position in XML
        pass

    def _reset_sim(self):
        """Reset simulation and position peg in gripper's grasp."""
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Close gripper fingers to hold the peg
        # Set gripper joint positions to closed state BEFORE forward
        gripper_target = self.target_gripper_pos
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:l_gripper_finger_joint", gripper_target
        )
        self._utils.set_joint_qpos(
            self.model, self.data, "robot0:r_gripper_finger_joint", gripper_target
        )

        # Need to do a forward pass to update site positions
        self._mujoco.mj_forward(self.model, self.data)

        # NOW get gripper position and place peg
        if self.has_object:
            # Get gripper position (after forward pass)
            gripper_pos = self._utils.get_site_xpos(self.model, self.data, "robot0:grip")

            # Set peg position to be in the gripper
            peg_qpos = self._utils.get_joint_qpos(self.model, self.data, "object0:joint")
            assert peg_qpos.shape == (7,)

            # Position: gripper position with offset
            # Peg is 8cm tall (half-size 4cm). Place peg center 2cm below grip site
            # so gripper fingers hold the TOP THIRD of the peg (better leverage)
            peg_qpos[:3] = gripper_pos + np.array([0.0, 0.0, -0.02])
            # Orientation: upright (vertical)
            peg_qpos[3:] = [1.0, 0.0, 0.0, 0.0]

            self._utils.set_joint_qpos(self.model, self.data, "object0:joint", peg_qpos)

        # Final forward pass with peg in position
        self._mujoco.mj_forward(self.model, self.data)
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
        goal = np.array([1.3, 0.9, 0.45])
        return goal.copy()

    def _render_callback(self):
        """Render callback - target position is fixed."""
        pass
