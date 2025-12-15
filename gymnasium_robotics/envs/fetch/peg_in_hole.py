import os

import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, MujocoPyFetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "peg_in_hole.xml")


class MujocoFetchPegInHoleEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description

    "PegInHole" environment for the Fetch robot. The task is for the robot to pick up a square peg
    and insert it into a square hole on a fixed plate. This is a classic contact-rich assembly task
    that requires precise position control and orientation alignment.

    The robot is a 7-DoF [Fetch Mobile Manipulator](https://fetchrobotics.com/) with a two-fingered
    parallel gripper. The robot is controlled by small displacements of the gripper in Cartesian
    coordinates and the inverse kinematics are computed internally by the MuJoCo framework.

    The control frequency of the robot is of `f = 25 Hz`. This is achieved by applying the same
    action in 20 subsequent simulator step (with a time step of `dt = 0.002 s`) before returning
    the control to the robot.

    ## Action Space

    The action space is a `Box(-1.0, 1.0, (4,), float32)`. An action represents the Cartesian
    displacement dx, dy, and dz of the end effector. In addition to a last action that controls
    closing and opening of the gripper.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file)                                | Joint | Unit         |
    | --- | ------------------------------------------------------------------ | ----------- | ----------- | --------------------------------------------------------------- | ----- | ------------ |
    | 0   | Displacement of the end effector in the x direction dx             | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 1   | Displacement of the end effector in the y direction dy             | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 2   | Displacement of the end effector in the z direction dz             | -1          | 1           | robot0:mocap                                                    | hinge | position (m) |
    | 3   | Gripper opening/closing (positive=open, negative=close)            | -1          | 1           | robot0:r_gripper_finger_joint, robot0:l_gripper_finger_joint    | hinge | position (m) |

    ## Observation Space

    The observation is a `goal-aware observation space`. It consists of a dictionary with
    information about the robot's end effector state, peg state, and goal. The dictionary
    consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic
      information of the peg object and gripper:

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
      position (peg position) and the desired goal (hole center), with additional penalties for
      orientation misalignment and a bonus for grasping the peg.

    To initialize this environment with one of the mentioned reward functions the type of reward
    must be specified in the id string when the environment is initialized. For `sparse` reward
    the id is the default of the environment, `FetchPegInHole-v1`. However, for `dense` reward
    the id must be modified to `FetchPegInHoleDense-v1` and initialized as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchPegInHoleDense-v1')
    ```

    ## Starting State

    When the environment is reset the gripper is placed in the following global cartesian
    coordinates `(x,y,z) = [1.3419 0.7491 0.555] m`, and its orientation in quaternions is
    `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`.

    The peg's position is randomized on top of the table in front of the hole plate. The initial `(x,y)` position of
    the peg is sampled from a uniform distribution with a range of approximately `[-0.10, 0.10] m`
    around a spawn center at `(1.35, 0.58)`.

    The hole plate with the square hole is fixed at position `(x,y,z) = [1.3, 0.9, 0.42] m`.
    The target position is at the hole center at `(x,y,z) = [1.3, 0.9, 0.45] m`.

    ## Episode End

    The episode will be `truncated` when the duration reaches a total of `max_episode_steps`
    which by default is set to 100 timesteps. The episode is never `terminated` since the task
    is continuing with infinite horizon.

    ## Arguments

    To increase/decrease the maximum number of timesteps before the episode is `truncated`
    the `max_episode_steps` argument can be set at initialization:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchPegInHole-v1', max_episode_steps=150)
    ```

    ## Version History

    * v1: Initial version.
    """

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.3, 0.6, 0.45, 1.0, 0.0, 0.0, 0.0],  # Peg upright (vertical)
        }
        MujocoFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.12,  # Peg spawn randomization range
            target_range=0.0,  # No randomization for target - fixed hole position
            distance_threshold=0.01,  # 1cm - tighter than default
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

        # Peg-in-hole specific parameters
        self.grip_reward = 0.1  # Bonus for grasping peg
        self.orientation_weight = 0.5  # Weight for orientation penalty in dense reward
        self.alignment_threshold = 0.1  # Orientation success threshold (radians ~6°)

    def compute_reward(self, achieved_goal, goal, info):
        """Compute reward with optional orientation penalty and grip bonus.

        For peg-in-hole assembly, both position and orientation matter.
        The peg must reach the hole center AND be properly aligned (vertical).
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
            # Dense reward: distance + orientation penalty + grip bonus
            reward = -d

            # Add orientation penalty (peg should be vertical)
            if "orientation_error" in info:
                reward += -self.orientation_weight * info["orientation_error"]

            # Add grip bonus only for dense reward
            if info.get("is_grasped", False):
                reward += self.grip_reward

        return reward

    def _is_grasped(self):
        """Check if the robot is currently grasping the peg."""
        obs = self._get_obs()

        # Get gripper finger positions (from observation)
        # obs['observation'][9:11] are the gripper finger joint positions
        gripper_state = obs["observation"][9:11]
        gripper_opening = gripper_state[0] + gripper_state[1]

        # Get relative position of peg to gripper
        # obs['observation'][6:9] is relative position
        peg_rel_pos = obs["observation"][6:9]
        dist_to_gripper = np.linalg.norm(peg_rel_pos)

        # Get peg z position - must be lifted off the table
        peg_z = obs["observation"][5]  # object z position
        lifted = peg_z > 0.44  # Table is at ~0.42

        # Grasped if: gripper is closed, peg is close, and peg is lifted
        is_grasped = (
            gripper_opening < 0.04  # Gripper closed (max open ~0.1)
            and dist_to_gripper < 0.05  # Peg close to gripper
            and lifted  # Peg is lifted
        )

        return is_grasped

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

    def step(self, action):
        """Override step to include orientation and grasp information in info dict."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if grasped and add to info
        is_grasped = self._is_grasped()
        info["is_grasped"] = is_grasped

        # Calculate orientation error and add to info
        orientation_error = self._get_peg_orientation_error()
        info["orientation_error"] = orientation_error

        # Recompute reward with orientation and grip info
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info

    def _sample_goal(self):
        """Fixed goal position at hole center.

        The hole plate is at (1.3, 0.9, 0.42), and the goal is at the hole entrance.
        The peg center should reach (1.3, 0.9, 0.45) to be considered inserted.
        """
        goal = np.array([1.3, 0.9, 0.45])
        return goal.copy()

    def _render_callback(self):
        """Render callback - target position is fixed, no need to update dynamically."""
        # Do nothing - target0 site is already at the correct position in XML
        pass

    def _reset_sim(self):
        """Reset simulation and randomize peg position."""
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Hole plate: center (1.3, 0.9), peg should spawn in front
        # Spawn peg on table, in front of hole plate (y < 0.7)
        if self.has_object:
            # Spawn center: slightly to the right and in front
            spawn_center = np.array([1.35, 0.58])
            peg_xpos = spawn_center + self.np_random.uniform(-0.10, 0.10, size=2)

            # Clamp to valid table area (away from hole plate)
            peg_xpos[0] = np.clip(peg_xpos[0], 1.20, 1.48)
            peg_xpos[1] = np.clip(peg_xpos[1], 0.48, 0.68)

            # Set peg position (upright orientation)
            peg_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert peg_qpos.shape == (7,)
            peg_qpos[:2] = peg_xpos
            # Z position: table height (0.42) + peg half-height (0.04) - small offset
            peg_qpos[2] = 0.45
            # Quaternion for upright orientation: (w, x, y, z) = (1, 0, 0, 0)
            peg_qpos[3:] = [1.0, 0.0, 0.0, 0.0]

            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", peg_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True


class MujocoPyFetchPegInHoleEnv(MujocoPyFetchEnv, EzPickle):
    """MujocoPy version of the FetchPegInHole environment. See MujocoFetchPegInHoleEnv for documentation."""

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.3, 0.6, 0.45, 1.0, 0.0, 0.0, 0.0],
        }
        MujocoPyFetchEnv.__init__(
            self,
            model_path=MODEL_XML_PATH,
            has_object=True,
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.12,
            target_range=0.0,  # No randomization for target - fixed hole position
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
