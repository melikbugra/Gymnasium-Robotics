import os

import numpy as np
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.fetch import MujocoFetchEnv, MujocoPyFetchEnv

# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join("fetch", "assembly.xml")


class MujocoFetchAssemblyEnv(MujocoFetchEnv, EzPickle):
    """
    ## Description

    "Assembly" environment for the Fetch robot. The task is for the robot to pick up a rectangular
    prism (box) and insert it into a square hole on top of a container box.

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
    information about the robot's end effector state, prism state, and goal. The dictionary
    consists of the following 3 keys:

    * `observation`: its value is an `ndarray` of shape `(25,)`. It consists of kinematic
      information of the prism object and gripper:

    | Num | Observation                                              | Min    | Max    | Unit                     |
    |-----|----------------------------------------------------------|--------|--------|--------------------------|
    | 0   | End effector x position in global coordinates            | -Inf   | Inf    | position (m)             |
    | 1   | End effector y position in global coordinates            | -Inf   | Inf    | position (m)             |
    | 2   | End effector z position in global coordinates            | -Inf   | Inf    | position (m)             |
    | 3   | Prism x position in global coordinates                   | -Inf   | Inf    | position (m)             |
    | 4   | Prism y position in global coordinates                   | -Inf   | Inf    | position (m)             |
    | 5   | Prism z position in global coordinates                   | -Inf   | Inf    | position (m)             |
    | 6   | Relative prism x position w.r.t. gripper                 | -Inf   | Inf    | position (m)             |
    | 7   | Relative prism y position w.r.t. gripper                 | -Inf   | Inf    | position (m)             |
    | 8   | Relative prism z position w.r.t. gripper                 | -Inf   | Inf    | position (m)             |
    | 9   | Joint displacement of the right gripper finger           | -Inf   | Inf    | position (m)             |
    | 10  | Joint displacement of the left gripper finger            | -Inf   | Inf    | position (m)             |
    | 11  | Global x rotation of the prism (Euler)                   | -Inf   | Inf    | angle (rad)              |
    | 12  | Global y rotation of the prism (Euler)                   | -Inf   | Inf    | angle (rad)              |
    | 13  | Global z rotation of the prism (Euler)                   | -Inf   | Inf    | angle (rad)              |
    | 14  | Relative prism linear velocity in x direction            | -Inf   | Inf    | velocity (m/s)           |
    | 15  | Relative prism linear velocity in y direction            | -Inf   | Inf    | velocity (m/s)           |
    | 16  | Relative prism linear velocity in z direction            | -Inf   | Inf    | velocity (m/s)           |
    | 17  | Prism angular velocity along x axis                      | -Inf   | Inf    | angular velocity (rad/s) |
    | 18  | Prism angular velocity along y axis                      | -Inf   | Inf    | angular velocity (rad/s) |
    | 19  | Prism angular velocity along z axis                      | -Inf   | Inf    | angular velocity (rad/s) |
    | 20  | End effector linear velocity x direction                 | -Inf   | Inf    | velocity (m/s)           |
    | 21  | End effector linear velocity y direction                 | -Inf   | Inf    | velocity (m/s)           |
    | 22  | End effector linear velocity z direction                 | -Inf   | Inf    | velocity (m/s)           |
    | 23  | Right gripper finger linear velocity                     | -Inf   | Inf    | velocity (m/s)           |
    | 24  | Left gripper finger linear velocity                      | -Inf   | Inf    | velocity (m/s)           |

    * `desired_goal`: this key represents the final goal to be achieved. In this environment
      it is a 3-dimensional `ndarray`, `(3,)`, that consists of the three cartesian coordinates
      of the target position inside the assembly box `[x,y,z]`. This is a fixed position.

    * `achieved_goal`: this key represents the current state of the prism, as if it would have
      achieved a goal. The value is an `ndarray` with shape `(3,)` representing the current
      prism position `[x,y,z]`.

    ## Rewards

    The reward can be initialized as `sparse` or `dense`:
    - *sparse*: the returned reward can have two values: `-1` if the prism hasn't reached its
      final target position inside the box, and `0` if the prism is in the final target position
      (the prism is considered to have reached the goal if the Euclidean distance between the
      prism and the goal is lower than 0.05 m).
    - *dense*: the returned reward is the negative Euclidean distance between the achieved goal
      position (prism position) and the desired goal (target inside the box).

    To initialize this environment with one of the mentioned reward functions the type of reward
    must be specified in the id string when the environment is initialized. For `sparse` reward
    the id is the default of the environment, `FetchAssembly-v1`. However, for `dense` reward
    the id must be modified to `FetchAssemblyDense-v1` and initialized as follows:

    ```python
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)

    env = gym.make('FetchAssemblyDense-v1')
    ```

    ## Starting State

    When the environment is reset the gripper is placed in the following global cartesian
    coordinates `(x,y,z) = [1.3419 0.7491 0.555] m`, and its orientation in quaternions is
    `(w,x,y,z) = [1.0, 0.0, 1.0, 0.0]`.

    The prism's position is randomized on top of the table. The initial `(x,y)` position of
    the prism is the gripper's x and y coordinates plus an offset sampled from a uniform
    distribution with a range of `[-0.15, 0.15] m`.

    The assembly box with the square hole is fixed at position `(x,y,z) = [1.3, 0.9, 0.50] m`.
    The target position is inside the box at `(x,y,z) = [1.3, 0.9, 0.42] m`.

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

    env = gym.make('FetchAssembly-v1', max_episode_steps=100)
    ```

    ## Version History

    * v1: Initial version.
    """

    def __init__(self, reward_type: str = "sparse", **kwargs):
        initial_qpos = {
            "robot0:slide0": 0.4049,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
            "object0:joint": [1.3, 0.6, 0.45, 1.0, 0.0, 0.0, 0.0],
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
            obj_range=0.15,
            target_range=0.0,  # No randomization for target - fixed position
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

        # Grip reward bonus (small enough not to dominate, large enough to matter)
        self.grip_reward = 0.1

    def compute_reward(self, achieved_goal, goal, info):
        """Compute reward with optional grip bonus.

        The grip bonus encourages the robot to grasp the prism, helping it learn
        that grasping is a necessary step toward the goal.
        """
        # Base reward: distance to goal
        d = np.linalg.norm(achieved_goal - goal, axis=-1)

        if self.reward_type == "sparse":
            reward = -(d > self.distance_threshold).astype(np.float32)
        else:
            reward = -d

        # Add grip bonus if info contains grip information
        # This is computed in step() and passed via info dict
        if isinstance(info, dict) and info.get("is_grasped", False):
            reward = reward + self.grip_reward

        return reward

    def _is_grasped(self):
        """Check if the robot is currently grasping the prism."""
        obs = self._get_obs()

        # Get gripper finger positions (from observation)
        # obs['observation'][9:11] are the gripper finger joint positions
        gripper_state = obs["observation"][9:11]
        gripper_opening = gripper_state[0] + gripper_state[1]

        # Get relative position of prism to gripper
        # obs['observation'][6:9] is relative position
        object_rel_pos = obs["observation"][6:9]
        dist_to_gripper = np.linalg.norm(object_rel_pos)

        # Get prism z position - must be lifted off the table
        prism_z = obs["observation"][5]  # object z position
        lifted = prism_z > 0.44  # Table is at ~0.42

        # Grasped if: gripper is closed, prism is close, and prism is lifted
        is_grasped = (
            gripper_opening < 0.04  # Gripper closed (max open ~0.1)
            and dist_to_gripper < 0.05  # Prism close to gripper
            and lifted  # Prism is lifted
        )

        return is_grasped

    def step(self, action):
        """Override step to include grip information in info dict."""
        obs, reward, terminated, truncated, info = super().step(action)

        # Check if grasped and add to info
        is_grasped = self._is_grasped()
        info["is_grasped"] = is_grasped

        # Recompute reward with grip bonus
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info

    def _sample_goal(self):
        """Fixed goal position inside the assembly box.

        The goal is at the bottom center of the assembly box.
        Box position: (1.3, 0.9, 0.50), target offset inside: (0, 0, -0.08)
        """
        goal = np.array([1.3, 0.9, 0.42])
        return goal.copy()

    def _render_callback(self):
        """Render callback - target position is fixed, no need to update dynamically."""
        # Do nothing - target0 site is already at the correct position in XML
        pass

    def _reset_sim(self):
        """Reset simulation and randomize prism position, avoiding the assembly box area."""
        self._mujoco.mj_resetData(self.model, self.data)

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        # Assembly box position (x, y) = (1.3, 0.9), size ~0.1m
        # Prism should not spawn inside or too close to the box
        box_x, box_y = 1.3, 0.9
        box_margin = 0.12  # Keep prism at least this far from box center

        # Randomize start position of object, avoiding the box area
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2].copy()
            max_attempts = 100
            for _ in range(max_attempts):
                # Sample random position
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(
                    -self.obj_range, self.obj_range, size=2
                )
                # Check if too close to gripper
                dist_to_gripper = np.linalg.norm(
                    object_xpos - self.initial_gripper_xpos[:2]
                )
                # Check if too close to box
                dist_to_box = np.linalg.norm(object_xpos - np.array([box_x, box_y]))

                if dist_to_gripper >= 0.1 and dist_to_box >= box_margin:
                    break

            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True


class MujocoPyFetchAssemblyEnv(MujocoPyFetchEnv, EzPickle):
    """MujocoPy version of the FetchAssembly environment. See MujocoFetchAssemblyEnv for documentation."""

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
            obj_range=0.15,
            target_range=0.0,  # No randomization for target - fixed position
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            **kwargs,
        )
        EzPickle.__init__(self, reward_type=reward_type, **kwargs)

    def _sample_goal(self):
        """Fixed goal position inside the assembly box."""
        goal = np.array([1.3, 0.9, 0.42])
        return goal.copy()

    def _render_callback(self):
        """Render callback - target position is fixed."""
        pass
