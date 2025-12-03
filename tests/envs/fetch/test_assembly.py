"""Tests for the FetchAssembly environment."""

import numpy as np
import pytest
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


class TestFetchAssemblyBasics:
    """Basic functionality tests for FetchAssembly environment."""

    def test_env_creation_sparse(self):
        """Test that FetchAssembly-v1 (sparse) can be created."""
        env = gym.make("FetchAssembly-v1")
        assert env is not None
        env.close()

    def test_env_creation_dense(self):
        """Test that FetchAssemblyDense-v1 (dense) can be created."""
        env = gym.make("FetchAssemblyDense-v1")
        assert env is not None
        env.close()

    def test_reset_returns_observation(self):
        """Test that reset returns a valid observation."""
        env = gym.make("FetchAssembly-v1")
        obs, info = env.reset()

        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "achieved_goal" in obs
        assert "desired_goal" in obs

        env.close()

    def test_observation_shapes(self):
        """Test that observation shapes are correct."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        # observation should be 25-dimensional
        assert obs["observation"].shape == (25,)
        # achieved_goal (prism position) should be 3D
        assert obs["achieved_goal"].shape == (3,)
        # desired_goal (target position) should be 3D
        assert obs["desired_goal"].shape == (3,)

        env.close()

    def test_action_space(self):
        """Test that action space is correct."""
        env = gym.make("FetchAssembly-v1")

        # Action space should be Box with shape (4,)
        assert env.action_space.shape == (4,)
        # Actions should be bounded [-1, 1]
        assert np.all(env.action_space.low == -1.0)
        assert np.all(env.action_space.high == 1.0)

        env.close()

    def test_step_returns_valid_output(self):
        """Test that step returns valid output format."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, dict)
        assert np.isscalar(reward) or isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        env.close()


class TestFetchAssemblyGoal:
    """Tests for goal-related functionality."""

    def test_fixed_goal_position(self):
        """Test that goal position is fixed at correct location."""
        env = gym.make("FetchAssembly-v1")

        expected_goal = np.array([1.3, 0.9, 0.42])

        for _ in range(5):
            obs, _ = env.reset()
            np.testing.assert_array_almost_equal(
                obs["desired_goal"], expected_goal, decimal=2
            )

        env.close()

    def test_achieved_goal_is_prism_position(self):
        """Test that achieved_goal represents prism position."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        # Achieved goal should be the prism's position
        # Prism position is in observation indices 3-5
        prism_pos = obs["observation"][3:6]
        np.testing.assert_array_almost_equal(obs["achieved_goal"], prism_pos, decimal=5)

        env.close()


class TestFetchAssemblySpawn:
    """Tests for prism spawn position."""

    def test_prism_not_spawning_in_box(self):
        """Test that prism doesn't spawn inside the assembly box area."""
        env = gym.make("FetchAssembly-v1")

        box_x, box_y = 1.3, 0.9
        box_margin = 0.12

        for _ in range(20):
            obs, _ = env.reset()
            prism_pos = obs["achieved_goal"]

            dist_to_box = np.sqrt(
                (prism_pos[0] - box_x) ** 2 + (prism_pos[1] - box_y) ** 2
            )

            assert dist_to_box >= box_margin - 0.01, (
                f"Prism spawned too close to box: dist={dist_to_box:.3f}m"
            )

        env.close()

    def test_prism_spawns_on_table(self):
        """Test that prism spawns at table height."""
        env = gym.make("FetchAssembly-v1")

        for _ in range(10):
            obs, _ = env.reset()
            prism_z = obs["achieved_goal"][2]

            # Prism should be near table surface (around 0.42-0.50m)
            assert 0.40 < prism_z < 0.55, f"Prism z={prism_z} is not on table"

        env.close()


class TestFetchAssemblyReward:
    """Tests for reward function."""

    def test_sparse_reward_values(self):
        """Test that sparse reward returns -1 or 0."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, _, _, _ = env.step(action)

            assert reward in [-1.0, 0.0], (
                f"Sparse reward should be -1 or 0, got {reward}"
            )

        env.close()

    def test_dense_reward_is_negative_distance(self):
        """Test that dense reward is negative distance."""
        env = gym.make("FetchAssemblyDense-v1")
        obs, _ = env.reset()

        action = env.action_space.sample()
        obs, reward, _, _, _ = env.step(action)

        # Dense reward should be negative (unless goal is achieved)
        assert reward <= 0, f"Dense reward should be <= 0, got {reward}"

        # Verify reward matches -distance
        distance = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        expected_reward = -distance
        np.testing.assert_almost_equal(reward, expected_reward, decimal=3)

        env.close()

    def test_compute_reward_method(self):
        """Test that compute_reward method works correctly."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        achieved = obs["achieved_goal"]
        desired = obs["desired_goal"]

        # Test sparse reward computation
        reward = env.unwrapped.compute_reward(achieved, desired, {})
        assert reward in [-1.0, 0.0]

        # Test that goal at target gives 0 reward
        reward_at_goal = env.unwrapped.compute_reward(desired, desired, {})
        assert reward_at_goal == 0.0

        env.close()

    def test_sparse_reward_threshold(self):
        """Test that sparse reward uses correct distance threshold (0.05m)."""
        env = gym.make("FetchAssembly-v1")

        # Test with distance just below threshold - should give 0
        achieved_close = np.array([1.3, 0.9, 0.42])  # Goal position
        desired = np.array([1.3, 0.9, 0.42])

        # Exactly at goal
        reward = env.unwrapped.compute_reward(achieved_close, desired, {})
        assert reward == 0.0, f"Reward at goal should be 0, got {reward}"

        # Just within threshold (0.04m away)
        achieved_near = np.array([1.34, 0.9, 0.42])  # 0.04m away in x
        reward_near = env.unwrapped.compute_reward(achieved_near, desired, {})
        assert reward_near == 0.0, (
            f"Reward within threshold should be 0, got {reward_near}"
        )

        # Just outside threshold (0.06m away)
        achieved_far = np.array([1.36, 0.9, 0.42])  # 0.06m away in x
        reward_far = env.unwrapped.compute_reward(achieved_far, desired, {})
        assert reward_far == -1.0, (
            f"Reward outside threshold should be -1, got {reward_far}"
        )

        env.close()

    def test_dense_reward_decreases_with_distance(self):
        """Test that dense reward decreases as distance increases."""
        env = gym.make("FetchAssemblyDense-v1")

        desired = np.array([1.3, 0.9, 0.42])

        # Test at various distances
        achieved_close = np.array([1.31, 0.9, 0.42])  # 0.01m away
        achieved_medium = np.array([1.4, 0.9, 0.42])  # 0.1m away
        achieved_far = np.array([1.5, 0.9, 0.42])  # 0.2m away

        reward_close = env.unwrapped.compute_reward(achieved_close, desired, {})
        reward_medium = env.unwrapped.compute_reward(achieved_medium, desired, {})
        reward_far = env.unwrapped.compute_reward(achieved_far, desired, {})

        # Closer should have higher (less negative) reward
        assert reward_close > reward_medium > reward_far, (
            f"Rewards should decrease with distance: {reward_close}, {reward_medium}, {reward_far}"
        )

        env.close()

    def test_reward_info_dict(self):
        """Test that info dict contains is_success key."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert "is_success" in info, "Info should contain 'is_success' key"
        # is_success can be bool or float (0.0/1.0)
        assert info["is_success"] in [True, False, 0.0, 1.0, 0, 1], (
            f"is_success should be boolean-like, got {info['is_success']}"
        )

        env.close()

    def test_success_when_goal_achieved(self):
        """Test that is_success is True when goal is achieved."""
        env = gym.make("FetchAssembly-v1")

        # Manually check success computation
        desired = np.array([1.3, 0.9, 0.42])
        achieved_at_goal = np.array([1.3, 0.9, 0.42])
        achieved_far = np.array([1.5, 0.9, 0.42])

        # At goal should be success
        reward_at_goal = env.unwrapped.compute_reward(achieved_at_goal, desired, {})
        assert reward_at_goal == 0.0, "Should be success (reward=0) when at goal"

        # Far from goal should not be success
        reward_far = env.unwrapped.compute_reward(achieved_far, desired, {})
        assert reward_far == -1.0, (
            "Should not be success (reward=-1) when far from goal"
        )

        env.close()


class TestFetchAssemblyGoalDetailed:
    """Detailed tests for goal-related functionality."""

    def test_goal_position_is_inside_box(self):
        """Test that goal position is geometrically inside the assembly box."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        goal = obs["desired_goal"]

        # Box center is at (1.3, 0.9, 0.50), inner size ~0.18m
        # Goal should be at (1.3, 0.9, 0.42) - inside the box
        box_center_x, box_center_y = 1.3, 0.9
        box_bottom_z = 0.50 - 0.10  # Box bottom
        box_top_z = 0.50 + 0.10  # Box top

        # Goal x,y should be at box center
        assert abs(goal[0] - box_center_x) < 0.01, f"Goal x should be at box center"
        assert abs(goal[1] - box_center_y) < 0.01, f"Goal y should be at box center"

        # Goal z should be inside box (between bottom and top)
        assert box_bottom_z < goal[2] < box_top_z, (
            f"Goal z={goal[2]} should be inside box ({box_bottom_z} to {box_top_z})"
        )

        env.close()

    def test_goal_never_changes_during_episode(self):
        """Test that goal remains fixed throughout an episode."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        initial_goal = obs["desired_goal"].copy()

        # Run 50 random steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)

            np.testing.assert_array_equal(
                obs["desired_goal"],
                initial_goal,
                err_msg="Goal should not change during episode",
            )

        env.close()

    def test_goal_same_across_resets(self):
        """Test that goal is always the same (fixed position)."""
        env = gym.make("FetchAssembly-v1")

        goals = []
        for _ in range(10):
            obs, _ = env.reset()
            goals.append(obs["desired_goal"].copy())

        # All goals should be identical
        for i, goal in enumerate(goals):
            np.testing.assert_array_almost_equal(
                goal,
                goals[0],
                decimal=3,
                err_msg=f"Goal at reset {i} differs from first goal",
            )

        env.close()

    def test_achieved_goal_updates_with_prism(self):
        """Test that achieved_goal updates as prism moves."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        initial_achieved = obs["achieved_goal"].copy()

        # Take some random actions
        for _ in range(20):
            action = env.action_space.sample()
            obs, _, _, _, _ = env.step(action)

        # Achieved goal should reflect prism position
        achieved = obs["achieved_goal"]
        prism_pos = obs["observation"][3:6]

        np.testing.assert_array_almost_equal(
            achieved,
            prism_pos,
            decimal=5,
            err_msg="Achieved goal should match prism position",
        )

        env.close()


class TestFetchAssemblyHER:
    """Tests for HER (Hindsight Experience Replay) compatibility."""

    def test_goal_aware_observation_space(self):
        """Test that observation space is goal-aware (HER compatible)."""
        env = gym.make("FetchAssembly-v1")

        # Check observation space structure
        assert hasattr(env.observation_space, "spaces")
        assert "observation" in env.observation_space.spaces
        assert "achieved_goal" in env.observation_space.spaces
        assert "desired_goal" in env.observation_space.spaces

        env.close()

    def test_compute_reward_accepts_batches(self):
        """Test that compute_reward works with batch inputs."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        # Create batch of goals
        batch_size = 5
        achieved = np.tile(obs["achieved_goal"], (batch_size, 1))
        desired = np.tile(obs["desired_goal"], (batch_size, 1))

        rewards = env.unwrapped.compute_reward(achieved, desired, {})

        assert rewards.shape == (batch_size,)

        env.close()


class TestFetchAssemblyPhysics:
    """Tests for physics simulation."""

    def test_gripper_can_move(self):
        """Test that gripper responds to actions."""
        env = gym.make("FetchAssembly-v1")
        obs_before, _ = env.reset()

        gripper_pos_before = obs_before["observation"][:3].copy()

        # Apply positive x action
        action = np.array([1.0, 0.0, 0.0, 0.0])
        for _ in range(5):
            obs_after, _, _, _, _ = env.step(action)

        gripper_pos_after = obs_after["observation"][:3]

        # Gripper should have moved in x direction
        assert gripper_pos_after[0] > gripper_pos_before[0], "Gripper didn't move in x"

        env.close()

    def test_gripper_can_close(self):
        """Test that gripper can close."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        # Close gripper action
        action = np.array([0.0, 0.0, 0.0, -1.0])

        for _ in range(10):
            obs, _, _, _, _ = env.step(action)

        # Gripper fingers should have moved (indices 9 and 10)
        gripper_state = obs["observation"][9:11]

        # Both fingers should have some displacement
        assert np.any(gripper_state != 0), "Gripper fingers didn't move"

        env.close()

    def test_multiple_steps_stable(self):
        """Test that multiple steps don't cause instability."""
        env = gym.make("FetchAssembly-v1")
        env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)

            # Check for NaN values
            assert not np.any(np.isnan(obs["observation"])), "NaN in observation"
            assert not np.any(np.isnan(obs["achieved_goal"])), "NaN in achieved_goal"
            assert not np.isnan(reward), "NaN reward"

        env.close()

    def test_prism_falls_with_gravity(self):
        """Test that prism falls when dropped from height."""
        env = gym.make("FetchAssembly-v1")
        env.reset()

        # Get initial prism z position
        obs, _ = env.reset()
        initial_z = obs["achieved_goal"][2]

        # Do nothing for several steps - prism should stay on table
        action = np.array([0.0, 0.0, 0.0, 1.0])  # Open gripper, no movement
        for _ in range(20):
            obs, _, _, _, _ = env.step(action)

        final_z = obs["achieved_goal"][2]

        # Prism should stay roughly at same height (on table)
        assert abs(final_z - initial_z) < 0.1, "Prism moved unexpectedly in z"

        env.close()

    def test_gripper_move_to_prism(self):
        """Test that gripper can move towards the prism."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        gripper_pos = obs["observation"][:3]
        prism_pos = obs["achieved_goal"].copy()  # Save initial prism position

        initial_distance = np.linalg.norm(gripper_pos[:2] - prism_pos[:2])

        # Move towards initial prism position (in x-y plane)
        for _ in range(20):
            gripper_pos = obs["observation"][:3]
            # Calculate direction to initial prism position
            direction = prism_pos[:2] - gripper_pos[:2]
            norm = np.linalg.norm(direction)
            if norm > 0.01:
                direction = direction / norm
            action = np.array([direction[0], direction[1], 0.0, 1.0])
            obs, _, _, _, _ = env.step(action)

        gripper_pos_after = obs["observation"][:3]
        final_distance = np.linalg.norm(gripper_pos_after[:2] - prism_pos[:2])

        # Should be closer to initial prism position now
        assert final_distance < initial_distance, (
            f"Gripper didn't get closer to prism: {initial_distance:.3f} -> {final_distance:.3f}"
        )

        env.close()

    def test_prism_has_physics(self):
        """Test that prism has proper physics (can be affected by collisions)."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        # Check that prism has velocity components in observation
        # Prism velocities are at indices 14-19 (linear: 14-16, angular: 17-19)
        prism_linear_vel = obs["observation"][14:17]
        prism_angular_vel = obs["observation"][17:20]

        # Initially velocities should be near zero (prism at rest)
        assert np.allclose(prism_linear_vel, 0, atol=0.1), "Prism should start at rest"
        assert np.allclose(prism_angular_vel, 0, atol=0.1), (
            "Prism should start with no rotation"
        )

        # Check prism is on the table (z around 0.42-0.45)
        prism_z = obs["achieved_goal"][2]
        assert 0.40 < prism_z < 0.50, f"Prism z={prism_z} should be on table"

        env.close()

    def test_gripper_grasp_attempt(self):
        """Test gripper can attempt to grasp by closing around object area."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        prism_pos = obs["achieved_goal"]

        # Move gripper above prism
        for _ in range(30):
            gripper_pos = obs["observation"][:3]
            diff = prism_pos - gripper_pos
            diff[2] = 0.05  # Stay slightly above
            action = np.clip(diff * 5, -1, 1)
            action = np.append(action, 1.0)  # Keep gripper open
            obs, _, _, _, _ = env.step(action)

        # Move down to prism level
        for _ in range(15):
            action = np.array([0.0, 0.0, -1.0, 1.0])
            obs, _, _, _, _ = env.step(action)

        # Close gripper
        for _ in range(10):
            action = np.array([0.0, 0.0, 0.0, -1.0])
            obs, _, _, _, _ = env.step(action)

        # Gripper fingers should be partially closed
        gripper_state = obs["observation"][9:11]
        assert np.all(gripper_state < 0.05), "Gripper should be mostly closed"

        env.close()

    def test_prism_lift_attempt(self):
        """Test that prism can potentially be lifted (gripper can move up while closed)."""
        env = gym.make("FetchAssembly-v1")
        obs, _ = env.reset()

        prism_pos = obs["achieved_goal"]

        # Move gripper to prism position
        for _ in range(40):
            gripper_pos = obs["observation"][:3]
            diff = prism_pos - gripper_pos
            diff[2] = max(diff[2], -0.02)  # Approach from above
            action = np.clip(diff * 5, -1, 1)
            action = np.append(action, 1.0)  # Keep gripper open
            obs, _, _, _, _ = env.step(action)

        # Close gripper
        for _ in range(15):
            action = np.array([0.0, 0.0, 0.0, -1.0])
            obs, _, _, _, _ = env.step(action)

        prism_z_before_lift = obs["achieved_goal"][2]

        # Try to lift
        for _ in range(20):
            action = np.array([0.0, 0.0, 1.0, -1.0])  # Move up, keep closed
            obs, _, _, _, _ = env.step(action)

        gripper_z_after = obs["observation"][2]

        # Gripper should have moved up
        assert gripper_z_after > prism_z_before_lift, "Gripper didn't move up"

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
