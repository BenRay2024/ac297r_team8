"""This module tracks the state odf scene and scen elements like pedestrians, groups and obstacles"""
from typing import List

import numpy as np

from pysocialforce.utils import stateutils


class PedState:
    """Tracks the state of pedstrains and social groups"""

    def __init__(self, state, groups, config):
        self.default_tau = config("tau", 0.5)
        self.step_width = config("step_width", 0.4)
        self.agent_radius = config("agent_radius", 0.35)
        self.max_speed_multiplier = config("max_speed_multiplier", 1.3)

        self.max_speeds = None
        self.initial_speeds = None

        self.ped_states = []
        self.group_states = []
        self.reached_goal = np.full(state.shape[0], False)  # Initialize with False

        self.update(state, groups)

    def update(self, state, groups):
        self.state = state
        self.groups = groups

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        tau = self.default_tau * np.ones(state.shape[0])
        if state.shape[1] < 7:
            self._state = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1)
        else:
            self._state = state
        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds
        self.ped_states.append(self._state.copy())

    def get_states(self):
        # Find the maximum number of pedestrians at any timestep
        max_pedestrians = max(len(timestep) for timestep in self.ped_states)

        # Initialize the placeholder for absent pedestrians
        placeholder = [0] * 7  # Assuming a state has 7 elements

        padded_states = []
        for timestep in self.ped_states:
            # Copy current timestep's state to avoid modifying the original data
            padded_timestep = list(timestep)
            
            # Check if padding is needed for this timestep
            while len(padded_timestep) < max_pedestrians:
                padded_timestep.append(placeholder)
            
            padded_states.append(padded_timestep)

        return np.array(padded_states), self.group_states


    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        return self.state[:, 4:6]

    def tau(self):
        return self.state[:, 6:7]

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def step(self, force, groups=None):
        # Calculate desired velocity and update state
        desired_velocity = self.vel() + self.step_width * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)

        next_state = self.state.copy()
        next_state[:, 0:2] += desired_velocity * self.step_width
        next_state[:, 2:4] = desired_velocity

        # Check if pedestrians have reached their goal
        goal_distance = np.linalg.norm(next_state[:, 0:2] - self.goal(), axis=1)
        goal_tolerance = 1
        arrived_indices = (goal_distance < goal_tolerance) | (next_state[:, 1] <= 0)

        # Update reached_goal status
        next_state[self.reached_goal, 0:2] = [-100, -100]  # Keep pedestrians who reached the goal at (0,0)
        self.reached_goal = self.reached_goal | arrived_indices

        next_groups = self.groups
        if groups is not None:
            next_groups = groups
        self.update(next_state, next_groups)


    # def initial_speeds(self):
    #     return stateutils.speeds(self.ped_states[0])

    def desired_directions(self):
        return stateutils.desired_directions(self.state)[0]

    def add_pedestrians(self, new_state, new_group=None):
        # Extend the pedestrian state array
        tau = self.default_tau * np.ones((new_state.shape[0], 1))
        if new_state.shape[1] < 7:
            new_state = np.concatenate((new_state, tau), axis=1)
        combined_state = np.concatenate((self._state, new_state), axis=0)

        self.update(combined_state, self.groups)

        # Recalculate speeds and other attributes for the new combined state
        self.initial_speeds = self.speeds()  # speeds() should now operate on the updated self._state
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds
        self.reached_goal = np.concatenate((self.reached_goal, np.array([False] * len(new_state))))
        # if new_group is not None:
        #     new_group_indices = [i + self.size() - new_state.shape[0] for i in new_group]
        #     combined_groups = self.groups + [new_group_indices]
        #     self.update(combined_state, combined_groups)  # Update state with new groups
        # else:
        #     self.update(combined_state, self.groups)  # Update state without new groups

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> List[List]:
        return self._groups

    @groups.setter
    def groups(self, groups: List[List]):
        if groups is None:
            self._groups = []
        else:
            self._groups = groups
        self.group_states.append(self._groups.copy())

    def has_group(self):
        return self.groups is not None

    # def get_group_by_idx(self, index: int) -> np.ndarray:
    #     return self.state[self.groups[index], :]

    def which_group(self, index: int) -> int:
        """find group index from ped index"""
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1


class EnvState:
    """State of the environment obstacles"""

    def __init__(self, obstacles, resolution=10):
        self.resolution = resolution
        self.obstacles = obstacles

    @property
    def obstacles(self) -> List[np.ndarray]:
        """obstacles is a list of np.ndarray"""
        return self._obstacles

    @obstacles.setter
    def obstacles(self, obstacles):
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if obstacles is None:
            self._obstacles = []
        else:
            self._obstacles = []
            for startx, endx, starty, endy in obstacles:
                samples = int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution)
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples))
                    )
                )
                self._obstacles.append(line)
