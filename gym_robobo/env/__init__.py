import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
from collections import OrderedDict

import robobo

import logging
logger = logging.getLogger(__name__)


class RoboboStateSpace(spaces.Dict):
    """
    TODO:
        - Account for the position of the camera
        - Account for velocity
    """
    def __init__(self):
        super().__init__(spaces=self.get_spaces())

    def get_spaces(self):
        return {
            'position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            'front_cam': spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.int),
            'sensor_readings': spaces.Box(low=0, high=0.2, shape=(8,))
        }


class RoboboActionSpace(spaces.Dict):
    def __init__(self):
        super().__init__(spaces=self.get_spaces())

    def get_spaces(self):
        return {
            'speed': spaces.Box(low=-50, high=-50, shape=(2,)),
            'camera': spaces.Dict({
                'tilt': spaces.Box(low=0, high=np.pi / 2, shape=(1,)),
                'pan': spaces.Box(low=0, high=2 * np.pi, shape=(1,))
            })
        }


class RoboboEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        self.viewer = None
        self.server_process = None
        self.server_port = None
        self.rob = robobo.SimulationRobobo().connect()
        self.observation_space = RoboboStateSpace()
        self.action_space = RoboboActionSpace()
        self._initialize()

    def _initialize(self):
        self.rob.play_simulation()

    def step(self, action):
        # Execute one time step within the environment
        self._do_action(action)
        obs = self.get_observation()
        reward = self.get_reward(action, obs)
        done = self.determine_if_done()
        info = {}
        return obs, reward, done, info

    def get_observation(self):
        position = self._get_position()
        front_cam = self._get_front_cam()
        sensor_readings = self._get_sensor_readings()
        return OrderedDict(
            front_cam=front_cam,
            position=position,
            sensor_readings=sensor_readings
        )

    def get_reward(self, action, obs):
        # TODO: Implement reward function
        return 1

    def determine_if_done(self):
        # TODO: Implement function to determine if we're in a terminal state
        return False

    def _do_action(self, action):
        self.rob.move(action['speed'][0], action['speed'][1])
        self.rob.set_phone_pan(action['camera']['pan'][0], 0.1)
        self.rob.set_phone_tilt(action['camera']['tilt'][0], 0.1)

    def _get_position(self):
        return np.array(self.rob.position())

    def _get_front_cam(self):
        return self.rob.get_image_front()

    def _get_sensor_readings(self):
        sensor_readings = self.rob.read_irs()
        sensor_readings = np.where(not sensor_readings, 0, sensor_readings)  # Replace False values with 0
        return sensor_readings

    def reset(self):
        # Reset the state of the environment to an initial state
        pass

    def render(self, mode='human', close=False):
        """
        TODO: Write methods for optional rendering when training in headless mode
        :param mode:
        :param close:
        :return:
        """
        pass
