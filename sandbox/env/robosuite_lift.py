"""
gym wrapper code from the robosuite repository: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
"""

import numpy as np
from gym import spaces
from gym.core import Env
from robosuite.wrappers import Wrapper

class GymWrapper(Wrapper, Env):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module
    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to robot-state and object-state.
    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        self.env.spec = type(self.env).__name__

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_robot-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.metadata = None

        # set up observation and action spaces
        obs = self.env.reset()
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        flat_ob = self._flatten_obs(obs)
        self.obs_dim = flat_ob.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)
        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.
        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed
        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.
        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._flatten_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.
        Args:
            action (np.array): Action to take in environment
        Returns:
            4-tuple:
                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        return self._flatten_obs(ob_dict), reward, done, info