from abc import ABC, abstractmethod
import torch
from dm_control import suite
from torchrl.envs import DMControlWrapper, TransformedEnv
from torchrl.envs.transforms import Compose, Resize, ToTensorImage, RewardSum
from tensordict.nn import TensorDictModule
from tensordict import TensorDict


class Environment(ABC):
    """
    Abstract base class for environments.
    """

    def __init__(self):
        self.state_space = None
        self.action_space = None
        self.observation_dim = None
        self.observation_space = None
        self.current_state = None
        self.image_states = None

    @abstractmethod
    def step(self, action):
        """
        Take a step in the environment based on the given action.
        
        :param action: The action to take.
        :return: A tuple containing the next state, reward, done flag, 
            and additional info (s, r, done, info).
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the environment to its initial state and return the initial observation.
        
        :return: The initial observation of the environment.
        """
        pass

    @abstractmethod
    def render(self):
        """
        Render the current state of the environment.
        """
        pass

    @abstractmethod
    def get_actions(self):
        """
        Get the available actions in the environment.
        
        :return: A list of available actions.
        """
        pass

    def _get_reward(self, state, action):
        """
        Calculate the reward for a given state and action.
        This is the environment's internal reward function.
        
        r(s, a)
        
        :param state: The current state of the environment.
        :param action: The action taken in the environment.
        :return: The reward for the given state and action.
        """
        pass

    def _get_next_state(self, state, action):
        """
        Get the next state based on the current state and action.
        This is the environment's internal transition function.
        
        p(s' | s, a)
        
        :param state: The current state of the environment.
        :param action: The action taken in the environment.
        :return: The next state after taking the action.
        """
        pass


class DMCEnv(Environment):
    """
    Wrapper for a DeepMind Control Suite environment.
    """
    def __init__(self, domain_name, task_name, from_pixels=False, pixels_only=True, render_kwargs=None):
        super().__init__()
        
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        
        self.wrapped_env = DMControlWrapper(
            self.env, 
            from_pixels=from_pixels, 
            pixels_only=pixels_only,
            render_kwargs=render_kwargs
        )
        
        self.observation_space = self.wrapped_env.observation_spec
        self.action_space = self.wrapped_env.action_spec
        # Check if 'pixels' key exists before getting its shape
        if "pixels" in self.observation_space.keys():
            self.observation_dim = self.observation_space["pixels"].shape
        else:
            self.observation_dim = None # Or handle it differently if needed
        
        self.current_state = self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.
        
        :return: The initial observation tensor.
        """
        reset_td = self.wrapped_env.reset()
        self.current_state = reset_td.get("pixels")
        return self.current_state

    def step(self, action):
        """
        Take a step in the environment.
        
        :param action: The action to take.
        :return: A tuple containing the next state, reward, done flag, and info.
        """
        # Create a TensorDict to pass to the wrapped environment's step method
        action_td = TensorDict({"action": action}, batch_size=[])
        
        # Step the environment
        step_td = self.wrapped_env.step(action_td)
        
        # Extract the necessary values
        next_state = step_td.get(("next", "pixels"))
        reward = step_td.get(("next", "reward"))
        done = step_td.get(("next", "done"))
        
        # Update current state
        self.current_state = next_state
        
        # Return the simplified tuple
        return next_state, reward, done, {} # Empty info dict for consistency
    
    def render(self):
        """
        Render the current state of the environment.
        """
        if self.image_states is None:
            print("Cannot render without pixel-based observations enabled.")
            return None
        return self.image_states

    def get_actions(self):
        """
        Get the available actions in the environment.
        
        :return: The action specification object.
        """
        return self.action_space


if __name__ == "__main__":
    env = DMCEnv(domain_name="cartpole", task_name="swingup", from_pixels=True)
    obs = env.reset()
    print("Initial Observation shape:", obs.shape)
    action = env.get_actions().sample()
    next_obs, reward, done, info = env.step(action)
    print("Next Observation shape:", next_obs.shape)
    print("Reward:", reward)
    print("Done:", done)
    print("Info:", info)