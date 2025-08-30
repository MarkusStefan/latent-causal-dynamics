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
    def __init__(self, domain_name, task_name, from_pixels=False, pixels_only=True, render_kwargs=None, img_size: int = 64):
        super().__init__()
        
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        
        base = DMControlWrapper(
            self.env, 
            from_pixels=from_pixels, 
            pixels_only=pixels_only,
            render_kwargs=render_kwargs
        )
        if from_pixels:
            transforms = Compose(
                ToTensorImage(),  # HWC[0..255] -> CHW float[0..1]
                Resize(img_size, img_size),
            )
            self.wrapped_env = TransformedEnv(base, transforms)
        else:
            self.wrapped_env = base
        
        self.observation_space = self.wrapped_env.observation_spec
        self.action_space = self.wrapped_env.action_spec
        # Check if 'pixels' key exists before getting its shape
        if from_pixels and "pixels" in self.observation_space.keys():
            self.observation_dim = self.observation_space["pixels"].shape  # [C,H,W]
        else:
            self.observation_dim = None # or state features
        
        self.from_pixels = from_pixels
        self.current_state = self.reset()

    def reset(self):
        """
        Reset the environment to its initial state.
        
        :return: The initial observation tensor.
        """
        reset_td = self.wrapped_env.reset()
        self.current_state = reset_td.get("pixels") if self.from_pixels else reset_td.get("observation")
        return self.current_state


    def step(self, action):
        """
        Take a step in the environment.
        
        :param action: The action to take.
        :return: A tuple containing the next state, reward, done flag, and info.
        """
        action = action.to(dtype=torch.float32)
        action_td = TensorDict({"action": action}, batch_size=[])
        step_td = self.wrapped_env.step(action_td)
        next_state = step_td.get(("next", "pixels")) if self.from_pixels else step_td.get(("next", "observation"))
        reward = step_td.get(("next", "reward"))
        done = step_td.get(("next", "done"))
        self.current_state = next_state
        return next_state, reward, done, {}


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