import numpy as np

from . import register_env
from .half_cheetah import HalfCheetahEnv


@register_env('cheetah-mass')
class HalfCheetahMassEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for
        model-based control", 2012
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self, task={}, n_tasks=2, randomize_tasks=True, env_type='train'):
        self.env_type = env_type
        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        #self.mass_scale_set =[0.75, 0.85, 1.0, 1.15, 1.25]
        #self.damping_scale_set=[0.75, 0.85, 1.0, 1.15, 1.25]
        super(HalfCheetahMassEnv, self).__init__()
        self.original_mass = np.copy(self.model.body_mass)
        self.original_damping = np.copy(self.model.dof_damping)
        self._goal = 0.
        self.mass_scale = 1.0
        #self._goal_vel = self.tasks[0].get('velocity', 0.0)
        #self._goal = self._goal_vel
        #super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run

        observation = self._get_obs()
        #reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=reward_run,
            reward_ctrl=reward_ctrl, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        if self.env_type == 'test':
            masses = np.random.uniform(1.5, 1.8, size=(num_tasks,))
            tasks = [{'mass': mass} for mass in masses]
        else:
            masses = np.random.uniform(0.2, 1.5, size=(num_tasks,))
            tasks = [{'mass': mass} for mass in masses]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)

        #random_index = self.np_random.randint(len(self.mass_scale_set))
        #self.mass_scale = self.mass_scale_set[random_index]

        #random_index = self.np_random.randint(len(self.damping_scale_set))
        #self.damping_scale = self.damping_scale_set[random_index]

        self.change_env()
        return self._get_obs()
    def change_env(self):
        mass = np.copy(self.original_mass)
        #damping = np.copy(self.original_damping)
        mass *= self.mass_scale
        #damping *= self.damping_scale

        self.model.body_mass[:] = mass
        #self.model.dof_damping[:] = damping
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._mass = self._task['mass']
        self.mass_scale = self._mass
        self.reset()
