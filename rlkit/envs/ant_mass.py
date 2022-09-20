import numpy as np

from rlkit.envs.ant_multitask_base import MultitaskAntEnv
from . import register_env


@register_env('ant-mass')
class AntMassEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=False, randomize_tasks=True, env_type='train', **kwargs):
        self.forward_backward = forward_backward
        self.env_type = env_type
        self._task = task
        self.mass_scale = 0.8
        self._goal = 0.
        self.tasks = self.sample_tasks(n_tasks)
        super(AntMassEnv, self).__init__(task, n_tasks, **kwargs)
        self.original_mass = np.copy(self.model.body_mass)
        #super(AntMassEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, a):
        self.xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]

        reward_ctrl = -0.005 * np.square(a).sum()
        reward_run = (xposafter - self.xposbefore) / self.dt
        reward_contact = 0.0
        reward_survive = 0.05
        reward = reward_run + reward_ctrl + reward_contact + reward_survive
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(
            reward_forward=reward_run,
            reward_ctrl=reward_ctrl,
            reward_contact=reward_contact,
            reward_survive=reward_survive
        )
    def get_all_task_idx(self):
        return range(len(self.tasks))
    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        self.xposbefore = self.get_body_com("torso")[0]
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
    def sample_tasks(self, num_tasks):
        if self.env_type == 'test':
            masses = np.random.uniform(0.8, 1.0, size=(num_tasks,))
            tasks = [{'mass': mass} for mass in masses]
        else:
            masses = np.random.uniform(0.2, 0.8, size=(num_tasks,))
            tasks = [{'mass': mass} for mass in masses]
        return tasks
    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._mass = self._task['mass']
        self.mass_scale = self._mass
        self.reset()
