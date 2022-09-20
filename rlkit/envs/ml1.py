
import math
import numpy as np
from gym import spaces
from gym.utils import seeding
from gym import utils
from . import register_env
import time
from metaworld.benchmarks import ML1 as ML1

@register_env('ML1_test')
class ML1_test(ML1):
    def __init__(self, task_name='reach-v1',env_type='train', n_tasks=2, randomize_tasks=True):
        #self._task = task
        self._serializable_initialized = True
        super(ML1_test, self).__init__(task_name=task_name, env_type=env_type, n_goals=50)
        #ML1.__init__(self, task_name=task_name, env_type=env_type, n_goals=50)
    #def initsample(self, n_tasks,randomize_tasks=True):
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self.set_task(self.tasks[idx])
        self._goal = self.active_env.goal
        # assume parameterization of task by single vector
    def get_all_task_idx(self):
        return range(len(self.tasks))




