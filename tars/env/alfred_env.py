import json
import argparse
import numpy as np
from gym import spaces
from tars.alfred.env.thor_env import ThorEnv
from tars.base.env import Env


class AlfredEnv(Env):
    def __init__(self, json_file, lang_idx, reward_type='dense', viz=False):
        self.reward_type = reward_type
        self.viz = viz
        self.lang_idx = lang_idx

        self.env = ThorEnv()  # FIXME: use self.viz

        with open(json_file) as f:
            self.traj_data = json.load(f)

        # scene setup
        self.scene_name = 'FloorPlan%d' % self.traj_data['scene']['scene_num']
        self.object_poses = self.traj_data['scene']['object_poses']
        self.object_toggles = self.traj_data['scene']['object_toggles']
        self.dirty_and_empty = self.traj_data['scene']['dirty_and_empty']

        # reset
        args = argparse.Namespace()
        args.reward_config = '/home/ankit/Code/group-tars-alfred/tars/alfred/models/config/rewards.json'
        self.env.set_task(self.traj_data, args, reward_type=self.reward_type)
        self.reset()

        # env setup
        obs_space = spaces.Box(low=0, high=255, shape=self.env.img_shape, dtype=np.int32) # image
        ac_space = spaces.Tuple([
                        spaces.Discrete(len(self.conf.actions)),
                        spaces.Box(low=0, high=1, shape=self.env.img_shape, dtype=np.int32)
                    ])  # action, segmentation mask

        super(AlfredEnv, self).__init__(obs_space, ac_space)

    def get_obs(self, state):
        return state.frame

    def reset(self):
        self.env.reset(self.scene_name)
        self.env.restore_scene(self.object_poses, self.object_toggles, self.dirty_and_empty)

        state = self.env.step(dict(self.traj_data['scene']['init_action']))
        return self.get_obs(state)

    def step(self, action):
        action_idx, interact_mask = action
        action = self.conf.actions.index2word(action_idx),
        interact_mask = interact_mask if action in self.conf.interact_actions else None

        done = (action == self.conf.stop_action)
        next_obs = self.get_obs(self.env.last_event)
        reward = 0 if done else self.conf.failure_reward

        if not done:
            success, event, _, err, _ = self.env.va_interact(action, interact_mask, debug=True)

            if success:
                next_obs = self.get_obs(event)
                reward, d = self.env.get_transition_reward()
                done = done or d

        return next_obs, reward, done, None

    def get_expert_traj(self):
        if 'plan' not in self.traj_data:
            raise RuntimeError('expert plan not provided')

        for action in self.traj_data['plan']['low_actions']:
            yield action['api_action']

    def run_expert_traj(self, step_by_step=False):
        for action in self.get_expert_traj():
            self.env.step(action)
            if step_by_step:
                yield action

    def cache(self):
        # saves images, etc for supervised learning setup
        raise NotImplementedError

    @property
    def privelged_state(self):
        return self.env.last_event

    @property
    def lang_insts(self):
        lang_insts = self.traj_data['turk_annotations']['anns'][self.lang_idx]
        return lang_insts['task_desc'], lang_insts['high_descs']
