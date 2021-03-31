import json
import argparse
import numpy as np
from PIL import Image
from gym import spaces
from torchvision.transforms import Compose
from tars.alfred.env.thor_env import ThorEnv
from tars.base.env import Env


class AlfredEnv(Env):
    def __init__(self, transforms=Compose([]), reward_type='dense', viz=True):
        self.transforms = transforms
        self.reward_type = reward_type
        self.viz = viz

        self.thor_env = ThorEnv()  # FIXME: use self.viz

        # env setup
        obs_space = spaces.Box(low=0, high=255, shape=self.thor_env.img_shape, dtype=np.int32) # image
        ac_space = spaces.Tuple([
                        spaces.Discrete(len(self.conf.actions)),
                        spaces.Box(low=0, high=1, shape=self.thor_env.img_shape[:2], dtype=np.int32)
                    ])  # action, segmentation mask

        super(AlfredEnv, self).__init__(obs_space, ac_space)


    def setup_task(self, json_file, lang_idx):
        self.json_file = json_file
        self.lang_idx = lang_idx
        
        with open(json_file) as f:
            self.traj_data = json.load(f)

        # scene setup
        self.scene_name = 'FloorPlan%d' % self.traj_data['scene']['scene_num']
        self.object_poses = self.traj_data['scene']['object_poses']
        self.object_toggles = self.traj_data['scene']['object_toggles']
        self.dirty_and_empty = self.traj_data['scene']['dirty_and_empty']
        self.pddl_params = self.traj_data['pddl_params']
        self.high_level_actions = self.traj_data['plan']['high_pddl']
        self.low_level_actions = self.traj_data['plan']['low_actions']

        # print goal instr
        print("Task: %s" % (self.traj_data['turk_annotations']['anns'][lang_idx]['task_desc']))

        return self.reset()


    def get_obs(self, state):
        return self.transforms(Image.fromarray(state.frame))


    def reset(self):
        self.num_failures = 0
        self.episode_len = 0

        args = argparse.Namespace()
        args.reward_config = self.conf.reward_config

        self.thor_env.reset(self.scene_name)
        self.thor_env.restore_scene(self.object_poses, self.object_toggles, self.dirty_and_empty)

        # initialize to start position
        state = self.thor_env.step(dict(self.traj_data['scene']['init_action']))

        # setup task for reward
        self.thor_env.set_task(self.traj_data, args, reward_type=self.reward_type)

        return self.get_obs(state)
    

    def step(self, action):
        action_idx, interact_mask = action

        action_name = self.conf.actions.index2word(action_idx)
        interact_mask = interact_mask if self.is_interact_action(action_name) else None

        done = (action_name == self.conf.stop_action)
        next_obs = self.get_obs(self.thor_env.last_event)
        reward = 0 if done else self.conf.failure_reward

        if not done:
            success, event, _, err, _ = self.thor_env.va_interact(action_name, interact_mask, smooth_nav=False)

            if self.conf.debug:
                debug_str = "{}: {}".format("SUCCESS" if success else "FAILURE", action_name)
                if err:
                    debug_str += " (err: {})".format(err)
                print(debug_str)

            if success:
                next_obs = self.get_obs(event)
                reward, d = self.thor_env.get_transition_reward()
                done = done or d
            else:
                self.num_failures += 1
                if self.num_failures >= self.conf.max_failures:
                    print("Interact API failed %d times" % self.num_failures + "; latest error '%s'" % err)
                    done = True

        self.episode_len += 1
        if self.episode_len >= self.conf.max_steps:
            done = True

        return next_obs, reward, done, None


    def close(self):
        self.thor_env.stop()


    def get_expert_traj(self):
        if 'plan' not in self.traj_data:
            raise RuntimeError('expert plan not provided')

        for action in self.traj_data['plan']['low_actions']:
            yield action['api_action']


    def run_expert_traj(self):
        for action in self.get_expert_traj():
            self.thor_env.step(action, smooth_nav=True)


    def cache(self):
        # saves images, etc for supervised learning setup
        raise NotImplementedError

    def is_interact_action(self, action_name):
        return action_name in self.conf.interact_actions

    def is_action_changing_object_pos(self, action_name):
        return action_name in self.conf.actions_change_object_pos

    @property
    def full_state(self):
        return self.thor_env.last_event


    @property
    def goal_inst(self):
        lang_insts = self.traj_data['turk_annotations']['anns'][self.lang_idx]
        return lang_insts['task_desc']


    @property
    def low_level_insts(self):
        lang_insts = self.traj_data['turk_annotations']['anns'][self.lang_idx]
        return lang_insts['high_descs']
