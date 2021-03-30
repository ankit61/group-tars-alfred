import os
from tars.base.dataset import Dataset, DatasetType
from tars.base.configurable import Configurable
from tars.base.policy import Policy
from tars.envs.alfred_env import AlfredEnv


class Evaluator(Configurable):
    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def evaluate_split(self, split: DatasetType):
        '''
            Run evaluate on entire dataset
        '''
        data = Dataset(split)
        for task_dir, lang_idx in data.tasks():
            json_file = os.path.join(task_dir, 'traj_data.json')
            self.evaluate(json_file, lang_idx)

    def evaluate(self, json_file, lang_idx):
        env = AlfredEnv(json_file, lang_idx, self.policy.get_img_transforms())

        img = env.reset()
        self.policy.reset()
        self.at_start(env, img)

        text_transform = self.policy.get_text_transforms()
        goal_inst = text_transform([env.goal_inst], is_goal=True)
        low_level_insts = text_transform([env.low_level_insts], is_goal=False)

        done = False
        tot_reward = 0
        while not done:
            self.at_step_begin(env)

            # set up
            img = img.unsqueeze(0) # add batch size dimension

            # predict
            action, int_mask = self.policy(img, goal_inst, low_level_insts)
            action, int_mask = self.policy.clean_preds(action, int_mask)
            action, int_mask = action.squeeze(0), int_mask.squeeze(0) # since batch size = 1

            # update env
            img, reward, done, _ = env.step((action, int_mask))
            self.at_step_end(
                env,
                (img, goal_inst, low_level_insts),
                (action, int_mask),
                (img, reward, done)
            )
            tot_reward += reward

        env.close()
        print('Total Reward: ', tot_reward)

    def at_step_begin(self, env):
        '''
            Args:
                env: current environment
        '''
        pass

    def at_step_end(self, env, policy_in, policy_out, nrd):
        '''
            Args:
                env: current environment
                policy_in: tuple containing input given to the policy
                policy_out: tuple containing output of the policy
                nrd: tuple of (next state, reward, done) after taking executing
                    policy_out
        '''
        pass

    def at_start(self, env, start_state):
        '''
            Args:
                env: current environment
        '''
        pass

    def at_end(self, env):
        '''
            Args:
                env: current environment
        '''
        pass
