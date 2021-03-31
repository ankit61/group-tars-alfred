import os
from tars.base.dataset import Dataset, DatasetType
from tars.config.base.dataset_config import DatasetConfig
from tars.base.configurable import Configurable
from tars.base.policy import Policy
from tars.envs.alfred_env import AlfredEnv


class Evaluator(Configurable):
    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy
        self.env = AlfredEnv(transforms=policy.get_img_transforms())


    def evaluate_split(self, split: DatasetType):
        '''
            Run evaluate on entire dataset
        '''
        self.split_name = split.value
        self.at_start()
        data = Dataset(split)
        print("Dataset start: {}\nDataset end: {}".format(data.conf.start_idx, data.conf.end_idx))
        for i, task in enumerate(data.tasks()):
            print("\n============ TRAJ {} OF {} ============".format(i + 1 + data.conf.start_idx, (data.conf.end_idx if data.conf.end_idx else len(data))))
            print("Evaluating {} ({})".format(task['task'], task['repeat_idx']))
            json_file = os.path.join(data.data_dir, task['task'], DatasetConfig().traj_file)
            self.evaluate(json_file, task['repeat_idx'])
        self.at_end()


    def evaluate(self, json_file, lang_idx):
        img = self.env.setup_task(json_file, lang_idx)

        self.policy.reset()
        self.at_episode_start(img)

        text_transform = self.policy.get_text_transforms()
        goal_inst = text_transform([self.env.goal_inst], is_goal=True)
        low_level_insts = text_transform([self.env.low_level_insts], is_goal=False)

        done = False
        tot_reward = 0
        while not done:
            self.at_step_start()

            # set up
            img = img.unsqueeze(0) # add batch size dimension

            # predict
            action, int_mask = self.policy(img, goal_inst, low_level_insts)
            action, int_mask = self.policy.clean_preds(action, int_mask)
            action, int_mask = action.squeeze(0), int_mask.squeeze(0) # since batch size = 1

            # update env
            img, reward, done, _ = self.env.step((action, int_mask))
            self.at_step_end(
                (img, goal_inst, low_level_insts),
                (action, int_mask),
                (img, reward, done)
            )
            tot_reward += reward

        self.at_episode_end(tot_reward)
        print('Total Reward: ', tot_reward)


    def at_step_start(self):
        '''
            Args:
        '''
        pass


    def at_step_end(self, policy_in, policy_out, nrd):
        '''
            Args:
                policy_in: tuple containing input given to the policy
                policy_out: tuple containing output of the policy
                nrd: tuple of (next state, reward, done) after taking executing
                    policy_out
        '''
        pass


    def at_episode_start(self, start_state):
        '''
            Args:
                start_state: starting state of agent
        '''
        pass


    def at_episode_end(self, tot_reward):
        '''
            Args:
                tot_reward: cumulative episode reward
        '''
        pass


    def at_start(self):
        pass

    
    def at_end(self):
        self.env.close()