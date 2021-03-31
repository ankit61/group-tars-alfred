import numpy as np
import os
import json
import pandas as pd
import pickle
from collections import defaultdict
from tars.base.evaluator import Evaluator
from tars.envs.alfred_env import AlfredEnv
from datetime import datetime


class MetricsEvaluator(Evaluator):
    def __init__(self, policy):
        super().__init__(policy)

        self.successes = []
        self.failures = []
        self.task_level_results = {}

        self.results_for_df = defaultdict(lambda: [])
        self.episode_metrics = {}
        self.interact_obj_ids = None
        self.objects_already_interacted_with = [] # prevent double counting for IAPP
        self.expert_interact_objects, self.expert_interact_objects_action = [], [] # used by IAPP metric


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

        predicted_action, predicted_mask = policy_out

        # update FONS metric
        self.update_fons()

        # Interaction Action Prediction Performance (IAPP) Metric
        iapp = self.iapp_metric(self.expert_interact_objects, self.expert_interact_objects_action, predicted_action,
                                predicted_mask)
        # FIXME: ran into divide by 0 error, wrapped it in a max(x, 1) for now
        self.episode_metrics["iapp"] += iapp / max(len(
            self.expert_interact_objects), 1)  # percentage of correct actions predicted correctly
                                                


    def at_episode_start(self, start_state):
        '''
            Args:
                start_state: starting state of agent
        '''
        # reset episode metrics, prefetch per-episode values for metrics
        self.episode_metrics['fons'] = False
        self.episode_metrics['iapp'] = 0

        self.fons_obj_id = self.get_fons_obj_id()

        self.objects_already_interacted_with = []
        self.expert_interact_objects, self.expert_interact_objects_action = self.find_objects_to_interact_with()
        

    def at_episode_end(self, total_reward):
        '''
            Args:
                tot_reward: cumulative episode reward
        '''
        self.update_metrics(total_reward)
        

    def at_end(self):
        super().at_end()
        # save task-level metrics
        self.save_results()

    
    def update_metrics(self, total_reward):
        # check if goal was satisfied
        goal_satisfied = self.env.thor_env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True
        else:
            success = False

        # goal_conditions
        pcs = self.env.thor_env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(self.env.traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(self.env.episode_len))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(self.env.episode_len))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        log_entry = {'trial': self.env.traj_data['task_id'],
                     'type': self.env.traj_data['task_type'],
                     'repeat_idx': int(self.env.lang_idx),
                     'goal_instr': self.env.goal_inst,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(total_reward),
                     'fons': int(self.episode_metrics['fons']),
                     'iapp': float(self.episode_metrics['iapp'])}

        for (k, v) in log_entry.items():
            self.results_for_df[k].append(v)

        if success:
            self.successes.append(log_entry)
        else:
            self.failures.append(log_entry)

        # overall results
        self.task_level_results['all'] = self.get_task_level_metrics(self.successes, self.failures)

        # intrinsic metrics
        print("-------------")
        fons_succs = sum(self.results_for_df['fons'])
        fons_eps = len(self.results_for_df['fons'])
        print("FONS Rate: %d/%d = %.3f" % (fons_succs, fons_eps, fons_succs / fons_eps))
        print("Mean IAPP: %.3f" % np.mean(self.results_for_df['iapp']))

        # task-level metrics
        print("-------------")
        print("SR: %d/%d = %.3f" % (self.task_level_results['all']['success']['num_successes'],
                                    self.task_level_results['all']['success']['num_evals'],
                                    self.task_level_results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (self.task_level_results['all']['goal_condition_success']['completed_goal_conditions'],
                                    self.task_level_results['all']['goal_condition_success']['total_goal_conditions'],
                                    self.task_level_results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (self.task_level_results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (self.task_level_results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results

        for task_type in self.conf.task_types:
            task_successes = [s for s in (list(self.successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(self.failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                self.task_level_results[task_type] = self.get_task_level_metrics(task_successes, task_failures)
            else:
                self.task_level_results[task_type] = {}


    def get_task_level_metrics(self, successes, failures):
        '''
        compute overall succcess and goal_condition success rates along with path-weighted metrics
        '''
        # stats
        num_successes, num_failures = len(successes), len(failures)
        num_evals = len(successes) + len(failures)
        total_path_len_weight = sum([entry['path_len_weight'] for entry in successes]) + \
                                sum([entry['path_len_weight'] for entry in failures])
        completed_goal_conditions = sum([entry['completed_goal_conditions'] for entry in successes]) + \
                                   sum([entry['completed_goal_conditions'] for entry in failures])
        total_goal_conditions = sum([entry['total_goal_conditions'] for entry in successes]) + \
                               sum([entry['total_goal_conditions'] for entry in failures])

        # metrics
        sr = float(num_successes) / num_evals
        pc = completed_goal_conditions / float(total_goal_conditions)
        plw_sr = (float(sum([entry['path_len_weighted_success_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_success_spl'] for entry in failures])) /
                  total_path_len_weight)
        plw_pc = (float(sum([entry['path_len_weighted_goal_condition_spl'] for entry in successes]) +
                        sum([entry['path_len_weighted_goal_condition_spl'] for entry in failures])) /
                  total_path_len_weight)

        # result table
        res = dict()
        res['success'] = {'num_successes': num_successes,
                          'num_evals': num_evals,
                          'success_rate': sr}
        res['goal_condition_success'] = {'completed_goal_conditions': completed_goal_conditions,
                                        'total_goal_conditions': total_goal_conditions,
                                        'goal_condition_success_rate': pc}
        res['path_length_weighted_success_rate'] = plw_sr
        res['path_length_weighted_goal_condition_success_rate'] = plw_pc

        return res


    def save_results(self):
        results = {'successes': list(self.successes),
                   'failures': list(self.failures),
                   'results': dict(self.task_level_results)}

        save_path_root = os.path.dirname(self.policy.conf.saved_model_path)
        save_path = os.path.join(save_path_root, 'our_code_task_results_' + self.split_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)

        results_df = pd.DataFrame(self.results_for_df)
        df_save_path = os.path.join(save_path_root, 'results_df_' + self.split_name + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.pkl')
        results_df.to_pickle(df_save_path)


    def update_fons(self):
        '''
        Assumptions:

        How do positions/coordinates work? Assuming positions/coordinates are absolute for whole environment instead of
        a particular scene/image
        '''
        fons = False
        for obj in self.env.thor_env.last_event.metadata['objects']:
            if obj['objectId'] == self.fons_obj_id and obj['visible']:
                fons = True
        self.episode_metrics['fons'] = self.episode_metrics['fons'] or fons
       


    # Note: expert_interact_objects, expert_interact_objects_action are arguments so they are not computed every time
    def iapp_metric(self, expert_interact_objects, expert_interact_objects_action, predicted_action, predicted_mask):

        agent_inter_object = self.env.full_state.metadata['actionReturn']

        ''' 
        FIXME: getting error: "if agent_inter_object in expert_inter_object and predicted_action == expert_inter_object_action \
TypeError: 'in <string>' requires string as left operand, not tuple"
        '''
        # for expert_inter_object, expert_inter_object_action in zip(expert_interact_objects, expert_interact_objects_action):
        #     if agent_inter_object in expert_inter_object and predicted_action == expert_inter_object_action \
        #             and (agent_inter_object, predicted_action) not in self.objects_already_interacted_with:
        #         self.objects_already_interacted_with.append((agent_inter_object, predicted_action)) # prevent double counting if agent stuck in loop, etc.
        #         return True
        return False


    def get_fons_obj_id(self):
        '''
        Assumptions:

        'First object' here is assumed to be the first object the expert 
        interacts with in the low-level actions
        '''
        for action in self.env.low_level_actions:
            if 'objectId' in action['api_action']:
                return action['api_action']['objectId']
        return None


    def find_objects_to_interact_with(self):
        interact_objects = []
        interact_objects_action = []
        for action in self.env.low_level_actions:
            if "objectId" in action['api_action']:  # interactions with objects
                objectId = action['api_action']['objectId']
                interact_objects.append(objectId[:objectId.find('|')])
                interact_objects_action.append(action['api_action']['action'])

        return interact_objects, interact_objects_action
