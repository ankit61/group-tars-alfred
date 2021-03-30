import json
import os
from datetime import datetime
from tars.base.configurable import Configurable
from tars.base.policy import Policy
from tars.envs.alfred_env import AlfredEnv


class Evaluator(Configurable):
    def __init__(self, args, policy: Policy):
        super().__init__()
        self.args = args
        self.policy = policy
        self.successes = []
        self.failures = []
        self.results = {}

    def evaluate(self, thor_env, json_file, lang_idx):
        env = AlfredEnv(thor_env, json_file, lang_idx, self.policy.get_img_transforms())

        with open(json_file) as f:
            traj_data = json.load(f)

        img = env.reset()
        self.policy.reset()
        self.at_start(env, img)

        text_transform = self.policy.get_text_transforms()
        goal_inst = text_transform([env.goal_inst], is_goal=True)
        low_level_insts = text_transform([env.low_level_insts], is_goal=False)

        done = False
        t = 0
        fails = 0
        tot_reward = 0
        while not done:
            # break if max_steps reached
            if t >= self.args.max_steps:
                break

            self.at_step_begin(env)

            # set up
            img = img.unsqueeze(0) # add batch size dimension

            # predict
            action, int_mask = self.policy(img, goal_inst, low_level_insts)
            action, int_mask = self.policy.clean_preds(action, int_mask)
            action, int_mask = action.squeeze(0), int_mask.squeeze(0) # since batch size = 1

            # update env
            img, reward, done, success = env.step((action, int_mask))
            if done:
                break
            self.at_step_end(
                env,
                (img, goal_inst, low_level_insts),
                (action, int_mask),
                (img, reward, done)
            )
            if not success:
                fails += 1
                if fails >= self.args.max_fails:
                    print("Interact API failed %d times" % fails)
                    break

            tot_reward += reward
            t += 1

        self.at_end(env)
        print('Total Reward: ', tot_reward)


        # calculate task-level metrics

        # check if goal was satisfied
        goal_satisfied = env.env.get_goal_satisfied()
        if goal_satisfied:
            print("Goal Reached")
            success = True

        # goal_conditions
        pcs = env.env.get_goal_conditions_met()
        goal_condition_success_rate = pcs[0] / float(pcs[1])

        # SPL
        path_len_weight = len(traj_data['plan']['low_actions'])
        s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(t))
        pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(t))

        # path length weighted SPL
        plw_s_spl = s_spl * path_len_weight
        plw_pc_spl = pc_spl * path_len_weight

        # log success/fails
        log_entry = {'trial': traj_data['task_id'],
                     'type': traj_data['task_type'],
                     'repeat_idx': int(lang_idx),
                     'goal_instr': env.goal_inst,
                     'completed_goal_conditions': int(pcs[0]),
                     'total_goal_conditions': int(pcs[1]),
                     'goal_condition_success': float(goal_condition_success_rate),
                     'success_spl': float(s_spl),
                     'path_len_weighted_success_spl': float(plw_s_spl),
                     'goal_condition_spl': float(pc_spl),
                     'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                     'path_len_weight': int(path_len_weight),
                     'reward': float(reward)}
        if success:
            self.successes.append(log_entry)
        else:
            self.failures.append(log_entry)

        # overall results
        self.results['all'] = self.get_metrics(self.successes, self.failures)

        print("-------------")
        print("SR: %d/%d = %.3f" % (self.results['all']['success']['num_successes'],
                                    self.results['all']['success']['num_evals'],
                                    self.results['all']['success']['success_rate']))
        print("GC: %d/%d = %.3f" % (self.results['all']['goal_condition_success']['completed_goal_conditions'],
                                    self.results['all']['goal_condition_success']['total_goal_conditions'],
                                    self.results['all']['goal_condition_success']['goal_condition_success_rate']))
        print("PLW SR: %.3f" % (self.results['all']['path_length_weighted_success_rate']))
        print("PLW GC: %.3f" % (self.results['all']['path_length_weighted_goal_condition_success_rate']))
        print("-------------")

        # task type specific results
        task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                      'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                      'pick_and_place_with_movable_recep']
        for task_type in task_types:
            task_successes = [s for s in (list(self.successes)) if s['type'] == task_type]
            task_failures = [f for f in (list(self.failures)) if f['type'] == task_type]
            if len(task_successes) > 0 or len(task_failures) > 0:
                self.results[task_type] = self.get_metrics(task_successes, task_failures)
            else:
                self.results[task_type] = {}
        

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


    def get_metrics(self, successes, failures):
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
                   'results': dict(self.results)}

        save_path = os.path.dirname(self.args.model_path)
        save_path = os.path.join(save_path, 'our_code_task_results_' + self.args.eval_split + '_' + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + '.json')
        with open(save_path, 'w') as r:
            json.dump(results, r, indent=4, sort_keys=True)
