from tars.base.evaluator import Evaluator
from tars.envs.alfred_env import AlfredEnv
import numpy as np


class MetricsEvaluator(Evaluator):
    def __init__(self, policy):
        super().__init__(policy)
        self.json_file_metrics = dict()
        self.episode_metrics = dict()
        self.np_obj_id = None # used by the NP metric
        self.objects_already_interacted_with = [] # prevent double counting for IAPP
        self.expert_interact_objects, self.expert_interact_objects_action = [], [] # used by IAPP metric

    def at_step_begin(self, env):
        '''
            Args:
                env: current environment
        '''


    def at_step_end(self, env, policy_in, policy_out, nrd):
        '''
            Args:
                env: current environment
                policy_in: tuple containing input given to the policy
                policy_out: tuple containing output of the policy
                nrd: tuple of (next state, reward, done) after taking executing
                    policy_out
        '''

        predicted_action, predicted_mask = policy_out

        # Update Navigation Performance (NP) metric
        if self.episode_metrics['np'] != 1:
            self.episode_metrics['np'] = int(self.np_metric(env, self.np_obj_id))

        # Interaction Action Prediction Performance (IAPP) Metric
        iapp = self.iapp_metric(env, self.expert_interact_objects, self.expert_interact_objects_action, predicted_action,
                                predicted_mask)
        self.episode_metrics["iapp"] += iapp / len(
            self.expert_interact_objects)  # percentage of correct actions predicted correctly


    def at_start(self, env, start_state):
        '''
            Args:
                env: current environment
        '''
        # reset episode metrics, prefetch per-episode values for metrics
        self.episode_metrics = dict()
        self.np_obj_id = self.get_np_obj_id(env)

        self.objects_already_interacted_with = []
        self.expert_interact_objects, self.expert_interact_objects_action = MetricsEvaluator.find_objects_to_interact_with(
            env)


    def at_end(self, env: AlfredEnv):
        '''
            Args:
                env: current environment
        '''
        # save episode metrics
        self.json_file_metrics[env.json_file] = self.episode_metrics


    def np_metric(self, env: AlfredEnv, np_obj_id):
        '''
        Assumptions:

        How do positions/coordinates work? Assuming positions/coordinates are absolute for whole environment instead of
        a particular scene/image
        '''
        for obj in env.env.last_event.metadata['objects']:
            if obj['objectId'] == np_obj_id and obj['visible']:
                return True
        return False


    # Note: expert_interact_objects, expert_interact_objects_action are arguments so they are not computed every time
    def iapp_metric(self, env: AlfredEnv, expert_interact_objects, expert_interact_objects_action, predicted_action, predicted_mask):

        agent_inter_object = env.env.get_target_instance_id(predicted_mask)

        for expert_inter_object, expert_inter_object_action in zip(expert_interact_objects, expert_interact_objects_action):
            if agent_inter_object in expert_inter_object and predicted_action == expert_inter_object_action \
                    and (agent_inter_object, predicted_action) not in self.objects_already_interacted_with:
                self.objects_already_interacted_with.append((agent_inter_object, predicted_action)) # prevent double counting if agent stuck in loop, etc.
                return True
        return False


    @staticmethod
    def get_np_obj_id(env: AlfredEnv):
        '''
        Assumptions:

        'First object' here is assumed to be the first object the expert 
        interacts with in the low-level actions
        '''
        for action in env.low_level_actions:
            if 'objectId' in action['api_action']:
                return action['api_action']['objectId']
        return ""


    @staticmethod
    def find_objects_to_interact_with(env: AlfredEnv):
        interact_objects = []
        interact_objects_action = []
        for action in env.low_level_actions:
            if "objectId" in action['api_action']:  # interactions with objects
                objectId = action['api_action']['objectId']
                interact_objects.append(objectId[:objectId.find('|')])
                interact_objects_action.append(action['api_action']['action'])

        return interact_objects, interact_objects_action
