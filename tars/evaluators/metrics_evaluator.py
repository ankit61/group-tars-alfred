from tars.base.evaluator import Evaluator
from tars.envs.alfred_env import AlfredEnv
import numpy as np


# TODO: don't want to evaluate the static methods everytime. Only once at start, but then might have to change signature of methods
# TODO: let's have a dictionary of the metrics (for episode and for all episodes)? Maybe instantiate the global one in the parent Evaluator
class MetricsEvaluator(Evaluator):
    def __init__(self, policy):
        super().__init__(policy)
        self.episode_metrics = dict()  # FIXME: see second TODO. This is the metrics for one episode
        self.objects_already_interacted_with = [] # prevent double counting for IAPP

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
        # Navigation Performance (NP) Metric
        if self.episode_metrics['np'] != 1:
            self.episode_metrics['np'] = int(self.np_metric(env, self.get_np_obj_id(env)))

        # Interaction Action Prediction Performance (IAPP) Metric
        attempted_interactions = self.get_attempted_interactions(env) # FIXME: see first TODO
        predicted_action, predicted_mask = "", "" # FIXME pass these from the model
        iapp = self.iapp_metric(env, attempted_interactions, predicted_action, predicted_mask)
        self.episode_metrics["iapp"] += iapp / len(attempted_interactions) # percentage of attempted actions predicted correctly

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

    # Note: np_obj_id is an argument so it is not computed every time
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

        We need to define what the first object should be: the first object mentioned in the task_desc, the first object
        expert interacts with in plan, one of the items in the pddl_params
        Assuming the third option (i.e. object_target in pddl_params)
        '''
        for action in env.low_level_actions:
            if 'objectId' in action['api_action']:
                return action['api_action']['objectId']
        return ""


    @staticmethod
    def get_attempted_interactions(env: AlfredEnv):
        # interact_objects = []
        # interact_objects_action = []
        attempted_interactions = []
        for action in env.low_level_actions:
            if 'objectId' in action['api_action']:  # interactions with objects
                objectId = action['api_action']['objectId']
                attempted_interactions.append(action['api_action'])
                # interact_objects.append(objectId)
                # interact_objects_action.append()
                # interact_objects.append(objectId[:objectId.find('|')])
                # interact_objects_action.append(action['api_action']['action'])

        return attempted_interactions
