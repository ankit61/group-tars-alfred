from tars.base.evaluator import Evaluator
from tars.envs.alfred_env import AlfredEnv
import numpy as np


class MetricsEvaluator(Evaluator):
    def __init__(self, policy):
        super().__init__(policy)
        self.json_file_metrics = dict()
        self.episode_metrics = dict()
        self.objects_already_interacted_with = [] # prevent double counting for IAPP
        self.object_to_navigate_to = "" # used by the NP metric
        self.expert_interact_objects, self.expert_interact_objects_action = [], [] # used by IAPP metric

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
        predicted_action, predicted_mask = policy_out

        # Navigation Performance (NP) Metric
        np = self.navigation_performance_metric(env, self.object_to_navigate_to)
        self.episode_metrics[
            "np"] = 1 if np else 0  # for the whole episode (i.e. navigated to the first object it has to interact with

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
        # reset
        self.episode_metrics = dict()
        self.objects_already_interacted_with = []
        self.object_to_navigate_to = ""
        self.expert_interact_objects, self.expert_interact_objects_action = [], []  # used by IAPP metric

        self.object_to_navigate_to = MetricsEvaluator.get_object_to_navigate_to(env)
        self.expert_interact_objects, self.expert_interact_objects_action = MetricsEvaluator.find_objects_to_interact_with(
            env)


    def at_end(self, env: AlfredEnv):
        '''
            Args:
                env: current environment
        '''
        self.json_file_metrics[env.json_file] = self.episode_metrics


    # Note: object_to_navigate_to is an argument so it is not computed every time
    def navigation_performance_metric(self, env: AlfredEnv, object_to_navigate_to):
        '''
        Assumptions:

        How do positions/coordinates work? Assuming positions/coordinates are absolute for whole environment instead of
        a particular scene/image
        '''
        for object in env.env.last_event.metadata['objects']:
            if object_to_navigate_to in object['name']:
                if object['visible']:  # this means that the agent is near the object & object is in its field of view
                    return True
        return False


    # Note: expert_interact_objects, expert_interact_objects_action are arguments so they are not computed every time
    def iapp_metric(self, env: AlfredEnv, expert_interact_objects, expert_interact_objects_action, predicted_action,
                    predicted_mask):

        agent_inter_object = env.env.get_target_instance_id(predicted_mask)

        for expert_inter_object, expert_inter_object_action in zip(expert_interact_objects,
                                                                   expert_interact_objects_action):
            if agent_inter_object in expert_inter_object and predicted_action == expert_inter_object_action \
                    and (agent_inter_object, predicted_action) not in self.objects_already_interacted_with:
                self.objects_already_interacted_with.append((agent_inter_object, predicted_action)) # prevent double counting if agent stuck in loop, etc.
                return True
        return False


    @staticmethod
    def get_object_to_navigate_to(env: AlfredEnv):
        '''
        Assumptions:

        We need to define what the first object should be: the first object mentioned in the task_desc, the first object
        expert interacts with in plan, one of the items in the pddl_params
        Assuming the third option (i.e. object_target in pddl_params)
        '''
        object_to_navigate_to = ""
        for action in env.high_level_actions:
            if 'objectId' in action['planner_action']:
                object_to_navigate_to = action['planner_action']['coordinateObjectId'][0]
                break
        return object_to_navigate_to


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
