from tars.base.evaluator import Evaluator
from tars.envs.alfred_env import AlfredEnv
import numpy as np

class MetricsEvaluator(Evaluator):
    def __init__(self, policy):
        super().__init__(policy)

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

    def navigation_performance_metric(self, env : AlfredEnv, object_to_navigate_to, object_to_navigate_to_coordinates):
        '''
        Assumptions:

        How do positions/coordinates work? Assuming positions/coordinates are absolute for whole environment instead of
        a particular scene/image
        '''
        agent_pos = env.env.last_event.metadata['agent']['position']
        agent_coordinates = np.asarray([agent_pos['x'], agent_pos['y'], agent_pos['z']])

        for object_coord in object_to_navigate_to_coordinates:
            if self.is_agent_near_target_object(agent_coordinates, object_coord):
                return True
        return False


    @staticmethod
    def get_object_to_navigate_to(env : AlfredEnv):
        '''
        Assumptions:

        We need to define what the first object should be: the first object mentioned in the task_desc, the first object
        expert interacts with in plan, one of the items in the pddl_params
        Assuming the third option (i.e. object_target in pddl_params)
        '''

        object_to_navigate_to = env.pddl_params['object_target']  # string name
        object_to_navigate_to_coordinates = []

        for object_pose in env.object_poses:
            if object_to_navigate_to in object_pose['objectName']:
                pos = object_to_navigate_to['position']
                coordinates = (pos['x'], pos['y'], pos['z'])
                object_to_navigate_to_coordinates.append(np.asarray(coordinates))

        return object_to_navigate_to, object_to_navigate_to_coordinates


    @staticmethod
    def is_agent_near_target_object(agent_position, object_position, threshold=1):
        percent_difference = np.absolute(agent_position - object_position)/object_position > threshold
        if percent_difference.all():
            return False
        return True
