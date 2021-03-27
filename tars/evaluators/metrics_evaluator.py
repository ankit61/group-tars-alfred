from tars.base.evaluator import Evaluator


class MetricsEvaluator:
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
