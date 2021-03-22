import os
from pathlib import Path
from tars.base.config import Config
from vocab import Vocab


class AlfredEnvConfig(Config):
    def __init__(self):
        super(AlfredEnvConfig, self).__init__()

        self.nav_actions = set([
                            'MoveAhead', 'RotateRight', 'RotateLeft',
                            'LookUp', 'LookDown'
                        ])

        self.interact_actions = set([
                                'PickupObject', 'PutObject', 'OpenObject',
                                'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff',
                                'SliceObject'
                            ])

        self.stop_action = '<<Stop>>'

        self.actions = Vocab(
                            list(self.nav_actions) +
                            list(self.interact_actions) +
                            [self.stop_action]
                        )

        self.reward_config = os.path.join(str(Path(__file__).parents[2]), 'alfred/models/config/rewards.json')

        self.failure_reward = -10
