import os
from pathlib import Path
from tars.base.config import Config
from vocab import Vocab


class AlfredEnvConfig(Config):
    nav_actions = set([
                        'MoveAhead', 'RotateRight', 'RotateLeft',
                        'LookUp', 'LookDown'
                    ])

    interact_actions = set([
                            'PickupObject', 'PutObject', 'OpenObject',
                            'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff',
                            'SliceObject'
                        ])

    stop_action = '<<Stop>>'

    actions = Vocab(
                    list(nav_actions) +
                    list(interact_actions) +
                    [stop_action]
                )

    reward_config = os.path.join(str(Path(__file__).parents[2]), 'alfred/models/config/rewards.json')

    failure_reward = -10
