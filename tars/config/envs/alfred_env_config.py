import os
from pathlib import Path
from tars.base.config import Config
from vocab import Vocab


class AlfredEnvConfig(Config):
    nav_actions = {'MoveAhead', 'RotateRight', 'RotateLeft', 'LookUp', 'LookDown'}

    interact_actions = {'PickupObject', 'PutObject', 'OpenObject', 'CloseObject', 'ToggleObjectOn', 'ToggleObjectOff',
                        'SliceObject'}

    actions_change_object_pos = {'PickupObject', 'PutObject'}

    stop_action = '<<stop>>'

    actions = Vocab(
                    list(nav_actions) +
                    list(interact_actions) +
                    [stop_action]
                )

    reward_config = os.path.join(str(Path(__file__).parents[2]), 'alfred/models/config/rewards.json')

    failure_reward = -10

    max_failures = 10
    max_steps = 1000

    debug = False
