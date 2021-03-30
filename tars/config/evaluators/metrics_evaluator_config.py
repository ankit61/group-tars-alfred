import os
from tars.base.config import Config
from tars.config.main_config import MainConfig


class MetricsEvaluatorConfig(Config):
    
    task_types = ['pick_and_place_simple', 'pick_clean_then_place_in_recep', 'pick_heat_then_place_in_recep',
                    'pick_cool_then_place_in_recep', 'pick_two_obj_and_place', 'look_at_obj_in_light',
                    'pick_and_place_with_movable_recep']
