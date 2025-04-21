from mmengine.config import read_base

with read_base():
    from .math_prm800k_500_4shot_cot_gen import math_datasets  # noqa: F401, F403
