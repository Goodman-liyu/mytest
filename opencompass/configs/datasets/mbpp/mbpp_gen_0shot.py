from mmengine.config import read_base

with read_base():
    from .mbpp_liyu_0shot import mbpp_datasets  # noqa: F401, F403
