import tyro
import yaml
from easydict import EasyDict
import copy

BASE_KEY = 'base'


def merge_a_into_b(a, b):
  # merge dict a into dict b. values in a will overwrite b.
  for k, v in a.items():
    if isinstance(v, dict) and k in b:
      assert isinstance(
        b[k], dict
      ), "Cannot inherit key '{}' from base!".format(k)
      merge_a_into_b(v, b[k])
    else:
      b[k] = v
  pass

def _expand_cfg_base(cfg,
                    loaded_cfg):
  loaded_cfg = copy.deepcopy(loaded_cfg)

  for sub_key in cfg:
    sub_cfg = cfg.get(sub_key)
    if isinstance(sub_cfg, dict):
      _expand_cfg_base(sub_cfg, loaded_cfg)

  if BASE_KEY in cfg:
    base_cfg = loaded_cfg.get(cfg[BASE_KEY])
    del cfg[BASE_KEY]
    _expand_cfg_base(base_cfg, loaded_cfg)
    merge_a_into_b(cfg, base_cfg)
    cfg.clear()
    cfg.update(base_cfg)
  pass

def parse_cfg_from_yaml_cli(cfg_path: str,
                            sub_key: str = None):
  
  with open(cfg_path, 'r') as f:
    cfg_total = EasyDict(yaml.safe_load(f))

  if sub_key is not None:
    cfg = copy.deepcopy(cfg_total[sub_key])
    _expand_cfg_base(cfg=cfg, loaded_cfg=cfg_total)

  else:
    cfg = cfg_total

  cfg_overridden, cfg_unknown = tyro.cli(dict, default=dict(cfg), return_unknown_args=True)

  return EasyDict(cfg_overridden)


