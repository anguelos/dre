from typing import List, Dict, Tuple, Union, Any, TypeAlias
import torch

t_device: TypeAlias = Union[str, torch.cuda.device]
t_str2int: TypeAlias = Dict[str, int]
t_int2str: TypeAlias = Dict[int, str]
