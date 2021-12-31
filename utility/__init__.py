from .coords import *
from .config import *
from .files import *
from .image import *
from .log import *
from .math import *
from .model import *
from .random import *
from .spectral import *
from .freq import *
from .tracker import *
from .visualize import *
from .tensor import *
from .torch import *

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))

def make_tuple(x, n):
    if is_sequence(x):
        return x
    return tuple([x for _ in range(n)])

def dict_apply(source, fn=lambda x: x):
    try:
        result = source.copy()
        for k,v in result.items():
            result[k] = fn(v)
    except Exception as e:
        print(f"dict_apply failure:\t{e}")
        return source
    return result

def dict_deep_get(d, key_path, split_ch='/', default=None, create_if_missing=False, dict_type=dict):
    if type(key_path) is str:
        parts = key_path.split(split_ch)
    elif type(key_path) is list:
        parts = key_path
    else:
        assert False
    for i, part in enumerate(parts):
        is_last = (i == len(parts)-1)
        if part in d:
            d = d[part]
        else:
            if create_if_missing:
                if is_last:
                    d[part] = default
                else:
                    d[part] = dict_type()
                d = d[part]
            else:
                return default
    return d

def dict_flatten(d, prefix='', join_char='/'):
    out = {}
    for k, v in d.items():
        cur_k = k if prefix == '' else prefix + join_char + k
        if isinstance(v, dict):
            out.update(dict_flatten(v, cur_k, join_char))
        elif isinstance(v, list) or isinstance(v, tuple):
            out.update(dict_flatten({f'{i}': a for i, a in enumerate(v)}, cur_k, join_char))
        else:
            out[cur_k] = v
    return out