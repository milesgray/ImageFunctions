from .coords import *
from .files import *
from .image import *
from .log import *
from .math import *
from .model import *
from .random import *
from .freq import *
from .freq_np import *
from .tracker import *

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