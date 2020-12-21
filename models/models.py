import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model

def summary(model, logger=print):
    linfo_name = "LAYER NAME"
    linfo_shape = "LAYER SHAPE"
    linfo_params = "TOTAL PARAMS"
    linfo_mean = "MEAN"
    linfo_var = "VARIANCE"
    linfo_norm = "NORM"
    linfo_nansum = "NANSUM"
    logger(f"[{linfo_name:45s}]\t{linfo_shape:25s}\t|{linfo_params:18s}|{linfo_mean:8s}|{linfo_var:8s}|{linfo_norm:8s}|{linfo_nansum:8s}")
    for layer in list(model.keys()):
        if "opt" in layer: continue
        try:
            real_layer = model[layer]
            if isinstance(real_layer, (dict, list)):
                summary(real_layer)
            else:
                linfo = munch.DefaultMunch()
                linfo.shape = real_layer.shape
                linfo.name = layer
                linfo.params = linfo.shape.numel()
                linfo.mean = float(real_layer.mean().cpu().numpy())
                linfo.var = float(real_layer.var().cpu().numpy())
                linfo.norm = float(real_layer.norm().cpu().numpy())
                linfo.nansum = float(real_layer.nansum().cpu().numpy())
                logger(f"[{linfo.name:45s}]\t{str(linfo.shape):25s}\t|{linfo.params:17d} |{linfo.mean:5.7f} |{linfo.var:5.7f} |{linfo.norm:5.7f} |{linfo.nansum:5.7f} ")
        except:
            logger(f"[{linfo.name:45s}]\t{str(linfo.shape):25s}\t|{linfo.params:17d}")
