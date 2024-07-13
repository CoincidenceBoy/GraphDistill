
MODEL_DICT = dict()

def get_model(key, *args, **kwargs):
    if key in MODEL_DICT:
        return MODEL_DICT[key](*args, **kwargs)
    raise ValueError('model_name `{}` is not expected'.format(key))

def register_model(arg=None, **kwargs):
    def _register_model(cls):
        key = kwargs.get('key')
        if key is None:
            key = cls.__name__

        MODEL_DICT[key] = cls
        return cls

    if callable(arg):
        return _register_model(arg)
    return _register_model