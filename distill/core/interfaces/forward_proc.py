from .registry import register_forward_proc_func
import inspect


@register_forward_proc_func
def forward_all(model, data, **kwargs):
    # 获取函数的参数签名
    sig = inspect.signature(model.forward)
    # 构建一个字典，包含函数所需的所有参数
    func_args = {name: data[name] for name, param in sig.parameters.items() if name in data}
    return model(**func_args, **kwargs)

@register_forward_proc_func
def forward_batch_only(model, sample_batch, targets=None, supp_dict=None, **kwargs):
    return model(sample_batch)

