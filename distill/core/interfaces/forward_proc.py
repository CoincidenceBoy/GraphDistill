from .registry import register_forward_proc_func

@register_forward_proc_func
def forward_batch_only(model, sample_batch, targets=None, supp_dict=None, **kwargs):
    return model(sample_batch)

