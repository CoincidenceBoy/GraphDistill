import torch
import torch.nn.functional as F

def register_hooks(training_box, data, num_layers=2):

    model = training_box.model
    training_box.model_io_dict['ndr_values'] = []
    """
    为模型中的每个GATConv层注册forward hook,在每次前向传播时自动计算NDR。
    """
    def hook_fn(module, input, output):
        if len(training_box.model_io_dict['ndr_values']) < num_layers:  # 强制只记录两层的NDR
            ndr = compute_ndr(output, data['edge_index'])
            training_box.model_io_dict['ndr_values'].append(ndr)  # 将NDR存储在模型中
            print(f"Computed NDR for layer {len(training_box.model_io_dict['ndr_values'])}: Mean NDR = {ndr.mean().item():.4f}")

    # 确保不会重复注册hook
    if hasattr(model, 'hook_handles'):
        for handle in model.hook_handles:
            handle.remove()

    model.hook_handles = []  # 清空之前的handle

    for conv in get_convs(model):
        handle = conv.register_forward_hook(hook_fn)
        model.hook_handles.append(handle)

# 计算邻域差异率(NDR)的函数
def compute_ndr(x, edge_index):
    """
    计算邻域差异率 (NDR),用于捕捉节点之间的非平滑性（即,节点与其邻居特征的差异）。
    NDR值越高,表示节点的嵌入特征与邻居的差异性越大,避免过度平滑。
    """
    row, col = edge_index
    deg = torch.bincount(row)  # 计算每个节点的度数
    deg_inv_sqrt = deg.pow(-0.5)  # 计算度的倒数平方根,用于归一化
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # 计算节点特征与邻居特征的加权和
    x_j = x[col]
    out = torch.zeros_like(x)
    out[row] += norm.unsqueeze(-1) * x_j

    # 计算余弦相似度,并将其转换为距离度量,得到NDR
    cosine_similarity = F.cosine_similarity(x, out, dim=1)
    return 1 - cosine_similarity

def get_convs(model):
    if model.__class__.__name__ == 'GCNModel':
        return model.conv
    elif model.__class__.__name__ == 'GATModel':
        return model.gat_list