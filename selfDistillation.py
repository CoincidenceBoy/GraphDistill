import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

# 计算邻域差异率(NDR)的函数
def compute_ndr(x, edge_index):
    """
    计算邻域差异率 (NDR)，用于捕捉节点之间的非平滑性（即，节点与其邻居特征的差异）。
    NDR值越高，表示节点的嵌入特征与邻居的差异性越大，避免过度平滑。
    """
    row, col = edge_index
    deg = torch.bincount(row)  # 计算每个节点的度数
    deg_inv_sqrt = deg.pow(-0.5)  # 计算度的倒数平方根，用于归一化
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    # 计算节点特征与邻居特征的加权和
    x_j = x[col]
    out = torch.zeros_like(x)
    out[row] += norm.unsqueeze(-1) * x_j

    # 计算余弦相似度，并将其转换为距离度量，得到NDR
    cosine_similarity = F.cosine_similarity(x, out, dim=1)
    return 1 - cosine_similarity


# 定义支持n层的GAT模型
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_layers):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()  # 使用ModuleList来管理多层GAT
        self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=0.5))  # 第一层

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads, dropout=0.5))  # 中间层

        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.5))  # 最后一层
        self.ndr_values = []  # 用于存储每层的NDR值

    def forward(self, x, edge_index):
        """
        前向传播函数。
        通过注册的hook，计算每一层的NDR值，并将其存储在self.ndr_values中。
        """
        for conv in self.convs:
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(conv(x, edge_index))  # 执行每一层卷积
        return x


# 注册hook函数
def register_hooks(model, data, num_layers=2):
    """
    为模型中的每个GATConv层注册forward hook，在每次前向传播时自动计算NDR。
    """
    def hook_fn(module, input, output):
        if len(model.ndr_values) < num_layers:  # 强制只记录两层的NDR
            ndr = compute_ndr(output, data.edge_index)
            model.ndr_values.append(ndr)  # 将NDR存储在模型中
            print(f"Computed NDR for layer {len(model.ndr_values)}: Mean NDR = {ndr.mean().item():.4f}")

    # 确保不会重复注册hook
    if hasattr(model, 'hook_handles'):
        for handle in model.hook_handles:
            handle.remove()

    model.hook_handles = []  # 清空之前的handle

    # 为每一层的GATConv注册forward hook
    for conv in model.convs:
        handle = conv.register_forward_hook(hook_fn)
        model.hook_handles.append(handle)


# 加载数据集
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Planetoid(root='./data/Planetoid', name='cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

# 初始化模型，指定层数，比如4层或其他层数
num_layers = 4  # 你可以更改为2, 8, 16等层数
model = GAT(dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes, heads=4, num_layers=num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 注册hook
register_hooks(model, data, num_layers)


# 自定义损失函数
def custom_loss(out, labels, mask, model, alpha=0.01, beta=0.001, gamma=0.001):
    """
    自定义损失函数，包括交叉熵损失和自适应差异保持（ADR）正则化。
    计算所有相邻层的NDR差异。
    """
    # 1. 计算交叉熵损失 (CE)
    loss = F.cross_entropy(out[mask], labels[mask])

    # 2. LL：浅层和深层logits蒸馏损失
    # 自适应差异保持(ADR)正则化：计算所有相邻层之间的NDR差异
    if len(model.ndr_values) >= 2:  # 确保有多层NDR值
        for i in range(len(model.ndr_values) - 1):
            ndr1 = model.ndr_values[i]  # 第i层的NDR
            ndr2 = model.ndr_values[i + 1]  # 第i+1层的NDR
            adr_loss = torch.mean(F.relu(ndr1 - ndr2))  # 如果第i层的NDR大于第i+1层，则增加惩罚
            loss += alpha * adr_loss  # 将ADR正则化项加入到总损失中
            print(f"ADR Loss between layer {i + 1} and layer {i + 2}: {adr_loss.item():.4f}")

    # 3. LN：邻域差异率（NDR）的正则化项，防止特征的过度平滑
    # 假设 LN 的逻辑是对某一层的特征进行差异化约束
    if len(model.ndr_values) >= 1:
        ln_loss = torch.mean(torch.stack(model.ndr_values))  # 假设 LN 对所有层的NDR平均
        loss += beta * ln_loss  # 加入LN正则化项
        print(f"LN Loss (NDR regularization): {ln_loss.item():.4f}")

    # 4. LG：图级别嵌入的蒸馏损失
    # 这里需要模型输出的图嵌入表示，并计算图级别的损失。
    if hasattr(model, 'graph_embeds'):
        # 假设 model.graph_embeds 是存储图嵌入表示的列表
        g_embeds = model.graph_embeds  # 取图嵌入表示
        if len(g_embeds) >= 2:
            lg_loss = torch.mean((g_embeds[-1] - g_embeds[0]) ** 2)  # 两层图嵌入之间的蒸馏
            loss += gamma * lg_loss  # 加入LG正则化项
            print(f"LG Loss (Graph Embedding distillation): {lg_loss.item():.4f}")

    # 清空NDR缓存，以便下一次前向传播重新计算
    model.ndr_values.clear()
    return loss


# 训练函数
def train():
    """
    模型的训练函数。每次训练时，自动计算损失函数，并根据优化器更新模型参数。
    """
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = custom_loss(out, data.y, data.train_mask, model)
    loss.backward()
    optimizer.step()
    return loss.item()


# 测试函数
@torch.no_grad()
def test():
    """
    测试模型在训练集、验证集和测试集上的性能，返回每个集合的准确率。
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
        accs.append(acc)
    return accs


best_acc = 0.0
# 主训练循环
for epoch in range(1, 301):
    loss = train()
    train_acc, val_acc, test_acc = test()
    best_acc = max(best_acc, test_acc)

    # 打印训练过程中的每个关键阶段信息
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

print(f'Final Best Test Accuracy: {best_acc:.4f}')
