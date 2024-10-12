import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GCNConv

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


init_wandb(
    name=f'GCN-{args.dataset}',
    lr=args.lr,
    epochs=args.epochs,
    hidden_channels=args.hidden_channels,
    device=device,
)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

if args.use_gdc:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(hidden_channels, out_channels,
                             normalize=not args.use_gdc)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
).to(device)

optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
times = []
# 训练教师模型
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        torch.save(model.state_dict(), 'teacher_model.pth')  # 保存教师模型
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    times.append(time.time() - start)


# ------------------------

def lsp_loss_function(z_s, z_t, edge_index):
    row, col = edge_index

    # 计算节点嵌入的相似性
    sim_s = F.normalize(torch.exp(-torch.norm(z_s[row] - z_s[col], dim=1, p=2)), dim=0)
    sim_t = F.normalize(torch.exp(-torch.norm(z_t[row] - z_t[col], dim=1, p=2)), dim=0)

    # 计算 KL 散度
    loss_lsp = F.kl_div(sim_s.log(), sim_t, reduction='batchmean')
    return loss_lsp

@torch.no_grad()
def test_model(model):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

# 加载教师模型
teacher_model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
).to(device)
teacher_model.load_state_dict(torch.load('teacher_model.pth'))  # 加载教师模型权重
# 输出最终测试准确率
train_acc, val_acc, test_acc = test_model(teacher_model)
print(f"Final Student Model Accuracy: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")


# 初始化学生模型和 LSP 模块
student_model = GCN(
    in_channels=dataset.num_features,
    hidden_channels=args.hidden_channels,
    out_channels=dataset.num_classes,
).to(device)


optimizer = torch.optim.Adam([
    dict(params=student_model.conv1.parameters(), weight_decay=5e-4),
    dict(params=student_model.conv2.parameters(), weight_decay=0)
], lr=args.lr)  # Only perform weight-decay on first convolution.


def train_student():
    student_model.train()
    optimizer.zero_grad()

    out_s = student_model(data.x, data.edge_index, data.edge_attr)
    out_t = teacher_model(data.x, data.edge_index)

    # 交叉熵损失
    loss_ce = F.cross_entropy(out_s[data.train_mask], data.y[data.train_mask])

    # LSP 损失
    loss_lsp = lsp_loss_function(out_s, out_t, data.edge_index)

    # 总损失
    loss = loss_ce + loss_lsp  # 调整权重
    loss.backward()
    optimizer.step()

    return float(loss)


# 训练学生模型并记录最终的准确率
for epoch in range(1, args.epochs + 1):
    loss = train_student()
    print(f'Epoch {epoch}, Loss: {loss}')
    if epoch % 10 == 0:  # 每 10 个 epoch 打印一次中间结果
        train_acc, val_acc, test_acc = test_model(student_model)
        print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")

# 输出最终测试准确率
train_acc, val_acc, test_acc = test_model(student_model)
print(f"Final Student Model Accuracy: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
