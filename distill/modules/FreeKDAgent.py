import tensorlayerx as tlx
import tensorlayerx.nn as nn

from .registry import register_model


@register_model(key='FreeKDAgent')
class FreeKDAgent(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FreeKDAgent, self).__init__()
        # 定义全连接层，参数顺序不同
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2_node = nn.Linear(in_features=hidden_dim, out_features=2)
        self.fc2_structure = nn.Linear(in_features=hidden_dim, out_features=2)

    def forward(self, state):
        # 使用 TensorLayerX 中的 relu 激活函数
        x = tlx.relu(self.fc1(state))
        # 使用 softmax 函数, dim 参数在 TensorLayerX 中是 axis
        node_action_probs = tlx.softmax(self.fc2_node(x), axis=1)
        structure_action_probs = tlx.softmax(self.fc2_structure(x), axis=1)
        return node_action_probs, structure_action_probs




