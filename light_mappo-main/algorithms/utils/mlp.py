import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""

#layer_N隐藏层重复的数量
#use_orthogonal 0表示不用正交初始化 1表示用正交初始化
#use_ReLU 0表示用tanh激活函数 1表示用ReLU激活函数


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]  ## 根据 use_ReLU 选择激活函数（False→Tanh, True→ReLU）
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        ## 第一层：输入层 Linear(input_dim → hidden_size)
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # 定义一个隐藏层模板（后续复制多份）
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func, nn.LayerNorm(hidden_size))
        # 使用 get_clones 复制 layer_N 个隐藏层（ModuleList）
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        """
        前向传播：
        1. 输入经过 fc1
        2. 依次通过 layer_N 个相同结构的隐藏层
        """
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        #从args中获取参数
        """
        use_feature_normalization:是否对原始输入做 LayerNorm(特征层归一化)。

        use_orthogonal:是否使用正交初始化(传给 MLPLayer)。

        use_ReLUL:是否使用 ReLU 激活（传给 MLPLayer)。

        stacked_frames:帧堆叠数(可能影响输入维度,但在这里没用到)。

        layer_N、hidden_size:传递给 MLPLayer 的层数和隐藏维度
        """
        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0] #输入观测向量的维度

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x)

        return x