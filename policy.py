import numpy as np
import paddle.nn as nn
import paddle

from config import *


class PolicyValueModelBase(nn.Layer):
    @paddle.no_grad()
    def policy_value_fn(self, board):
        legal_positions = board.availables
        curr_state = np.expand_dims(board.state, axis=0)
        act_probs, value = self(paddle.to_tensor(curr_state, dtype=paddle.float32))
        act_probs = zip(legal_positions, act_probs.numpy()[0][legal_positions])
        return act_probs, value[0]


class PolicyValueModel(PolicyValueModelBase):
    def __init__(self):
        super().__init__()
        l2_const = 1e-4
        self.base_net = nn.Sequential(
            nn.Conv2D(
                CHANNELS,
                32,
                3,
                stride=1,
                padding="same",
                weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(coeff=l2_const)),
                data_format="NHWC",
            ),
            nn.ReLU(),
            nn.Conv2D(
                32,
                64,
                3,
                stride=1,
                padding="same",
                weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(coeff=l2_const)),
                data_format="NHWC",
            ),
            nn.ReLU(),
            nn.Conv2D(
                64,
                128,
                3,
                stride=1,
                padding="same",
                weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(coeff=l2_const)),
                data_format="NHWC",
            ),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Conv2D(
                128,
                4,
                1,
                stride=1,
                padding="same",
                weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(coeff=l2_const)),
                data_format="NHWC",
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                WIDTH * HEIGHT * 4,
                WIDTH * HEIGHT,
                weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(coeff=l2_const)),
            ),
            nn.Softmax(),
        )
        self.values = nn.Sequential(
            nn.Conv2D(
                128,
                2,
                1,
                stride=1,
                padding="same",
                weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(coeff=l2_const)),
                data_format="NHWC",
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                2 * WIDTH * HEIGHT,
                64,
                weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(coeff=l2_const)),
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                1,
                weight_attr=paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(coeff=l2_const)),
            ),
            nn.Tanh(),
        )

    def forward(self, inputs):
        x = self.base_net(inputs)
        policy = self.policy(x)
        values = self.values(x)
        return policy, values


class PolicyValueModelResNet(PolicyValueModelBase):
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Pad2D(padding=(1, 1, 1, 1), data_format="NHWC"),
            nn.Conv2D(CHANNELS, 32, 3, stride=1, padding="same", bias_attr=False, data_format="NHWC"),
            nn.BatchNorm2D(32, 32, epsilon=1.001e-5, data_format="NHWC"),
            nn.ReLU(),
        )
        self.res_1 = nn.Sequential(
            nn.Conv2D(32, 32, 3, stride=1, padding="same", bias_attr=False, data_format="NHWC"),
            nn.BatchNorm2D(32, 32, epsilon=1.001e-5, data_format="NHWC"),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, stride=1, padding="same", bias_attr=False, data_format="NHWC"),
            nn.BatchNorm2D(32, 32, epsilon=1.001e-5, data_format="NHWC"),
        )
        self.res_2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, stride=1, padding="same", bias_attr=False, data_format="NHWC"),
            nn.BatchNorm2D(32, 32, epsilon=1.001e-5, data_format="NHWC"),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, stride=1, padding="same", bias_attr=False, data_format="NHWC"),
            nn.BatchNorm2D(32, 32, epsilon=1.001e-5, data_format="NHWC"),
        )
        self.res_3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, stride=1, padding="same", bias_attr=False, data_format="NHWC"),
            nn.BatchNorm2D(32, 32, epsilon=1.001e-5, data_format="NHWC"),
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, stride=1, padding="same", bias_attr=False, data_format="NHWC"),
            nn.BatchNorm2D(32, 32, epsilon=1.001e-5, data_format="NHWC"),
        )
        self.postprocess = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(32, 32, 3, stride=1, padding="same", bias_attr=False, data_format="NHWC"),
            nn.BatchNorm2D(32, 32, epsilon=1.001e-5, data_format="NHWC"),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Conv2D(32, 4, 1, stride=1, padding="same", data_format="NHWC"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((WIDTH + 2) * (HEIGHT + 2) * 4, WIDTH * HEIGHT),
            nn.Softmax(),
        )
        self.values = nn.Sequential(
            nn.Conv2D(32, 2, 1, stride=1, padding="same", data_format="NHWC"),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((WIDTH + 2) * (HEIGHT + 2) * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        x = inputs
        x = self.preprocess(x)
        x = self.res_1(x) + x
        x = self.res_2(x) + x
        x = self.res_3(x) + x
        x = self.postprocess(x)
        policy = self.policy(x)
        values = self.values(x)
        return policy, values


def mean_policy_value_fn(board):
    availables = board.availables
    action_probs = np.ones(len(availables)) / len(availables)
    return zip(availables, action_probs), None


class AlphaZeroError(nn.Layer):
    """AlphaZero Loss 函数"""

    def forward(self, mcts_probs, policy, rewards, values):

        assert rewards.shape == values.shape

        policy_loss = nn.functional.cross_entropy(policy, mcts_probs, soft_label=True)
        value_loss = nn.functional.mse_loss(values, rewards)
        total_loss = (value_loss + policy_loss).mean()

        return total_loss
