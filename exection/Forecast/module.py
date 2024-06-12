import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, num_classes, batch_first=True
    ):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=batch_first
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x的维度应为 [batch_size, sequence_length, input_size]
        # LSTM的输出为 (output, (h_n, c_n))
        output, (h_n, c_n) = self.lstm(x)
        output = output[:, -1, :]  # 选择最后一个时间步的输出
        # 使用全连接层进行分类
        output = self.fc(output)
        return output


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, num_classes)  # 第二个全连接层

    def forward(self, x):
        # 输入 x 的维度应为 [batch_size, input_size]
        x = self.fc1(x)  # 第一个全连接层
        x = self.relu(x)  # 使用ReLU激活函数
        x = self.fc2(x)  # 第二个全连接层
        return x


class NNk(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NNk, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 输入 x 的维度应为 [batch_size, input_size]
        x = self.fc1(x)  # 第一个全连接层
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)  # 使用ReLU激活函数
        x = self.fc3(x)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_first):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x的维度应为 [batch_size, sequence_length, input_size]
        # RNN的输出为 (output, h_n)
        output, _ = self.rnn(x)

        # 通常，分类任务的输出是最后一个时间步的输出
        # 你可以选择不同的策略来选择输出，例如平均池化等
        output = output[:, -1, :]  # 选择最后一个时间步的输出

        # 使用全连接层进行分类
        output = self.fc(output)
        return output


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y_rnn, h = self.rnn(x, None)
        y = self.fc(y_rnn[:, -1, :])
        return y


class ComplexModel(nn.Module):
    def __init__(
        self,
        rnn_input_size,
        rnn_hidden_size,
        rnn_num_layers,
        lstm_input_size,
        lstm_hidden_size,
        lstm_num_layers,
        rnn_batch_first,
        lstm_batch_first,
        num_classes,
    ):
        super(ComplexModel, self).__init__()
        self.rnn = nn.RNN(
            rnn_input_size, rnn_hidden_size, rnn_num_layers, batch_first=rnn_batch_first
        )
        self.lstm = nn.LSTM(
            lstm_input_size,
            lstm_hidden_size,
            lstm_num_layers,
            batch_first=lstm_batch_first,
        )
        # print(rnn_hidden_size, lstm_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size + lstm_hidden_size, num_classes)

    def forward(self, rnn_x, lstm_x):
        rnn_output, rnn_h_n = self.rnn(rnn_x)
        rnn_output = rnn_output[:, -1, :]

        lstm_output, lstm_h_n = self.lstm(lstm_x)
        lstm_output = lstm_output[:, -1, :]

        output = torch.cat([rnn_output, lstm_output], dim=1)

        output = self.fc(output)
        return output


if __name__ == "__main__":
    # 定义模型参数
    input_size = 7  # 输入数据的特征维度
    hidden_size = 128  # LSTM隐藏层的大小
    num_layers = 2  # LSTM层数
    num_classes = 2  # 分类的类别数
    batch_size = 16  # 批次大小
    sequence_length = 3  # 序列长度

    # 初始化分类模型
    lstm_classifier = LSTM(
        input_size, hidden_size, num_layers, num_classes, batch_first=True
    )

    # 生成虚拟输入数据
    input_data = torch.randn(batch_size, sequence_length, input_size)

    # 将输入数据传递给模型进行分类
    output = lstm_classifier(input_data)
    print(output)

    # 在输出上应用softmax等操作以获得类别概率
    output_probs = torch.softmax(output, dim=1)
    print(output_probs)
    predicted_classes = torch.argmax(output_probs, dim=1)
    print(predicted_classes)
