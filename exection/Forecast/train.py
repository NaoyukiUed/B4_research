import torch
import torch.nn as nn
import torch.optim as optim
from module import RNN, LSTM, NeuralNetwork, NNk, GRU
from testModel import test_model
from dataset import pr_data
from tqdm import tqdm
from copy import copy
import os
import re


def learning(model, inputs, labels, optimizer, criterion):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    return loss


def train_model(
    mod,
    early_stop,
    num_epochs,
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    batch_size,
    days_num,
    input_dir,
    inputtest_dir,
    model_dict,
    output_file,
    observed,
):
    # logデータ
    # model.train()
    # modelによってデータの形式を変える
    if mod < 2:
        module_num = 1
    elif mod < 6:
        module_num = 0

    # 訓練データを決定する
    train_loader = pr_data(
        input_dir,
        n=days_num,
        batch_size=batch_size,
        module=module_num,
        input_size=input_size,
    )
    test_loader = pr_data(
        inputtest_dir,
        n=days_num,
        batch_size=batch_size,
        module=module_num,
        input_size=input_size,
    )

    # モデルを決定する
    if mod == 0:
        model = NNk(input_size * days_num, hidden_size, num_classes)
    elif mod == 1:
        model = NeuralNetwork(input_size * days_num, hidden_size, num_classes)
    elif mod == 2:
        model = LSTM(input_size, hidden_size, num_layers, num_classes, batch_first=True)
    elif mod == 3:
        model = RNN(input_size, hidden_size, num_layers, num_classes, batch_first=True)
    elif mod == 4:
        model = GRU(input_size, hidden_size, num_classes)

    # 最適化手法と損失関数
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = best_f1 = 0.0
    epoch_cnt = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 入力データを取得する
        for inputs, labels in tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}]",
        ):
            # 学習
            loss = learning(model, inputs, labels, optimizer, criterion)

            # 訓練中に発生するデータを保持する
            running_loss += loss.item()

        # テスト段階
        if observed >0:
            print(f"Loss: {running_loss / len(train_loader)}")
        test_loss, accuracy, f1, conf = test_model(test_loader, model, criterion)

        # 出力段階
        # 良かった場合は記録する

        if observed == 1:
            print(
                f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f},F1 Score: {f1:.4f}"
            )
        elif observed == 2:
            TP = conf[0][0]
            FP = conf[0][1]
            FN = conf[1][0]
            TN = conf[1][1]
            if TP != 0|FN != 0:
                upper_recall = TP / (TP + FN)
            else:
                upper_recall = None
            if TP != 0|FP != 0:
                upper_precision = TP / (TP + FP)
            else:
                upper_precision = None
            if TN != 0 | FP != 0:
                downer_recall = TN / (TN + FP)
            else:
                downer_recall = None
            if TN != 0 | FN != 0:
                downer_precision = TN / (TN + FN)
            else:
                downer_precision = None
            print(
                f"Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f},F1 Score: {f1:.4f},upper_recall:{upper_recall:.4f},upper_precicion:{upper_precision:.4f},downer_recall:{downer_recall:.4f},downer_precicion:{downer_precision:.4f}"
            )
            print(f"{conf}")
        if accuracy > best_accuracy:
            best_conf = conf
            best_accuracy = accuracy
            best_f1 = f1
            best_model = copy(model)
            epoch_cnt = 0

        else:
            epoch_cnt += 1

        # もっともよいモデルを残す
        if epoch_cnt > early_stop:
            model_dir = re.sub(r"\/(?!.*\/).*", "", model_dict)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(best_model.state_dict(), model_dict)
            break

    # 結果の出力
    print("Finished Training")
    print(f"Test Accuracy: {best_accuracy:.2f},F1 Score: {best_f1:.4f}")
    print(f"{best_conf}")

    TP = best_conf[0][0]
    FP = best_conf[0][1]
    FN = best_conf[1][0]
    TN = best_conf[1][1]
    if TP != 0|FN != 0:
        upper_recall = TP / (TP + FN)
    else:
        upper_recall = None
    if TP != 0|FP != 0:
        upper_precision = TP / (TP + FP)
    else:
        upper_precision = None
    if TN != 0 | FP != 0:
        downer_recall = TN / (TN + FP)
    else:
        downer_recall = None
    if TN != 0 | FN != 0:
        downer_precision = TN / (TN + FN)
    else:
        downer_precision = None
    with open(output_file, "a") as f:
        f.write(
            f"{mod},{model_dict},{early_stop},{num_epochs},{input_size},{hidden_size},{num_layers},{num_classes},{batch_size},{days_num},{best_accuracy:.2f},{best_f1:.4f},{upper_recall},{upper_precision},{downer_recall},{downer_precision},{TP},{FP},{FN},{TN}\n"
        )
    # test_model(test_loader, model, criterion)
