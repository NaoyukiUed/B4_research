import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from dataset import split_data
from sklearn.metrics import confusion_matrix


def test_model(test_loader, model, criterion):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    total = 0
    correct = 0  # 用于记录正确的预测数量
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():  # 学習を行わない
        # テストローダとlstmテストローダを読み込む
        for inputs, labels in test_loader:
            # 推論
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    conf = confusion_matrix(all_labels, all_predictions)
    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average="weighted")

    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%,F1 Score: {f1:.4f}')
    return test_loss, test_accuracy, f1, conf


def test_model_auto(data_loader, model, criterion):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0
    total = 0
    correct = 0
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # GPUが使えたらGPUを使う
    with torch.no_grad():
        for inputs, labels in data_loader:
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)  # GPUが使えたらGPUを使う
            # 推論
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    # test_loss = total_loss / len(data_loader)
    test_accuracy = 100 * correct / total
    f1 = 100 * f1_score(all_labels, all_predictions, average="weighted")
    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%,F1 Score: {f1:.4f}')
    return test_accuracy, f1
