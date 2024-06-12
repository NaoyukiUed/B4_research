import sys
sys.path.append("c:\\Users\\noyku\\Desktop\\study\\exection\\Forecast")
sys.path.append("c:\\users\\noyku\\appdata\\local\\programs\\python\\python39\\lib\\site-packages")
import torch
from train import train_model
import os
import re

time = sys.argv[1]
type = sys.argv[2]

# GPUが使えるならばGPUを使う
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 変数
# observed 0:表示しない,1:ちょっと表示,2:結構表示
#出力先のディレクトリを決定する
if time == "short":
    input_dir = "exection/DATA/train0721"
    inputtest_dir = "exection/DATA/test2223"
    days_num = 5  # 何日分を使って予測を行うか
elif time == "middle":
    input_dir = "exection/DATA/30train05_21"
    inputtest_dir = "exection/DATA/30test22_23"
    days_num = 30  # 何日分を使って予測を行うか
observed = 0
pth_file = f"pth/{time}/{type}"
output_file = f"result/{time}/{type}.csv"


early_stop = 1  # アーリーストップ
num_epochs = 10  # epoch数
input_size = 4  # 入力データサイズ
hidden_size = 512  # 再起型ネットワークの中間層および出力層の総数
num_layers = 2  # LSTMを何個重ねるか
num_classes = 2  # 分類数
batch_size = 32  # バッチサイズ
if type == "nn":
    mod = 0  # どのモデルを用いるか
elif type == "lstm":
    mod = 2  # どのモデルを用いるか
elif type == "rnn":
    mod = 3  # どのモデルを用いるか
elif type == "gru":
    mod = 4  # どのモデルを用いるか
output_dir = re.sub(r"\/(?!.*\/).*", "", output_file)
os.makedirs(output_dir, exist_ok=True)
try:
    with open(output_file, "x") as f:
        f.write(
            "mod,model_dict,early_stop,num_epochs,input_size,hidden_size,num_layers,num_classes,batch_size,days_num,best_accuracy,best_f1,upper_recall,upper_precision,downer_recall,downer_precision,TP,FP,FN,TN\n"
        )
except FileExistsError:
    pass

for i in range(100):
    for size in range(4):
        for j in range(1000):
            model_dict = pth_file + f"/{j}_{size}.pth"
            if not os.path.isfile(model_dict):
                break
        input_size = size + 4
        train_model(
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
        )
