import pandas as pd

input_dir = "result/gru_data.csv"


def change_name(string):
    string = string.replace("gru/gru_", "short/gru/")
    return string


csv = pd.read_csv(input_dir)
csv["model_dict"] = csv["model_dict"].apply(change_name)
csv.to_csv(input_dir)
