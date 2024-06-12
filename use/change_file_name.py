import os

dir = "pth/gru"
files = os.listdir(dir)
for file in files:
    renamed_file = file.replace("gru_", "")
    os.rename(f"{dir}/{file}", f"{dir}/{renamed_file}")
