import shutil
import os
copy_file = "exection/Forecast/training/short_gru.py"
times = ["short","middle"]
types = ["nn","rnn","lstm"]
head_path = os.path.split(copy_file)[0]
for time in times:
    for type in types:
        copied_file = f"{head_path}/{time}_{type}.py"
        shutil.copy(copy_file,copied_file)
