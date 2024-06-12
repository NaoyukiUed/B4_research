import csv
import re

csv_path = "C:/Users/admin/Desktop/study_new/result/short/gru.csv"
tex_path = csv_path.replace("csv", "tex")
tex_path = re.sub(
    "[:/a-zA-Z_0-9]+/",
    "C:/Users/admin/Desktop/study_new/latex/figure/",
    tex_path,
)

# 読み込み
with open(csv_path) as f:
    reader = csv.reader(f)
    reader = [row for row in reader]

# 書き出し
with open(tex_path, "w") as tex_file:
    tex_file.write("\\begin{tabular}{")
    for _ in range(len(reader[0])):
        tex_file.write("c")
    tex_file.write("}\n")
    for num_list in reader:
        tex_file.write("\t")
        for i, num in enumerate(num_list):
            tex_file.write(f"{num}".replace("_", "\\_"))
            if i + 1 != len(num_list):
                tex_file.write("&")
        tex_file.write("\\\\\n")
    tex_file.write("\\end{tabular}")
