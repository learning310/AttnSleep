from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import os

checkpoint_dir = './saved/20_AttnSleep'
# 文件列表
outs_list = []
trgs_list = []
save_dir = os.path.abspath(os.path.join(checkpoint_dir, os.pardir))
for root, dirs, files in os.walk(checkpoint_dir):
    for file in files:
        if "outs" in file:
             outs_list.append(os.path.join(root, file))
        if "trgs" in file:
             trgs_list.append(os.path.join(root, file))

# 读取文件数据
all_outs = []
all_trgs = []
for i in range(len(outs_list)):
    outs = np.load(outs_list[i])
    trgs = np.load(trgs_list[i])
    all_outs.extend(outs)
    all_trgs.extend(trgs)
all_trgs = np.array(all_trgs).astype(int)
all_outs = np.array(all_outs).astype(int)

names = ['W', "N1", "N2", "N3", "REM"]
r = classification_report(all_trgs, all_outs, target_names=names, digits=6, output_dict=True)
del(r['accuracy'])
df = pd.DataFrame(r)
df.loc["accuracy"] = accuracy_score(all_trgs, all_outs)
df.loc["cohen"] = cohen_kappa_score(all_trgs, all_outs)
df = df * 100
df.loc["support"] = df.loc["support"] / 100

file_name = 'test' + "_classification_report.xlsx"
report_Save_path = os.path.join(save_dir, file_name)
df.to_excel(report_Save_path)