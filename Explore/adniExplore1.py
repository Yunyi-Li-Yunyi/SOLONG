import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np

adni = pd.read_csv('/N/u/liyuny/Quartz/cnode_ffr_main/data/adData.csv') #ELOAD data
folder='/N/slate/liyuny/cnode_ffr_main/results/adni/adniExplore1'
if not osp.exists(folder):
    os.makedirs(folder)
print(adni.head())
adni=adni[adni['tage']!=0]
adniEOAD=adni[adni['onset'].isin(['EOAD'])]
adniLOAD=adni[adni['onset'].isin(['LOAD'])]
plt.figure()
plt.xlabel("age")
plt.ylabel("VMncvt")
plt.plot(adniEOAD['tage'],adniEOAD['VMncvt'],linewidth=0.3, alpha=0.7)

plt.xlabel("age")
plt.ylabel("VMncvt")
plt.plot(adniLOAD['tage'],adniLOAD['VMncvt'],linewidth=0.3, alpha=0.7)
plt.show()

groups = adniEOAD.groupby('RID')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['tage'],group.VMncvt,label=name,linewidth=1, alpha=0.7)
# ax.legend()
plt.xlabel("age")
plt.ylabel("VMncvt")
# plt.savefig(folder + "/AV45_Age")
plt.show()

groups = adniLOAD.groupby('RID')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['tage'],group.VMncvt,label=name,linewidth=0.3, alpha=0.7)
# ax.legend()
plt.xlabel("age")
plt.ylabel("VMncvt")
plt.show()
