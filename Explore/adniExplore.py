import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np

adni = pd.read_csv('/N/u/liyuny/Quartz/cnode_ffr_main/data/adni_tau_amyloid.csv')
folder='/N/slate/liyuny/cnode_ffr_main/results/adni/adniExplore'
if not osp.exists(folder):
    os.makedirs(folder)
print(adni.head())
# adni=adni[adni['DX.bl'].isin(['AD','EMCI','LMCI'])]
plt.figure()
plt.xlabel("age")
plt.ylabel("AV45")
plt.plot(adni['t.age.month'],adni['AV45'],linewidth=0.3, alpha=0.7)
# plt.show()
#
plt.figure()
plt.xlabel("age")
plt.ylabel("TAU")
plt.plot(adni['t.age.month'],adni['TAU'],linewidth=0.3, alpha=0.7)
# plt.show()
#
# groups = adni.groupby('DX.bl')
# fig, ax = plt.subplots()
# for name, group in groups:
#     ax.plot(group['t.age'],group.AV45,label=name,linewidth=0.3, alpha=0.7)
# ax.legend()
# plt.xlabel("age")
# plt.ylabel("AV45")
# # plt.savefig(folder + "/AV45_Age")
#
# # plt.show()
#
# groups = adni.groupby('DX.bl')
# fig, ax = plt.subplots()
# for name, group in groups:
#     ax.plot(group['t.age'],group.TAU,label=name,linewidth=0.3, alpha=0.7)
# ax.legend()
# plt.xlabel("age")
# plt.ylabel("TAU")
# # plt.show()
# # plt.savefig(folder + "/TAU_Age")
#
groups = adni.groupby('RID')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['t.age.month'],group.AV45,label=name,linewidth=1, alpha=0.7)
# ax.legend()
plt.xlabel("age")
plt.ylabel("AV45")
# plt.savefig(folder + "/AV45_Age")
# plt.show()

groups = adni.groupby('RID')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group['t.age.month'],group.TAU,label=name,linewidth=0.3, alpha=0.7)
# ax.legend()
plt.xlabel("age")
plt.ylabel("TAU")
plt.show()
# plt.savefig(folder + "/TAU_Age")