import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp

adni = pd.read_csv('./adni_tau_amyloid.csv')
folder='../results/DataExplore/adniExplore'
if not osp.exists(folder):
    os.makedirs(folder)
print(adni.head())

plt.figure()
plt.xlabel("age")
plt.ylabel("AV45")
plt.scatter(adni['t.age'],adni['AV45'])
# plt.show()

plt.figure()
plt.xlabel("age")
plt.ylabel("TAU")
plt.scatter(adni['t.age'],adni['TAU'])
# plt.show()

groups = adni.groupby('DX.bl')
fig, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group['t.age'],group.AV45,label=name)
ax.legend()
plt.xlabel("age")
plt.ylabel("AV45")
plt.savefig(folder + "/AV45_Age")

# plt.show()

groups = adni.groupby('DX.bl')
fig, ax = plt.subplots()
for name, group in groups:
    ax.scatter(group['t.age'],group.TAU,label=name)
ax.legend()
plt.xlabel("age")
plt.ylabel("TAU")
# plt.show()
plt.savefig(folder + "/TAU_Age")
