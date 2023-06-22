import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

# plt.style.use("ggplot")
plt.rcParams["figure.autolayout"] = True
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
import warnings

warnings.filterwarnings("ignore")


angles = pd.read_csv("angles_bicep.csv")
percentages = pd.read_csv("percentages_bicep.csv")
bars = pd.read_csv("bars_bicep.csv")
save_file = "bicep.png"


_, ax1 = plt.subplots(figsize=(12,6))
#plt.title("Performance Tracking of the Bicep Exercise")
plt.grid(True)

ax1.plot(angles, '-r', label="°")
ax1.set_xlabel("Frame", fontsize=12)
ax1.set_ylabel("Angle in degree (°)", fontsize=12)
ax1.legend(loc = "upper left")

ax2 = ax1.twinx()
ax2.plot(percentages, '-b', label="%")
ax2.set_ylabel("How likely to the groundtruth pose (%)", fontsize=12)
ax2.legend(loc = "upper right")

plt.savefig(save_file)
plt.show()