from livelossplot import PlotLosses
import time
from matplotlib import pyplot as plt


liveloss = PlotLosses(groups={'loss': ['train-loss', 'val-loss'], 'lr': ['lr'], 'acc': ['train-acc', 'val-acc']}, mode='script')
for i in range(3):
    logs = {'train-loss': 1/(i+1), 'lr': 0.1/(i+1), 'train-acc': 1/(i+1)}
    liveloss.update(logs)
    logs = {'val-loss': 1/(i+1), 'val-acc': 0.1/(i+1)}
    liveloss.update(logs)
    liveloss.send()
    # time.sleep(1)
