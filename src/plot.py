import json
import matplotlib.pyplot as plt
from sys import argv
import numpy as np


metrics = ['d_adv_loss', 'd_aux_loss', 'g_adv_loss', 'g_aux_loss', 'd_acc']

config_path = argv[1]
with open(config_path) as f:
    config = json.load(f)


d_real_loss = [i/1000 for i in config['d_real_loss'][:500]]
d_fake_loss = [i/1000 for i in config['d_fake_loss'][:500]]
g_adv_loss = [i/1000 for i in config['g_adv_loss'][:500]]
g_aux_loss = [i for i in config['g_aux_loss'][:500]]

fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=[8,6]);
ax1.plot(d_real_loss,'r',linewidth=3.0)
ax1.plot(d_fake_loss,'b',linewidth=3.0)

ax1.set_xlabel('Epoch',fontsize=16)
ax1.set_ylabel('D Loss',fontsize=16)
ax1.legend(['Real Loss', 'Fake Loss'],fontsize=18)

ax2.plot(g_adv_loss,'r',linewidth=3.0)
ax2.plot(g_aux_loss,'b',linewidth=3.0)
ax2.set_xlabel('Epoch',fontsize=16)
ax2.set_ylabel('G Loss',fontsize=16)
ax2.legend(['Adv Loss', 'Aux Loss'],fontsize=18)

plt.title('AC GAN',fontsize=16)
plt.show()

