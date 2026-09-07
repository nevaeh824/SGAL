import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import PchipInterpolator
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'


FSFP_threshold = pd.read_csv(r'data\FSFP_threshold.csv')

x = FSFP_threshold['threshold']
yp = FSFP_threshold['rec_penalty']
ya = FSFP_threshold['rec_audit']
yr = FSFP_threshold['rec_restate']

zp = FSFP_threshold['prec_penalty']
za = FSFP_threshold['prec_audit']
zr = FSFP_threshold['prec_restate']

plt.figure(figsize=(10,6))

plt.plot(x, yp, 'b-', label='Rec of Penalty')
plt.plot(x, zp, 'r-', label='Prec of Penalty')
plt.plot(x, ya, 'b--', label='Rec of NsAduit')
plt.plot(x, za, 'r--', label='Prec of NsAduit')
plt.plot(x, yr, 'b:', label='Rec of Restate')
plt.plot(x, zr, 'r:', label='Prec of Restate')

plt.xticks(ticks= [0, 0.2, 0.4,0.6,0.8,1.0])
plt.yticks(ticks= [0.2, 0.4,0.6,0.8,1.0])

plt.xlabel('FSFP threshold', fontname='Times New Roman',fontsize=12)
plt.ylabel('Evaluation score', fontname='Times New Roman',fontsize=12)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=12)

plt.grid(True)
plt.tight_layout()
plt.savefig('output/fig-FSFPthreshold.png')
plt.show()

