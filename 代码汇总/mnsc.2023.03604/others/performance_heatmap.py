import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textwrap import wrap
import matplotlib

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 500)

matplotlib.rcParams['font.family'] = 'Times New Roman'

data = pd.read_excel("data\eval_chg.xlsx", sheet_name='evl_chg',index_col=0)

cmap = sns.diverging_palette(10, 240, as_cmap=True)

plt.figure(figsize=(12, 10))

# columns = [ '\n'.join(wrap(l, 13)) for l in data.columns ]
# print(data.columns)
# exit()
sns.heatmap(data, cmap=cmap, center=0, vmin=-0.2, vmax=0.2,
            cbar_kws={"shrink": 0.75, "ticks": [-0.2, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.2]},
            xticklabels=data.columns)

plt.tight_layout()
plt.savefig('output/fig-heatmap.png')
plt.show()

