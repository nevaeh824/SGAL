import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from scipy.stats.mstats import winsorize
from matplotlib.font_manager import FontProperties

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 100)
np.random.seed(3)

def FSFP_by_firm():
    df=pd.read_csv(r'data\FSFP.csv')
    plt.figure(dpi=300)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for symbol in tqdm(df["Symbol"].unique()):
        sub_df = df[df["Symbol"] == symbol]
        # plt.scatter(sub_df["DeclareYear"], sub_df["FSFP"], label=symbol, marker='')
        plt.plot(sub_df["ReportYear"], sub_df["FSFP"], linewidth=0.5, markersize=0)

    plt.xticks(ticks=range(2001, 2023, 3))
    plt.xlabel("Year", fontfamily = "Times New Roman")
    plt.ylabel("FS Fraud Propensity (FSFP)", fontfamily = "Times New Roman")
    # plt.title("FSFP Scatter plot for Companies")
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.savefig('output/fig-FSFP.png')
    plt.show()

def FSFP_avg():
    df=pd.read_csv(r'data\FSFP.csv')
    # FSFP_average = final_df.groupby('DeclareYear').aggregate({'FSFP':'mean'}).reset_index()
    grouped = df.groupby('ReportYear')['FSFP'].agg(['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)])
    grouped.columns = ['Mean', '5% percentile', '95% percentile']
    df_year = grouped.reset_index()
    plt.figure(dpi=300)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.plot(df_year["ReportYear"], df_year["Mean"], c='r', linewidth=0.5, markersize=0, label = 'Mean')
    plt.plot(df_year["ReportYear"], df_year["5% percentile"], c='b', ls='-.', linewidth=0.5, markersize=0, label = '5% percentile')
    plt.plot(df_year["ReportYear"], df_year["95% percentile"], c='b', ls='-.', linewidth=0.5, markersize=0, label = '95% percentile')
    plt.xticks(ticks=range(2001, 2023, 3))
    plt.xlabel("Year", fontfamily = "Times New Roman")
    plt.ylabel("FS Fraud Propensity (FSFP)", fontfamily = "Times New Roman")
    # plt.title("FSFP Scatter plot for Companies")
    plt.legend(loc='best',fontsize=8, ncol=3, prop = {'family' : "Times New Roman"})
    plt.show()
    plt.savefig('output/fig-FSFP-distribution.png')

if __name__ == '__main__':
    FSFP_by_firm()
    FSFP_avg()