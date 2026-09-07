import pandas as pd
import requests
import re
import os
from bs4 import BeautifulSoup
import datetime as dt
from tqdm import tqdm

def get_newscount(stock, startdate, enddate):
    headersParameters={
        'Content-Type': 'application/x-www-form-urlencoded',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/59.0.3071.109 Safari/537.36',
        'Connection': 'keep-alive',
        'Cache - Control': 'no-cache',
        'Cookie':'Hm_lvt_c69f27ad39d01ca2dde8645fb268cd6c=1650373177,1650873088,1651665891,1652596713; JSESSIONID=698802FCD8912084E85E3CFC7DBF0607; 202.120.234.111=698802FCD8912084E85E3CFC7DBF0607; Hm_lpvt_c69f27ad39d01ca2dde8645fb268cd6c=1652602352'
        }
    url = r'http://www.bjinfobank.com/DataList.do?method=DataList'

    search_conds1 = {'iw': stock,
                     'starTime': startdate,
                     'endTime': enddate,
                     'page': '1',
                     'pageSize': '100',
                     'db': 'HK',
                     'query': 'all',
                     'rl': '1',
                     'myorder': 'SUTM'
                     }

    r = requests.post(url, data=search_conds1, headers=headersParameters)
    html = BeautifulSoup(r.text, features="html.parser")
    try:
        tar_num = int(html.find(class_="mlgreen").string.strip())
    except:
        print('\r Can not find data of {} from{} to {}'.format(stock, startdate, enddate))
        tar_num = 'nan'

    return tar_num

def matrix_panel(df):
    df_transposed = df.transpose()
    df_transposed = df_transposed.reset_index().rename(columns={'index': 'Date'})
    panel_data = pd.melt(df_transposed, id_vars=['Date'], var_name='Symbol', value_name='Hist_name')
    panel_data = panel_data[(panel_data['Hist_name'] != 0) & (panel_data['Date'] != '2000-12-31')]

    return panel_data

if __name__ == '__main__':
    stockname_matrix = pd.read_excel("data/stockname_history.xlsx", sheet_name="Sheet1", index_col=0)
    stockname_panel =  matrix_panel(stockname_matrix)
    news_count_ls = []
    grouped = stockname_panel.groupby('Symbol')

    counter = 0
    df = pd.DataFrame()
    for symbol, group in tqdm(grouped):
        news_count_ls = []
        for index, row in group.iterrows():
            stockname = row['Hist_name']
            startdate = str(row['Date'].year) + "-01" + "-01"
            enddate = str(row['Date'].strftime('%Y-%m-%d'))
            news_count = get_newscount(stockname, startdate, enddate)
            news_count_ls.append(news_count)
        group['NewsCount'] = news_count_ls
        df = pd.concat([df, group])

        counter += 1
        if counter % 100 == 0:
            df.to_csv(r'data\news_num_by_firm.csv', index=False)
    # 最后保存结果
    df.to_csv(r'data\news_num_by_firm.csv', index=False)

