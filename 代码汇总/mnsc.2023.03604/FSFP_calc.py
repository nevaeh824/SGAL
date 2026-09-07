import pandas as pd
import numpy as np
import jieba
import os
import re
import glob
import logging
from datetime import datetime

logging.basicConfig(
    filename=r'log\FSFP_calc.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Script execution started.")

try:
    with open('terms/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    logging.info("Loaded stopwords.")

    with open('terms/finwords.txt', 'r', encoding='utf-8') as f:
        finwords = f.read().splitlines()
    logging.info("Loaded financial keywords.")

    with open('terms/fraudwords.txt', 'r', encoding='utf-8') as f:
        fraudwords = f.read().splitlines()
    logging.info("Loaded fraud-related keywords.")

    with open('terms/stocknames.txt', 'r', encoding='utf-8') as f:
        stocknames = f.read().splitlines()
    logging.info("Loaded stock names.")
except Exception as e:
    logging.error(f"Error loading term files: {e}")
    raise


def news_process():
    try:
        folder_path = 'news\data'
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        logging.info(f"Found {len(csv_files)} news files in folder '{folder_path}'.")

        df = []
        for file in csv_files:
            news = pd.read_csv(file)
            file_name = os.path.splitext(os.path.basename(file))[0]
            news['Symbol'] = file_name
            df.append(news)
        df_all = pd.concat(df, ignore_index=True)
        df_all.dropna(inplace=True)
        logging.info("Loaded and concatenated news files.")

        def filter_row(content):
            paragraphs = content.split('\n')
            for paragraph in paragraphs:
                if any(keyword in paragraph for keyword in finwords) and any(term in paragraph for term in fraudwords):
                    return True
            return False

        df_filterd = df_all[df_all['Content'].apply(filter_row)]
        logging.info("Filtered news content based on financial and fraud keywords.")

        def remove_non_chinese(string):
            if not isinstance(string, str):
                return ''
            return re.sub(r'[^\u4e00-\u9fa5]', '', string)

        df_filterd['Content_cn'] = df_filterd['Content'].apply(remove_non_chinese)
        logging.info("Removed non-Chinese characters from content.")

        jieba.load_userdict(finwords + fraudwords + stocknames)
        df_filterd['Content_cut'] = df_filterd['Content_cn'].apply(
            lambda x: " ".join([word for word in jieba.cut(x) if word not in stopwords]))
        logging.info("Performed word segmentation and removed stopwords.")

        output_path = r'news\news_processed.csv'
        df_filterd.to_csv(output_path, index=False, encoding='utf-8-sig')
        logging.info(f"Processed news data saved to '{output_path}'.")
    except Exception as e:
        logging.error(f"Error in news_process: {e}")
        raise


def FSFP_calc():
    try:
        df = pd.read_csv(r'news\news_processed.csv')
        logging.info("Loaded processed news data.")

        firm_num = pd.read_csv(r'data\firm_num.csv', usecols=['Year', 'Firm_num'])
        logging.info("Loaded firm number data.")

        news_num = pd.read_csv(r'data\news_num_by_firm.csv', usecols=['Symbol', 'Year', 'News_count'])
        logging.info("Loaded news count data.")

        df = pd.merge(df, firm_num, on=['Year'], how='left')
        df = pd.merge(df, news_num, on=['Symbol', 'Year'], how='left')
        logging.info("Merged data for FSFP calculation.")

        def count_terms(tokens, term_set):
            return sum(token in term_set for token in tokens.split())

        df['Term_freq'] = df['Content_cut'].apply(lambda x: count_terms(x, fraudwords))
        df['Firm_freq'] = df['Content_cut'].apply(lambda x: count_terms(x, stocknames))

        df['ATF_IIF'] = df['Term_freq'] * (df['Firm_freq'] / df['Firm_num'])
        df_grouped = df.groupby(['Symbol', 'Year'])['ATF_IIF'].sum().reset_index()
        df_grouped = pd.merge(df_grouped, news_num, on=['Symbol', 'Year'], how='left')
        df_grouped['FSFP'] = df_grouped['ATF_IIF'] / np.log(df_grouped['News_count'])
        logging.info("Calculated FSFP values.")

        output_path = r'data\FSFP.csv'
        df_grouped.to_csv(output_path, index=False)
        logging.info(f"FSFP data saved to '{output_path}'.")
    except Exception as e:
        logging.error(f"Error in FSFP_calc: {e}")
        raise


if __name__ == "__main__":
    start_time = datetime.now()
    logging.info("Starting news processing and FSFP calculation.")

    try:
        news_process()
        FSFP_calc()
    except Exception as e:
        logging.error(f"Script failed with error: {e}")

    end_time = datetime.now()
    logging.info(f"Script execution completed. Total time: {end_time - start_time}.")
