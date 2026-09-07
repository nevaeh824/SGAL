import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
import jieba
from tqdm import tqdm
import os
import re
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    filename='log/bus_sim_calc.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Script execution started.")

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 500)


# This data is obtained by crawling the part of “Business Summary” in the annual financial reports of all listed companies in the A-share market from 2001 to 2022 from Eastmoney.

def data_process():
    try:
        logging.info("Data processing started.")
        df = pd.read_csv(r"data\firm_bus_text.csv",
                         usecols=["Symbol", "Year", "BusSum"])
        df = df[df["Year"].between(2001, 2022)]
        df.dropna(inplace=True)
        logging.info(f"Loaded {len(df)} rows of business summary data.")

        def remove_non_chinese(string):
            if not isinstance(string, str):
                return ''
            return re.sub(r'[^\u4e00-\u9fa5]', '', string)

        df['BusSum_clear'] = df['BusSum'].apply(remove_non_chinese)

        with open('terms/stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = f.read().splitlines()
        # jieba.load_userdict("terms/finwords.txt")
        df['BusSum_cut'] = df['BusSum_clear'].apply(
            lambda x: " ".join([word for word in jieba.cut(x) if word not in stopwords]))

        df.to_csv('data/BusSum_cut.csv', index=False, encoding='utf-8-sig')
        logging.info("Processed business summary and saved to 'data/BusSum_cut.csv'.")
    except Exception as e:
        logging.error(f"Error in data_process: {e}")
        raise


def panel_to_matrix(data):
    try:
        logging.info("Converting similarity data to matrix format.")
        symbols = sorted(set(data['Symbol1']).union(set(data['Symbol2'])))

        similarity_matrix = pd.DataFrame(index=symbols, columns=symbols)
        similarity_matrix = similarity_matrix.fillna(0)

        for index, row in data.iterrows():
            symbol1 = row['Symbol1']
            symbol2 = row['Symbol2']
            similarity = row['Similarity']
            similarity_matrix.loc[symbol1, symbol2] = similarity
            similarity_matrix.loc[symbol2, symbol1] = similarity

        return similarity_matrix
    except Exception as e:
        logging.error(f"Error in panel_to_matrix: {e}")
        raise


def bus_sim_gen():
    try:
        logging.info("Business similarity generation started.")
        df = pd.read_csv('data/BusSum_cut.csv')
        df['BusSum_cut'] = df['BusSum_cut'].astype(str)
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df["BusSum_cut"])]

        # Training the Doc2Vec model
        model = Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
        model.save('model/doc2vec_bus')
        model = Doc2Vec.load('model/doc2vec_bus')
        logging.info("Doc2Vec model trained and saved.")

        df["Vector"] = df["BusSum_cut"].apply(lambda x: model.infer_vector(x.split()))

        results = []
        for year in tqdm(df["Year"].unique()):
            results = []
            yearly_data = df[df["Year"] == year]
            for i in range(len(yearly_data)):
                for j in range(i + 1, len(yearly_data)):
                    sim = cosine_similarity([yearly_data.iloc[i]["Vector"]], [yearly_data.iloc[j]["Vector"]])[0][0]
                    results.append({
                        "Symbol1": yearly_data.iloc[i]["Symbol"],
                        "Symbol2": yearly_data.iloc[j]["Symbol"],
                        "Year": year,
                        "Similarity": sim
                    })
            sim_df = pd.DataFrame(results)

            sim_mx = panel_to_matrix(sim_df)

            sim_mx.to_csv(r"output\bus_sim\bus_sim" + str(year) + ".csv", index=False)
            logging.info(f"Business similarity matrix for year {year} saved to output folder.")
    except Exception as e:
        logging.error(f"Error in bus_sim_gen: {e}")
        raise


def bus_sim_discr():
    try:
        logging.info("Business similarity distribution analysis started.")
        folder_path = r'output\bus_sim'

        result_data = []
        for filename in tqdm(os.listdir(folder_path)):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                firm_count = len(df)
                matrix = df.to_numpy()
                lower_tri_list = []
                for i in range(1, firm_count):
                    for j in range(0, i):
                        lower_tri_list.append(matrix[i][j])

                sim = pd.DataFrame()

                sim['sim'] = lower_tri_list

                desc_stats = sim['sim'].describe()
                count_gt_0 = (sim['sim'] > 0).sum()
                count_gt_0_1 = (sim['sim'] > 0.5).sum()

                result_data.append({
                    'Filename': filename,
                    '#firm': firm_count,
                    'Count (> 0)': count_gt_0,
                    'Count (> 0.5)': count_gt_0_1,
                    **desc_stats
                })
        result_df = pd.DataFrame(result_data)

        result_csv_path = r'output\bus_sim_discr.csv'
        result_df.to_csv(result_csv_path, index=False)

        logging.info(f"Business similarity distribution analysis saved to {result_csv_path}.")
    except Exception as e:
        logging.error(f"Error in bus_sim_discr: {e}")
        raise


if __name__ == '__main__':
    start_time = datetime.now()
    logging.info("Starting the business similarity calculation process.")

    try:
        data_process()
        bus_sim_gen()
        bus_sim_discr()
    except Exception as e:
        logging.error(f"Script failed with error: {e}")

    end_time = datetime.now()
    logging.info(f"Script execution completed. Total time: {end_time - start_time}.")