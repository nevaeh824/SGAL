import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
import logging
import warnings
warnings.filterwarnings("ignore")

pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_rows', 100)

logging.basicConfig(
    filename='log/factors_gen.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("Script execution started.")

def fin_fac():
    logging.info("Starting the financial factor generation process")

    try:
        # Load data
        logging.info("Loading balance sheet data")
        df1 = pd.read_csv("data\\CSMAR_data\资产负债表092923528(仅供vip使用)\FS_Combas.csv", usecols=['Stkcd', 'Accper','Typrep','A002000000','A001000000','A003000000','A001121000','A002120000','A001205000','A001123000','A001100000','A002100000','A001212000'])
        df1 = df1[df1['Typrep']=="A"]
        logging.info("Balance sheet data loaded successfully")

        logging.info("Loading income statement data")
        df2 = pd.read_csv("data\\CSMAR_data\利润表104957568(仅供vip使用)\FS_Comins.csv",usecols=['Stkcd', 'Accper','Typrep','B002000000','B001101000','B001100000'])
        df2 = df2[df2['Typrep'] == "A"]
        logging.info("Income statement data loaded successfully")

        logging.info("Loading cash flow data")
        df3 = pd.read_csv("data\\CSMAR_data\现金流量表(直接法)104959934(仅供vip使用)\FS_Comscfd.csv",usecols=['Stkcd', 'Accper','Typrep','C001000000','C002006000'])
        df3 = df3[df3['Typrep'] == "A"]
        logging.info("Cash flow data loaded successfully")

        df4 = pd.read_csv("data\\CSMAR_data\海外业务收入表133126116(仅供vip使用)\OFDI_OPERATEINCOME.csv",
                          usecols=['Symbol', 'EndDate', 'EarningsProportion', 'VestingPeriod'])
        df4.rename(columns={'Symbol':'Stkcd','EndDate':'Accper'},inplace=True)
        df4 = df4[df4['VestingPeriod'] == 1]

        # Convert Accper to datetime
        logging.info("Converting Accper to datetime format")
        df1['Accper'] = pd.to_datetime(df1['Accper'], format='%Y-%m-%d')
        df2['Accper'] = pd.to_datetime(df2['Accper'], format='%Y-%m-%d')
        df3['Accper'] = pd.to_datetime(df3['Accper'], format='%Y-%m-%d')
        df4['Accper'] = pd.to_datetime(df4['Accper'], format='%Y-%m-%d')

        # Helper function for end date based on frequency
        def state_end_date(freq_type):
            if freq_type == 'Y':
                start_date = pd.to_datetime('2002-01-01')
                end_date = pd.to_datetime('2023-01-01')
            if freq_type == 'Q1':
                start_date = pd.to_datetime('2001-03-31')
                end_date = pd.to_datetime('2022-03-31')
            if freq_type == 'Q2':
                start_date = pd.to_datetime('2001-06-30')
                end_date = pd.to_datetime('2022-06-30')
            if freq_type == 'Q3':
                start_date = pd.to_datetime('2001-09-30')
                end_date = pd.to_datetime('2022-09-30')
            if freq_type == 'Q4':
                start_date = pd.to_datetime('2001-12-31')
                end_date = pd.to_datetime('2022-12-31')

            return start_date, end_date

        df_merge = pd.DataFrame()
        for freq_type in ['Y','Q1','Q2','Q3','Q4']:
            logging.info(f"Processing frequency: {freq_type}")

            start_date, end_date = state_end_date(freq_type)
            df1_sub = df1[(df1['Accper'].dt.year >= start_date.year) & (df1['Accper'].dt.year <= end_date.year) & (
                        df1['Accper'].dt.month == start_date.month) & (df1['Accper'].dt.day == start_date.day)]
            df2_sub = df2[(df2['Accper'].dt.year >= start_date.year) & (df2['Accper'].dt.year <= end_date.year) & (
                        df2['Accper'].dt.month == start_date.month) & (df2['Accper'].dt.day == start_date.day)]
            df3_sub = df3[(df3['Accper'].dt.year >= start_date.year) & (df3['Accper'].dt.year <= end_date.year) & (
                        df3['Accper'].dt.month == start_date.month) & (df3['Accper'].dt.day == start_date.day)]
            df4_sub = df4[(df4['Accper'].dt.year >= start_date.year) & (df4['Accper'].dt.year <= end_date.year) & (
                    df4['Accper'].dt.month == start_date.month) & (df4['Accper'].dt.day == start_date.day)]

            # Merge dataframes
            logging.info("Merging dataframes")
            df = pd.merge(df1_sub, df2_sub, on=['Stkcd', 'Accper'], how = 'left')
            df = pd.merge(df, df3_sub, on=['Stkcd', 'Accper'], how = 'left')
            df = pd.merge(df, df4_sub, on=['Stkcd', 'Accper'], how='left')

            df = df.loc[:, ~df.columns.duplicated()]

            # Calculate financial variables
            logging.info("Calculating financial variables")
            vars = ['Lev', 'FCFR', 'ROA', 'ROE', '2yLoss', 'NCFO', '3yGrw', 'RTR', 'EIR', 'RIR', 'FSR', 'RR', 'IR', 'CAR', 'FAT']
            vars_new = [ var + '_'+ freq_type for var in vars]

            df[vars_new[0]] = df['A002000000'] / df['A001000000']
            df[vars_new[1]] = (df['C001000000'] - df['C002006000']) / df['A001000000']
            df[vars_new[2]] = df['B002000000'] / df['A001000000']
            df[vars_new[3]] = df['B002000000'] / df['A003000000']
            df[vars_new[4]] = df.groupby('Stkcd')['B002000000'].transform(lambda x: (x < 0).rolling(2).sum().eq(2).astype(int))
            df[vars_new[5]] = df.groupby('Stkcd')['C001000000'].transform(lambda x: (x < 0).rolling(2).sum().eq(2).astype(int))
            df[vars_new[6]] = df.groupby('Stkcd')['B001101000'].transform(lambda x: x.pct_change(periods=2))
            df[vars_new[7]] = (df['A001121000'] + df['A002120000']) / df['A001000000']
            df[vars_new[8]] = df['A001205000'] / df['A003000000']
            df[vars_new[9]] = df['C001000000'] / df['C002006000']
            df[vars_new[10]] = df['EarningsProportion']
            df[vars_new[11]] = df['A001121000'] / df['B001101000']
            df[vars_new[12]] = df['A001123000'] / df['B001101000']
            df[vars_new[13]] = df['A001100000'] / df['A002100000']
            df[vars_new[14]] = df['B001101000'] / df['A001212000']

            df = df[['Stkcd', 'Accper'] + vars_new]
            df = df.rename(columns={'Stkcd': 'Symbol', 'Accper': 'EndDate'})

            # Adjust year for frequency type
            if freq_type == 'Y':
                df['EndDate'] = df['EndDate'].dt.year - 1
            else:
                df['EndDate'] = df['EndDate'].dt.year

            # Load peer data
            logging.info("Loading peer data")
            df4 = pd.read_csv(r"output\peer_identified.csv",
                              usecols=['Symbol', 'Year', 'PeerID'])
            df4.rename(columns={'Year':'EndDate'},inplace=True)
            df = pd.merge(df, df4, on=['Symbol', 'EndDate'], how = 'left')

            # Calculate peer-based variables
            logging.info("Calculating peer-based variables")
            for var in vars_new:
                df[var + '_' + 'peer'] = df[var] - df.groupby('PeerID')[var].transform('mean')
            df = df.drop('PeerID',axis=1)

            # Calculate time-based variables
            logging.info("Calculating time-based variables")
            for var in vars_new:
                df[var + '_' + 'time'] = df[var].transform(lambda x: x.diff(periods=1))
            if df_merge.empty == False:
                df_merge = pd.merge(df_merge, df, on = ['Symbol', 'EndDate'], how = 'left')
            else:
                df_merge = df

        logging.info("Financial factor generation completed successfully")
        return df_merge

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise e

def gov_fac():
    try:
        logging.info("Starting to load governance-related data...")

        logging.info("Loading governance ability data (DF1)...")
        df1 = pd.read_csv(
            r"data\CSMAR_data\管理层治理能力112434566(仅供vip使用)\BDT_ManaGovAbil.csv",
            usecols=[
                'Symbol', 'Enddate', 'ContrshrProportion', 'Mngmhldn',
                'InsInvestorProp', 'Boardsize', 'IndDirectorRatio',
                'ContrshrNature', 'ConcurrentPosition'
            ]
        )
        df1 = df1.rename(columns={'Enddate': 'EndDate'})
        logging.info(f"Governance ability data loaded successfully, rows: {len(df1)}")

        logging.info("Loading audit firm data (DF2)...")
        df2 = pd.read_csv(
            r"data\CSMAR_data\上市公司审计机构列表113358650(仅供vip使用)\AR_LISTCOMPAUDIT.csv",
            usecols=[
                'Symbol', 'EndDate', 'Source', 'TerritoryForeignIdentity',
                'IsAuditCommitteeSetUp', 'IsChangeFirm', 'AccountingFirmName'
            ]
        )
        df2 = df2[(df2['Source'] == 0) & (df2['TerritoryForeignIdentity'] == 1)]
        logging.info(f"Audit firm data loaded successfully, rows: {len(df2)}")

        logging.info("Loading chairman and manager changes data (DF3)...")
        df3 = pd.read_csv(
            r"data\CSMAR_data\董事长与总经理变更文件113922166(仅供vip使用)\CG_Ceo.csv",
            usecols=['Stkcd', 'Annodt', 'Position']
        )
        df3 = df3.rename(columns={'Stkcd': 'Symbol', 'Annodt': 'EndDate'})
        logging.info(f"Chairman and manager changes data loaded successfully, rows: {len(df3)}")

        logging.info("Processing date formats...")
        df1['EndDate'] = pd.to_datetime(df1['EndDate'], format='%Y-%m-%d')
        df2['EndDate'] = pd.to_datetime(df2['EndDate'], format='%Y-%m-%d')
        df3['EndDate'] = pd.to_datetime(df3['EndDate'], format='%Y-%m-%d')
        df3['EndDate'] = df3['EndDate'].apply(lambda x: x.replace(month=12, day=31))
        logging.info("Date formats processed successfully.")

        logging.info("Filtering data by date range...")
        start_date = pd.to_datetime('2001-12-31')
        end_date = pd.to_datetime('2022-12-31')
        df1 = df1[(df1['EndDate'].dt.year >= start_date.year) & (df1['EndDate'].dt.year <= end_date.year)]
        df2 = df2[(df2['EndDate'].dt.year >= start_date.year) & (df2['EndDate'].dt.year <= end_date.year)]
        df3 = df3[(df3['EndDate'].dt.year >= start_date.year) & (df3['EndDate'].dt.year <= end_date.year)]
        logging.info("Date filtering completed.")

        logging.info("Processing chairman and manager positions...")
        df3['PositionChair'] = df3['Position'].apply(lambda x: 1 if x == 1 else 0)
        df3['PositionManag'] = df3['Position'].apply(lambda x: 1 if x == 2 else 0)
        df3 = df3.groupby(['Symbol', 'EndDate']).agg({'PositionChair': 'sum', 'PositionManag': 'sum'}).reset_index()
        df3['PositionChair'] = df3['PositionChair'].apply(lambda x: 1 if x > 0 else 0)
        df3['PositionManag'] = df3['PositionManag'].apply(lambda x: 1 if x > 0 else 0)
        logging.info("Position processing completed.")

        logging.info("Merging dataframes...")
        df = pd.merge(df1, df2, on=['Symbol', 'EndDate'])
        df = pd.merge(df, df3, on=['Symbol', 'EndDate'], how='left')
        df = df.loc[:, ~df.columns.duplicated()]

        logging.info("Filling missing values...")
        df['PositionChair'].fillna(0, inplace=True)
        df['PositionManag'].fillna(0, inplace=True)

        logging.info("Renaming and calculating columns...")
        df['TopHold'] = df['ContrshrProportion']
        df = df.rename(columns={
            'Mngmhldn': 'MHR',
            'InsInvestorProp': 'IHR',
            'Boardsize': 'Board',
            'IndDirectorRatio': 'IndDrc',
            'ContrshrNature': 'SOE',
            'ConcurrentPosition': 'Dual',
            'IsAuditCommitteeSetUp': '3yIAC',
            'IsChangeFirm': '3yAFC',
            'PositionManag': '3yMC',
            'PositionChair': '3yBC'
        })
        df['Top4Aud'] = df['AccountingFirmName'].apply(
            lambda x: 1 if '普华永道' in x or '毕马威' in x or '德勤' in x or '安永' in x else 0
        )
        vars = ['TopHold', 'MHR', 'IHR', 'Board', 'IndDrc', 'SOE', 'Dual', '3yIAC', '3yAFC', 'Top4Aud', '3yMC', '3yBC']

        df = df[['Symbol', 'EndDate'] + vars]
        df['EndDate'] = df['EndDate'].dt.year

        logging.info("Processing peer data...")
        df4 = pd.read_csv(r"output\peer_identified.csv", usecols=['Symbol', 'Year', 'PeerID'])
        df4.rename(columns={'Year': 'EndDate'}, inplace=True)
        df = pd.merge(df, df4, on=['Symbol', 'EndDate'], how='left')

        for var in vars:
            df[var + '_peer'] = df[var] - df.groupby(['EndDate', 'PeerID'])[var].transform('mean')
        df = df.drop('PeerID', axis=1)

        logging.info("Calculating time differences...")
        for var in vars:
            df[var + '_time'] = df[var].transform(lambda x: x.diff(periods=1))

        logging.info("Governance data processing completed successfully.")
        return df

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

def mrkt_fac():
    try:
        logging.info("Starting to load market-related data...")

        # Load weekly stock return data
        logging.info("Loading weekly stock return data (DF1)...")
        df1 = pd.DataFrame()
        for i in range(4):
            file_path = (
                r"data\CSMAR_data\周个股回报率文件222514782(仅供vip使用)\TRD_Week.csv"
                if i == 0
                else rf"data\CSMAR_data\周个股回报率文件222514782(仅供vip使用)\TRD_Week{i}.csv"
            )
            logging.info(f"Loading file: {file_path}")
            df1_sub = pd.read_csv(
                file_path,
                usecols=['Stkcd', 'Trdwnt', 'Wnvaltrd', 'Wsmvosd', 'Wsmvttl', 'Wretwd']
            )
            df1 = pd.concat([df1, df1_sub], axis=0) if not df1.empty else df1_sub
        logging.info(f"Weekly stock return data loaded successfully, rows: {len(df1)}")

        # Process DF1
        logging.info("Processing weekly stock return data...")
        df1['Trdwnt'] = pd.to_datetime(df1['Trdwnt'] + '-0', format='%Y-%W-%w').dt.strftime('%Y-%m-%d')
        df1 = df1.rename(columns={'Stkcd': 'Symbol', 'Trdwnt': 'EndDateW'})
        df1['EndDateW'] = pd.to_datetime(df1['EndDateW'])
        df1['EndDate'] = df1['EndDateW'] + pd.offsets.QuarterEnd(0)

        # Load market return data
        logging.info("Loading market return data (DF2)...")
        df2 = pd.read_csv(
            r"data\CSMAR_data\综合周市场回报率文件125349410(仅供vip使用)\TRD_Weekcm.csv",
            usecols=['Markettype', 'Trdwnt', 'Cwretwdeq']
        )
        df2 = df2[df2['Markettype'] == 21]
        df2['Trdwnt'] = pd.to_datetime(df2['Trdwnt'] + '-0', format='%Y-%W-%w').dt.strftime('%Y-%m-%d')
        df2 = df2.rename(columns={'Trdwnt': 'EndDateW'})
        df2['EndDateW'] = pd.to_datetime(df2['EndDateW'])
        df2['EndDate'] = df2['EndDateW'] + pd.offsets.QuarterEnd(0)
        logging.info(f"Market return data loaded successfully, rows: {len(df2)}")

        # Load balance sheet data
        logging.info("Loading balance sheet data (DF3)...")
        df3 = pd.read_csv(
            r"data\CSMAR_data\资产负债表092923528(仅供vip使用)\FS_Combas.csv",
            usecols=['Stkcd', 'Accper', 'Typrep', 'A003000000']
        )
        df3 = df3[df3['Typrep'] == "A"]
        df3 = df3.rename(columns={'Stkcd': 'Symbol', 'Accper': 'EndDate'})
        df3['EndDate'] = pd.to_datetime(df3['EndDate'])
        df3 = df3[df3['EndDate'].dt.month != 1]
        logging.info(f"Balance sheet data loaded successfully, rows: {len(df3)}")

        # Load income statement data
        logging.info("Loading income statement data (DF4)...")
        df4 = pd.read_csv(
            r"data\CSMAR_data\利润表104957568(仅供vip使用)\FS_Comins.csv",
            usecols=['Stkcd', 'Accper', 'Typrep', 'B002000000']
        )
        df4 = df4[df4['Typrep'] == "A"]
        df4 = df4.rename(columns={'Stkcd': 'Symbol', 'Accper': 'EndDate'})
        df4['EndDate'] = pd.to_datetime(df4['EndDate'])
        df4 = df4[df4['EndDate'].dt.month != 1]
        logging.info(f"Income statement data loaded successfully, rows: {len(df4)}")

        df5 = pd.read_csv(r"data\pricing_model.csv",usecols=['Symbol','EndDate','FF3','Liu3'])
        df5['EndDate'] = pd.to_datetime(df5['EndDate'])

        # Merging dataframes
        logging.info("Merging all dataframes...")
        df_merge = pd.merge(df1, df2, on=['EndDateW'], how='left')
        df_merge = df_merge.rename(columns={'EndDate_x': 'EndDate'})
        df_merge = pd.merge(df_merge, df3, on=['Symbol', 'EndDate'])
        df_merge = pd.merge(df_merge, df4, on=['Symbol', 'EndDate'])
        df_merge = pd.merge(df_merge, df5, on=['Symbol', 'EndDate'])
        df_merge = df_merge[(df_merge['EndDate'].dt.year >= 2001) & (df_merge['EndDate'].dt.year <= 2022)]
        logging.info("Merging completed.")

        # Calculate factors
        logging.info("Calculating market factors...")
        df_final = pd.DataFrame()
        vars = ['Rtn', 'Vol', 'Turnover', 'PE', 'BM', 'Beta','FF3','Liu3']
        for freq_type in ['Y', 'Q1', 'Q2', 'Q3', 'Q4']:
            logging.info(f"Calculating factors for frequency type: {freq_type}")
            if freq_type == 'Y':
                df_merge_grouped = df_merge.groupby(['Symbol', df_merge['EndDate'].dt.year])
            else:
                df_merge_sub = df_merge[df_merge['EndDate'].dt.quarter == int(freq_type[-1])]
                df_merge_grouped = df_merge_sub.groupby(['Symbol', df_merge_sub['EndDate'].dt.year])

            df = pd.DataFrame()
            df['Rtn' + '_' + freq_type] = df_merge_grouped['Wretwd'].sum()
            df['Vol' + '_' + freq_type] = df_merge_grouped['Wretwd'].std()
            df['Turnover' + '_' + freq_type] = df_merge_grouped['Wnvaltrd'].mean() / df_merge_grouped['Wsmvttl'].mean()
            if freq_type == 'Y':
                df['PE' + '_' + freq_type] = df_merge_grouped['Wsmvttl'].last() * 1000 / (df_merge_grouped['B002000000'].mean() * 4)
            else:
                df['PE' + '_' + freq_type] = df_merge_grouped['Wsmvttl'].last() * 1000 / df_merge_grouped['B002000000'].mean()
            df['BM' + '_' + freq_type] = df_merge_grouped['A003000000'].last() / df_merge_grouped['Wsmvttl'].last()
            df['Beta' + '_' + freq_type] = df_merge_grouped[['Wretwd', 'Cwretwdeq']].corr().iloc[0::2, -1].reset_index(level=2, drop=True)
            df['FF3' + '_' + freq_type] = df_merge_grouped['FF3'].mean()
            df['Liu3' + '_' + freq_type] = df_merge_grouped['Liu3'].mean()

            df.reset_index(inplace=True)

            # Adding peer adjustments
            df5 = pd.read_csv(r"output\peer_identified.csv", usecols=['Symbol', 'Year', 'PeerID'])
            df5.rename(columns={'Year': 'EndDate'}, inplace=True)
            df = pd.merge(df, df5, on=['Symbol', 'EndDate'], how='left')
            vars_new = [var + '_' + freq_type for var in vars]
            for var in vars_new:
                df[var + '_peer'] = df[var] - df.groupby(['EndDate', 'PeerID'])[var].transform('mean')
            df = df.drop('PeerID', axis=1)

            for var in vars_new:
                df[var + '_time'] = df[var].transform(lambda x: x.diff(periods=1))

            if not df_final.empty:
                df_final = pd.merge(df_final, df, on=['Symbol', 'EndDate'], how='left')
            else:
                df_final = df

        logging.info("Market factor calculation completed successfully.")
        return df_final

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

def sent_fac():
    try:
        logging.info("Processing NEWS_REPORT data")
        df1 = pd.read_csv(r"data\Datago_data\NEWS_REPORT.csv",
                          usecols=['stkcd', 'pub_date', 'content_senti_score_ref'])
        df1 = df1.rename(columns={'stkcd': 'Symbol', 'pub_date': 'EndDate'})
        df1['EndDate'] = pd.to_datetime(df1['EndDate'])
        df1 = df1[df1['content_senti_score_ref'] < 0].groupby(['Symbol', 'EndDate']).size().reset_index(name='NewsSent')
        logging.info("Finished processing NEWS_REPORT data")

        logging.info("Processing ANALYST_REPORT data")
        df2 = pd.read_csv(r"data\Datago_data\ANALYST_REPORT.csv",
                          usecols=['org_id', 'rpt_date', 'senti_score_div'])
        df2 = df2.rename(columns={'org_id': 'Symbol', 'rpt_date': 'EndDate'})
        df2['EndDate'] = pd.to_datetime(df2['EndDate'])
        df2 = df2[df2['senti_score_div'] < 0].groupby(['Symbol', 'EndDate']).size().reset_index(name='RepoSent')
        logging.info("Finished processing ANALYST_REPORT data")

        logging.info("Processing GUBA_POST data")
        df3 = pd.read_csv(r"data\Datago_data\GUBA_POST.csv",
                          usecols=['Stock_Id', 'Pub_Date', 'Posts_Neg_Regi_Sum'])
        df3 = df3.rename(columns={'Stock_Id': 'Symbol', 'Pub_Date': 'EndDate'})
        df3['EndDate'] = pd.to_datetime(df3['EndDate'])
        df3 = df3.groupby(['Symbol', 'EndDate'])['Posts_Neg_Regi_Sum'].sum().reset_index(name='GubaSent')
        logging.info("Finished processing GUBA_POST data")

        logging.info("Merging datasets")
        df1 = pd.merge(df1, df2, on=['Symbol', 'EndDate'], how='outer')
        df1 = pd.merge(df1, df3, on=['Symbol', 'EndDate'], how='outer')
        df1 = df1.loc[:, ~df1.columns.duplicated()]
        df1 = df1[(df1['EndDate'].dt.year >= 2001) & (df1['EndDate'].dt.year <= 2022)]
        logging.info("Datasets merged successfully")

        logging.info("Generating grouped data")
        df_merge = pd.DataFrame()
        vars = ['NewsSent', 'RepoSent', 'GubaSent']
        for freq_type in ['Y', 'Q1', 'Q2', 'Q3', 'Q4']:
            logging.info(f"Processing frequency type: {freq_type}")
            if freq_type == 'Y':
                df1_grouped = df1.groupby(['Symbol', df1['EndDate'].dt.year])
            else:
                df1_sub = df1[df1['EndDate'].dt.quarter == int(freq_type[-1])]
                df1_grouped = df1_sub.groupby(['Symbol', df1_sub['EndDate'].dt.year])

            df = df1_grouped.agg({'NewsSent': 'sum', 'RepoSent': 'sum', 'GubaSent': 'sum'}).reset_index()

            logging.info("Handling missing years")
            result_df = pd.DataFrame(columns=['Symbol', 'EndDate', 'NewsSent', 'RepoSent', 'GubaSent'])
            years = range(2001, 2023)
            for stock_code in df['Symbol'].unique():
                stock_data = df[df['Symbol'] == stock_code]
                missing_years = set(years) - set(stock_data['EndDate'])
                missing_data = pd.DataFrame({'Symbol': stock_code, 'EndDate': list(missing_years),
                                             'NewsSent': 0, 'RepoSent': 0, 'GubaSent': 0})
                result_df = pd.concat([result_df, missing_data, stock_data])

            result_df = result_df.sort_values(by=['Symbol', 'EndDate'], ascending=True)
            result_df['NewsSent'] = np.log(result_df['NewsSent'].astype('float') + 1)
            result_df['RepoSent'] = np.log(result_df['RepoSent'].astype('float') + 1)
            result_df['GubaSent'] = np.log(result_df['GubaSent'].astype('float') + 1)
            df = result_df
            df.columns = ['Symbol', 'EndDate',
                          'NewsSent' + '_' + freq_type,
                          'RepoSent' + '_' + freq_type,
                          'GubaSent' + '_' + freq_type]

            logging.info("Processing peer comparison data")
            df4 = pd.read_csv(r"output\peer_identified.csv",
                              usecols=['Symbol', 'Year', 'PeerID'])
            df4.rename(columns={'Year': 'EndDate'}, inplace=True)
            df = pd.merge(df, df4, on=['Symbol', 'EndDate'], how='left')
            vars_new = [var + '_' + freq_type for var in vars]
            for var in vars_new:
                df[var + '_' + 'peer'] = df[var] - df.groupby(['EndDate', 'PeerID'])[var].transform('mean')
            df = df.drop('PeerID', axis=1)

            logging.info("Calculating time-series changes")
            for var in vars_new:
                df[var + '_' + 'time'] = df[var].transform(lambda x: x.diff(periods=1))

            if not df_merge.empty:
                df_merge = pd.merge(df_merge, df, on=['Symbol', 'EndDate'], how='right')
            else:
                df_merge = df

        logging.info("sent_fac function completed successfully")
        return df_merge
    except Exception as e:
        logging.error(f"An error occurred in the sent_fac function: {e}")
        raise

def Label():
    try:
        logging.info("Starting Label function")

        # Process FSFP
        logging.info("Reading and processing FSFP data")
        df1 = pd.read_csv(r"data\FSFP.csv", usecols=['Symbol', 'Year', 'FSFP'])
        df1.rename(columns={'Year': 'EndDate'}, inplace=True)
        df1['FSFP_2c'] = df1['FSFP'].apply(lambda x: 1 if x > 0 else 0)
        median_nonzero = df1[df1['FSFP'] != 0]['FSFP'].median()
        df1['FSFP_3c'] = df1['FSFP'].apply(lambda x: 2 if x > median_nonzero else 0 if x == 0 else 1)
        logging.info("Finished processing FSFP data")

        # Process Regulatory Penalty
        logging.info("Reading and processing regulatory penalty data")
        df2 = pd.read_csv(
            r"data\CSMAR_data\违规信息总表101117679(仅供vip使用)\STK_Violation_Main.csv",
            usecols=['ViolationID', 'Symbol', 'ViolationYear', 'ViolationTypeID', 'Promulgator']
        )
        patterns = ['P2501', 'P2502', 'P2503', 'P2506']
        df2 = df2[df2['ViolationTypeID'].apply(lambda x: any(pattern in str(x) for pattern in patterns))]
        df2 = df2[df2['Promulgator'].notnull() & df2['Promulgator'].str.contains('证监会')]

        index = pd.MultiIndex.from_product(
            [df1['Symbol'].unique(), pd.date_range(start='2001-01-01', end='2022-12-31', freq='A-DEC').year],
            names=['Symbol', 'EndDate']
        )
        result_df = pd.DataFrame(0, index=index, columns=['Penalty'])

        df2['ViolationYear'] = df2['ViolationYear'].apply(lambda x: str(x).split(';'))

        for idx, row in df2.iterrows():
            subject_name = row['Symbol']
            violation_years = row['ViolationYear']
            for year in violation_years:
                try:
                    year = int(year)
                    if (subject_name, year) in result_df.index:
                        result_df.at[(subject_name, year), 'Penalty'] = 1
                except ValueError:
                    logging.warning(f"Invalid year: {year} in row {idx}")
                    continue
        df2 = result_df.sort_index().reset_index()
        logging.info("Finished processing regulatory penalty data")

        # Process Restatement
        logging.info("Reading and processing restatement data")
        df3 = pd.read_csv(
            r"data\CSMAR_data\上市公司财务重述情况表(2001-2021)160220256(仅供vip使用)\AIQ_LCFinRestatY.csv",
            usecols=['Symbol', 'RestatementYear', 'RestatementObject', 'RestatementType', 'IsFinancialDataChange']
        )
        df3 = df3[df3['RestatementObject'] == 1]
        df3 = df3[df3['RestatementType'].isin([1, 2, 3, 4, 5, 8, 9, 10, 13])]
        df3['EndDate'] = pd.to_datetime(df3['RestatementYear']).dt.year
        df3 = df3[(df3['EndDate'] >= 2001) & (df3['EndDate'] <= 2022)]
        df3['Restate'] = 1
        df3 = df3.groupby(['Symbol', 'EndDate'])['Restate'].sum().reset_index()
        df3['Restate'] = 1
        df3 = df3[['Symbol', 'EndDate', 'Restate']].sort_values(['Symbol', 'EndDate'])
        logging.info("Finished processing restatement data")

        # Process Non-standard audit opinion
        logging.info("Reading and processing non-standard audit opinion data")
        df4 = pd.read_csv(
            r"data\CSMAR_data\审计意见表文件085948094(仅供vip使用)\FIN_Audit.csv",
            usecols=['Stkcd', 'Accper', 'Audittyp']
        )
        df4 = df4.rename(columns={'Stkcd': 'Symbol', 'Accper': 'EndDate'})
        df4['EndDate'] = pd.to_datetime(df4['EndDate'])
        df4 = df4[df4['EndDate'].dt.month == 12]
        df4['EndDate'] = df4['EndDate'].dt.year
        df4 = df4[(df4['EndDate'] >= 2001) & (df4['EndDate'] <= 2022)]
        df4 = df4[~(df4['Audittyp'] == "标准无保留意见")]
        df4['NsAudit'] = 1
        df4 = df4.groupby(['Symbol', 'EndDate'])['NsAudit'].sum().reset_index()
        df4['NsAudit'] = 1
        logging.info("Finished processing non-standard audit opinion data")

        # Merge all datasets
        logging.info("Merging all datasets")
        df = pd.merge(df1, df2, on=['Symbol', 'EndDate'], how='left')
        df = pd.merge(df, df3, on=['Symbol', 'EndDate'], how='left')
        df = pd.merge(df, df4, on=['Symbol', 'EndDate'], how='left')
        df['Restate'] = df['Restate'].fillna(0)
        df['NsAudit'] = df['NsAudit'].fillna(0)
        df = df[['Symbol', 'EndDate', 'FSFP_2c', 'FSFP_3c', 'FSFP', 'Penalty', 'Restate', 'NsAudit']]

        # Add peer comparison
        logging.info("Adding peer comparison")
        df5 = pd.read_csv(r"output\peer_identified.csv", usecols=['Symbol', 'EndDate', 'PeerID'])
        df5['EndDate'] = pd.to_datetime(df5['EndDate']).dt.year
        df = pd.merge(df, df5, on=['Symbol', 'EndDate'], how='left')
        vars = ['FSFP', 'Penalty', 'Restate', 'NsAudit']
        for var in vars:
            df[var + '_peer'] = (df[var] - df.groupby(['EndDate', 'PeerID'])[var].transform('mean')).shift(1)
        df = df.drop('PeerID', axis=1)

        return df

    except Exception as e:
        logging.error(f"An error occurred in the Label function: {e}")
        raise

if __name__ == '__main__':
    try:
        logging.info("Starting to generate model input")

        # Generate financial factors
        logging.info("Generating financial factors")
        df1 = fin_fac()
        logging.info("Financial factors have been generated")

        # Generate governance factors
        logging.info("Generating governance factors")
        df2 = gov_fac()
        logging.info("Governance factors have been generated")

        # Generate market factors
        logging.info("Generating market factors")
        df3 = mrkt_fac()
        logging.info("Market factors have been generated")

        # Generate sentiment factors
        logging.info("Generating sentiment factors")
        df4 = sent_fac()
        logging.info("Sentiment factors have been generated")

        # Generate labels
        logging.info("Generating labels")
        df5 = Label()
        logging.info("Labels have been generated")

        # Merging all dataframes
        logging.info("Merging all factors and labels into a single dataframe")
        df = pd.merge(df1, df2, on=['Symbol', 'EndDate'], how='left')
        df = pd.merge(df, df3, on=['Symbol', 'EndDate'], how='left')
        df = pd.merge(df, df4, on=['Symbol', 'EndDate'], how='left')
        df = pd.merge(df, df5, on=['Symbol', 'EndDate'], how='left')

        # Renaming and setting index
        logging.info("Renaming EndDate to ReportYear and setting index")
        df = df.rename(columns={'EndDate': 'Year'})
        df.set_index(['Symbol', 'Year'], inplace=True)

        # Winsorizing columns
        logging.info("Winsorizing dataframe columns")
        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
                df[column] = winsorize(df[column].replace([np.inf, -np.inf], np.nan).fillna(0), limits=(0.01, 0.01))
            except Exception as e:
                logging.warning(f"Error processing column {column}: {e}")

        # Saving the final dataframe
        output_file = 'data/model_input.csv'
        df.to_csv(output_file)
        logging.info(f"Model input successfully saved to {output_file}")

    except Exception as e:
        logging.error(f"An error occurred during model input generation: {e}")
        raise