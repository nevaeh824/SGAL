import pandas as pd

def audit_fee_clac():
    df = pd.read_csv(r"data\CSMAR_data\审计意见表文件085948094(仅供vip使用)\FIN_Audit.csv",usecols=['Stkcd','Accper','Tcost'])
    df.rename(columns={'Stkcd':'Symbol','Accper':'EndDate','Tcost':'AuditFee'})
    df['EndDate'] = pd.to_datetime(df['EndDate'] )
    df['Symbol'] = df['Symbol'].astype(str)
    df['AuditFee'] = df['AuditFee'].astype(int)
    df['MV'] = df['MV'].astype(int)

    df2 = pd.read_csv(r"data\CSMAR_data\上市公司基本信息年度表091142326(仅供vip使用)\STK_LISTEDCOINFOANL.csv", usecols=['Symbol', 'EndDate', 'IndustryCodeC']) # This file is downloaded from CSMAR
    df2['EndDate'] = pd.to_datetime(df2['EndDate'])
    df2['Symbol'] = df2['Symbol'].astype(str)
    df = df2.merge(df,on=['Symbol','EndDate'], how='left')

    def process_column(value):
        if value.startswith('C'):
            return value
        else:
            return value[0]
    df['IndustryCodeC'] = df['IndustryCodeC'].apply(process_column)
    df['Year'] = pd.to_datetime(df['EndDate']).dt.year
    df = df[(df['Year'] >= 2001) & (df['Year'] <= 2022)]
    df=df[df['AuditFee'] > 0]
    df_indus = df.groupby(['IndustryCodeC','Year']).aggregate({'AuditFee':'mean','MV':'mean'}).reset_index()
    df_indus.to_csv(r'data\audit_fee_indus.csv')

    grouped = df_indus.groupby('Year')['AuditFee'].agg(['mean', lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)])
    grouped.columns = ['Mean', '5% percentile', '95% percentile']
    df_year = grouped.reset_index()
    df_year.to_csv(r'data\audit_fee_mean.csv')

def eco_value():
    df = pd.read_csv("output\detetion_results.csv",
                usecols=['Symbol', 'Year','IndustryCodeC', 'PredictFraud', 'RealFraud', 'PriceChg', 'ShareNum'])

    df_auditfee = pd.read_csv(r'data\audit_fee_indus.csv',
                            usecols=['IndustryCodeC','Year','AuditFee'])

    df = pd.merge(df, df_auditfee, on = ['IndustryCodeC','Year'], how='left')

    df['detection_value'] = -df['PredictFraud'] * df['RealFraud'] * df['PriceChg'] * df['ShareNum']
    df['fail_detect_cost'] = -(1 - df['PredictFraud']) * df['RealFraud'] * df['PriceChg'] * df['ShareNum']
    df['audit_cost'] = df['PredictFraud'] * df['AuditFee']
    df['net_value'] = df['detection_value']-df['fail_detect_cost']-df['audit_cost']

    df_year = df.groupby(['Year']).agg({'detection_value': 'sum',
                              'fail_detect_cost': 'sum',
                              'audit_cost':'sum',
                              'net_value':'sum'}).reset_index()

    df_year.to_csv(r'output\enco_value.csv',index=False)


if __name__ == '__main__':
    audit_fee_clac()
    eco_value()