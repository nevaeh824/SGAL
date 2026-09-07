import pandas as pd
import numpy as np
from pathlib import Path
import re
import pickle
import os

def load_and_prepare_fsfp_data(fsfp_file, value_column="fsfp_original"):
    """
    加载和准备FSFP数据
    """
    # 加载FSFP数据
    fsfp_df = pd.read_csv(fsfp_file)
    print(f"加载FSFP数据，共{len(fsfp_df)}行")

    # 兼容不同批次文件的列名：0509文件使用 code/week，旧版脚本使用 Stkcd/Week。
    rename_map = {}
    if 'Stkcd' not in fsfp_df.columns and 'code' in fsfp_df.columns:
        rename_map['code'] = 'Stkcd'
    if 'Week' not in fsfp_df.columns and 'week' in fsfp_df.columns:
        rename_map['week'] = 'Week'
    if rename_map:
        fsfp_df = fsfp_df.rename(columns=rename_map)

    if value_column in fsfp_df.columns:
        fsfp_df['FSFP'] = fsfp_df[value_column]
        print(f"使用 {value_column} 列作为FSFP计算输入")
    elif 'FSFP' in fsfp_df.columns:
        print("未找到指定取值列，使用 FSFP 列作为计算输入")
    else:
        raise ValueError(f"FSFP文件缺少取值列: {value_column} 或 FSFP")

    required_columns = {'Stkcd', 'Week', 'FSFP'}
    missing_columns = required_columns - set(fsfp_df.columns)
    if missing_columns:
        raise ValueError(f"FSFP文件缺少必要列: {sorted(missing_columns)}")
    
    # 解析周为年份和周数
    fsfp_df['Week'] = fsfp_df['Week'].astype(str)
    fsfp_df['year'] = fsfp_df['Week'].str[:4].astype(int)
    fsfp_df['week_num'] = fsfp_df['Week'].str[4:6].astype(int)
    
    return fsfp_df

def load_similarity_data(similarity_folder):
    """
    加载所有相似度矩阵数据
    """
    similarity_data = {}
    folder_path = Path(similarity_folder)
    
    for file in folder_path.glob("*.pkl"):  
        try:
            filename = file.stem
            year_match = re.search(r'(\d{4})', filename)
            if year_match:
                year = int(year_match.group(1))
                
                print(f"加载{year}年相似度矩阵...")
                with open(file, 'rb') as f:
                    similarity_data[year] = pickle.load(f)
                print(f"成功加载{year}年相似度矩阵，公司数量: {len(similarity_data[year]['companies'])}")
                    
        except Exception as e:
            print(f"处理文件{file}时出错: {e}")
    
    return similarity_data

def create_similarity_matrix(similarity_df):
    """
    创建公司间的相似度矩阵
    """
    # 获取所有唯一的公司
    all_companies = set(similarity_df['Symbol1']).union(set(similarity_df['Symbol2']))
    company_list = sorted(all_companies)
    company_to_idx = {company: idx for idx, company in enumerate(company_list)}
    
    # 创建相似度矩阵
    n = len(company_list)
    similarity_matrix = np.zeros((n, n))
    
    # 对角线设为1（公司与自身的相似度）
    np.fill_diagonal(similarity_matrix, 1)
    
    # 填充相似度值
    for _, row in similarity_df.iterrows():
        i = company_to_idx[row['Symbol1']]
        j = company_to_idx[row['Symbol2']]
        similarity_matrix[i, j] = row['Similarity']
        similarity_matrix[j, i] = row['Similarity']  # 对称矩阵
    
    return {
        'matrix': similarity_matrix,
        'companies': company_list,
        'company_to_idx': company_to_idx
    }

def load_group_data(group_file):
    """
    加载分组数据
    """
    group_df = pd.read_csv(group_file)
    print(f"加载分组数据，共{len(group_df)}行")
    
    return group_df

def create_fsfp_lookup(fsfp_df):
    """
    创建FSFP值的快速查找字典
    """
    lookup = {}
    for _, row in fsfp_df.iterrows():
        key = (row['year'], row['week_num'], row['Stkcd'])
        lookup[key] = row['FSFP']
    return lookup

def create_community_lookup(group_df):
    """
    创建社区分组的快速查找字典
    """
    community_lookup = {}
    company_to_community = {}  # 公司到社区的映射
    
    for _, row in group_df.iterrows():
        year = row['Year']
        company = row['Symbol']
        community_id = row['Community_ID']
        
        # 按年份和社区组织数据
        if year not in community_lookup:
            community_lookup[year] = {}
        
        if community_id not in community_lookup[year]:
            community_lookup[year][community_id] = []
        
        community_lookup[year][community_id].append(company)
        
        # 记录每个公司所属的社区
        company_to_community[(year, company)] = community_id
    
    return community_lookup, company_to_community


def calculate_peer_fsfp_weekly(fsfp_df, group_df, similarity_data, output_file):
    """
    按周计算同行FSFP：使用年度分组数据
    改进：将没有FSFP数据的同行公司视为0
    """
    print("开始按周计算同行FSFP...")
    
    # 预先建立查找字典
    fsfp_lookup = create_fsfp_lookup(fsfp_df)
    community_lookup, company_to_community = create_community_lookup(group_df)
    
    results = []
    
    # 按年份分组处理
    years = sorted(fsfp_df['year'].unique())
    
    for year in years:
        print(f"处理 {year} 年数据...")
        year_fsfp_data = fsfp_df[fsfp_df['year'] == year]
        
        if year not in similarity_data:
            print(f"  {year}年无相似度数据，使用原始FSFP")
            # 没有相似度数据，使用原始FSFP
            for _, row in year_fsfp_data.iterrows():
                results.append({
                    'Stkcd': row['Stkcd'],
                    'Week': row['Week'],
                    'Peer_FSFP': row['FSFP']  # 没有同行数据，使用自身
                })
            continue
        
        # 获取该年份的相似度矩阵信息
        sim_info = similarity_data[year]
        similarity_matrix = sim_info['matrix']
        company_list = sim_info['companies']
        company_to_idx = sim_info['company_to_idx']
        
        # 获取该年份的分组信息
        year_communities = community_lookup.get(year, {})
        
        # 按周处理
        weeks = sorted(year_fsfp_data['week_num'].unique())
        
        for week_num in weeks:
            #print(f"  处理 {year}年第{week_num}周...")
            week_data = year_fsfp_data[year_fsfp_data['week_num'] == week_num]
            week_id = f"{year}{week_num:02d}"
            
            # 按社区分组处理
            for community_id, community_companies in year_communities.items():
                if len(community_companies) == 0:
                    continue
                
                # 第一步：找出社区中所有公司（包括有FSFP和没有FSFP的）
                all_community_companies = []
                all_community_indices = []
                fsfp_values = []
                
                for company in community_companies:
                    if company in company_to_idx:  # 确保公司在相似度矩阵中
                        all_community_companies.append(company)
                        all_community_indices.append(company_to_idx[company])
                        # 获取FSFP值，如果没有则为0
                        fsfp_value = fsfp_lookup.get((year, week_num, company), 0.0)
                        fsfp_values.append(fsfp_value)
                
                # 如果没有有效数据，跳过
                if len(all_community_companies) == 0:
                    continue
                
                if len(all_community_companies) == 1:
                    # 如果只有一个公司，同行FSFP等于自身FSFP
                    peer_fsfp_sub = fsfp_values
                else:
                    # 第二步：提取完整的相似度子矩阵
                    sub_matrix = similarity_matrix[np.ix_(all_community_indices, all_community_indices)]
                    fsfp_vector = np.array(fsfp_values)
                    
                    # 第三步：矩阵运算
                    weighted_fsfp = sub_matrix @ fsfp_vector
                    total_weights = sub_matrix.sum(axis=1)
                    
                    # 避免除零错误
                    total_weights[total_weights == 0] = 1
                    
                    # 计算同行FSFP
                    peer_fsfp_sub = weighted_fsfp / total_weights
                
                # 第四步：存储结果（只为当前周有数据的公司存储）
                for i, company in enumerate(all_community_companies):
                    # 只存储当前周有FSFP数据的公司
                    if (year, week_num, company) in fsfp_lookup:
                        results.append({
                            'Stkcd': company,
                            'Week': week_id,
                            'Peer_FSFP': peer_fsfp_sub[i]
                        })
            
            # 处理当前周不在任何社区或有缺失的公司
            processed_companies = {r['Stkcd'] for r in results if r['Week'] == week_id}
            for _, row in week_data.iterrows():
                if row['Stkcd'] not in processed_companies:
                    # 使用原始FSFP作为同行FSFP
                    results.append({
                        'Stkcd': row['Stkcd'],
                        'Week': row['Week'],
                        'Peer_FSFP': row['FSFP']
                    })
    
    # 处理剩余记录
    print("处理剩余记录...")
    processed_keys = {(r['Stkcd'], r['Week']) for r in results}
    
    for _, row in fsfp_df.iterrows():
        key = (row['Stkcd'], row['Week'])
        if key not in processed_keys:
            results.append({
                'Stkcd': row['Stkcd'],
                'Week': row['Week'],
                'Peer_FSFP': row['FSFP']
            })
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"结果已保存到 {output_file}")
    return result_df

def main():
    """
    主函数
    """
    # 文件路径配置
    fsfp_file = "fsfp_weekly_filled_0509.csv"  # 包含Stkcd/code, fsfp_original, Week/week列
    group_file = "data/business_communities_all_years.csv"  # 分组数据
    similarity_folder = "similarity_matrices"  # 相似度数据文件夹
    output_file = "peer_fsfp_weekly_0509.csv"
       
    # 加载数据
    fsfp_df = load_and_prepare_fsfp_data(fsfp_file)
    group_df = load_group_data(group_file)
    
    # 加载相似度数据
    similarity_data = load_similarity_data(similarity_folder)
    
    # 计算同行FSFP
    result_df = calculate_peer_fsfp_weekly(fsfp_df, group_df, similarity_data, output_file)
    
    # 显示统计信息
    print(f"原始FSFP统计: 均值={fsfp_df['FSFP'].mean():.4f}, 标准差={fsfp_df['FSFP'].std():.4f}")
    print(f"同行FSFP统计: 均值={result_df['Peer_FSFP'].mean():.4f}, 标准差={result_df['Peer_FSFP'].std():.4f}")
    

if __name__ == "__main__":
    main()


