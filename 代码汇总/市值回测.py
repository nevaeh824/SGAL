import pandas as pd
import numpy as np
import os
from pathlib import Path
import time

def normalize_stock_code(code):
    """标准化股票代码为6位格式，例如：1 -> 000001"""
    try:
        code_int = int(float(code))  # 先转为浮点数再转整数，处理科学计数法
        return f"{code_int:06d}"
    except:
        # 如果已经是6位字符串格式，直接返回
        if isinstance(code, str) and len(code) == 6 and code.isdigit():
            return code
        # 其他情况尝试提取数字
        import re
        numbers = re.findall(r'\d+', str(code))
        if numbers:
            return f"{int(numbers[0]):06d}"
        return str(code).zfill(6)

def load_weekly_price_data(folder_path):
    """
    快速加载周度价格数据
    """
    print("开始快速加载价格数据...")
    
    price_data = {}
    price_index = {}
    folder = Path(folder_path)
    
    csv_files = list(folder.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    for i, csv_file in enumerate(csv_files):
        if i % 500 == 0 and i > 0:        
            print(f"  已加载 {i}/{len(csv_files)} 文件")
        
        try:
            stock_code = csv_file.stem
            normalized_code = normalize_stock_code(stock_code)
            
            df = pd.read_csv(csv_file, dtype={'Week': str})
            df['Week'] = df['Week'].astype(str)
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Open', 'Close'])
            
            if not df.empty:
                df.set_index('Week', inplace=True)
                price_data[normalized_code] = df
                price_index[normalized_code] = set(df.index)
                
        except Exception as e:
            continue
    
    print(f"价格数据加载完成，共 {len(price_data)} 只股票")
    return price_data, price_index

def load_market_cap_data(mcap_folder):
    """
    加载市值数据，每个公司一个CSV文件
    文件格式: Week, MrkVal
    Week格式如: 200502 (2005年第2周)
    注意：文件名（股票代码）会被标准化为6位格式
    """
    print("开始加载市值数据...")
    
    mcap_data = {}
    mcap_index = {}
    folder = Path(mcap_folder)
    
    csv_files = list(folder.glob("*.csv"))
    print(f"找到 {len(csv_files)} 个市值数据文件")
    
    # 统计原始文件名格式
    sample_files = csv_files[:5]
    print(f"示例文件名: {[f.stem for f in sample_files]}")
    
    for i, csv_file in enumerate(csv_files):
        if i % 500 == 0 and i > 0:
            print(f"  已加载 {i}/{len(csv_files)} 文件")
        
        try:
            stock_code = csv_file.stem
            # 关键：标准化股票代码为6位格式
            normalized_code = normalize_stock_code(stock_code)
            
            df = pd.read_csv(csv_file, dtype={'Week': str})
            df['Week'] = df['Week'].astype(str)
            df['MrkVal'] = pd.to_numeric(df['MrkVal'], errors='coerce')
            df = df.dropna(subset=['MrkVal'])
            df = df[df['MrkVal'] > 0]  # 只保留正数市值
            
            if not df.empty:
                df.set_index('Week', inplace=True)
                mcap_data[normalized_code] = df
                mcap_index[normalized_code] = set(df.index)
                
        except Exception as e:
            print(f"警告: 处理文件 {csv_file.name} 时出错: {e}")
            continue
    
    print(f"市值数据加载完成，共 {len(mcap_data)} 只股票有市值数据")
    
    # 显示一些标准化后的代码示例
    sample_codes = list(mcap_data.keys())[:5]
    print(f"标准化后的股票代码示例: {sample_codes}")
    
    return mcap_data, mcap_index

def load_weekly_fsfp_data(fsfp_file):
    """
    快速加载FSFP数据并创建高效查询索引
    """
    print("开始快速加载FSFP数据...")
    
    try:
        fsfp_df = pd.read_csv(fsfp_file, dtype={'Week': str})
        fsfp_df['Stock'] = fsfp_df['Stkcd']
        
        # 标准化股票代码
        fsfp_df['Stock'] = fsfp_df['Stock'].apply(normalize_stock_code)
        fsfp_df['Week'] = fsfp_df['Week'].astype(str)
        fsfp_df['FSFP'] = pd.to_numeric(fsfp_df['FSFP'], errors='coerce').fillna(0)
        
        # 创建快速查询字典，记录每个(股票, 周)的FSFP值
        fsfp_dict = {}
        fsfp_exists_dict = {}
        
        for _, row in fsfp_df.iterrows():
            key = (row['Stock'], row['Week'])
            fsfp_dict[key] = row['FSFP']
            fsfp_exists_dict[key] = True
        
        unique_stocks = fsfp_df['Stock'].nunique()
        unique_weeks = fsfp_df['Week'].nunique()
        print(f"FSFP数据加载完成，共 {len(fsfp_df)} 条记录")
        print(f"  涉及股票数: {unique_stocks}")
        print(f"  涉及周数: {unique_weeks}")
        
        return fsfp_dict, fsfp_exists_dict, fsfp_df
    
    except Exception as e:
        print(f"加载FSFP数据出错: {e}")
        return {}, {}, pd.DataFrame()

def get_all_weeks(price_index, start_year=None, end_year=None):
    """
    获取所有周标识，支持按年份筛选
    """
    all_weeks = set()
    for weeks in price_index.values():
        all_weeks.update(weeks)
    
    all_weeks = sorted(list(all_weeks))
    
    if start_year is not None:
        all_weeks = [w for w in all_weeks if int(w[:4]) >= start_year]
    if end_year is not None:
        all_weeks = [w for w in all_weeks if int(w[:4]) <= end_year]
    
    return all_weeks

def calculate_metrics(weekly_returns, initial_capital, final_capital, weeks_per_year=52, risk_free_rate=0.025):
    """
    计算各项投资指标
    """
    returns = weekly_returns[~np.isnan(weekly_returns)]
    
    if len(returns) == 0:
        return None
    
    # 总收益率
    total_return = (final_capital - initial_capital) / initial_capital
    
    # 年化收益率
    n_weeks = len(returns)
    annual_return = (1 + total_return) ** (weeks_per_year / n_weeks) - 1
    
    # 年化波动率
    weekly_volatility = np.std(returns, ddof=1)
    annual_volatility = weekly_volatility * np.sqrt(weeks_per_year)
    
    # 最大回撤
    cumulative_returns = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # 夏普比率
    weekly_risk_free = (1 + risk_free_rate) ** (1 / weeks_per_year) - 1
    excess_returns = returns - weekly_risk_free
    if np.std(excess_returns, ddof=1) > 0:
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(weeks_per_year)
    else:
        sharpe_ratio = 0
    
    return {
        '总收益率': total_return,
        '年化收益率': annual_return,
        '年化波动率': annual_volatility,
        '最大回撤': max_drawdown,
        '夏普比率': sharpe_ratio,
        '周数': n_weeks
    }

def run_weekly_backtest_marketcap(price_data, price_index, mcap_data, mcap_index, 
                                    fsfp_dict, fsfp_exists_dict,
                                    initial_capital=1000, start_year=None, end_year=None):
    """
    市值加权周度回测 - 按照市值比例分配资金
    只投资同时满足以下条件的股票：
    1. 有价格数据
    2. 有市值数据
    3. 有FSFP数据
    4. FSFP=0
    """
    print("\n开始市值加权周度回测...")
    if start_year:
        print(f"回测起始年份: {start_year}")
    if end_year:
        print(f"回测结束年份: {end_year}")
    print(f"初始资金: {initial_capital:.2f}")
    print(f"投资策略: 市值加权，只投资有FSFP数据且FSFP=0的股票")
    
    start_time = time.time()
    
    # 获取所有周
    all_weeks = get_all_weeks(price_index, start_year, end_year)
    total_weeks = len(all_weeks)
    print(f"总周数: {total_weeks}")
    print(f"有价格数据的股票数量: {len(price_data)}")
    print(f"有市值数据的股票数量: {len(mcap_data)}")
    print(f"有FSFP数据的股票数量: {len(fsfp_exists_dict)}")
    
    # 检查代码对齐情况
    price_stocks = set(price_data.keys())
    mcap_stocks = set(mcap_data.keys())
    fsfp_stocks = set([k[0] for k in fsfp_exists_dict.keys()])
    
    print(f"\n数据对齐情况:")
    print(f"  价格数据股票数: {len(price_stocks)}")
    print(f"  市值数据股票数: {len(mcap_stocks)}")
    print(f"  FSFP数据股票数: {len(fsfp_stocks)}")
    print(f"  三数据交集股票数: {len(price_stocks & mcap_stocks & fsfp_stocks)}")
    
    results = []
    current_capital = initial_capital
    
    # 周度回测主循环
    for i, week in enumerate(all_weeks):
        if i == total_weeks - 1:
            break
        
        # 每50周打印进度
        if i % 50 == 0 and i > 0:
            elapsed = time.time() - start_time
            progress = i / (total_weeks - 1) * 100
            weeks_per_sec = i / elapsed
            remaining = (total_weeks - i) / weeks_per_sec if weeks_per_sec > 0 else 0
            
            print(f"[{i:5d}/{total_weeks-1:5d}] {progress:5.1f}% | "
                  f"资金: {current_capital:12.2f} | "
                  f"用时: {elapsed:5.0f}s | "
                  f"剩余: {remaining/60:5.0f}分")
        
        # 收集本周所有符合条件的股票
        eligible_stocks = []  # 存储符合条件的股票代码
        market_caps = []      # 存储对应的市值
        
        # 遍历有价格数据的股票（所有代码已经是标准化后的6位格式）
        for stock_code in price_index.keys():
            # 检查是否有本周的价格数据
            if week not in price_index.get(stock_code, set()):
                continue
            
            # 检查是否有本周的市值数据
            if stock_code not in mcap_index or week not in mcap_index.get(stock_code, set()):
                continue
            
            # 检查是否有FSFP数据且FSFP=0
            key = (stock_code, week)
            if key not in fsfp_exists_dict:
                continue  # 没有FSFP数据，跳过
            
            if fsfp_dict[key] != 0:
                continue  # FSFP≠0，跳过
            
            # 获取市值
            try:
                mcap_value = mcap_data[stock_code].loc[week, 'MrkVal']
                if pd.isna(mcap_value) or mcap_value <= 0:
                    continue
                
                # 检查价格数据有效性
                open_price = price_data[stock_code].loc[week, 'Open']
                close_price = price_data[stock_code].loc[week, 'Close']
                if pd.isna(open_price) or pd.isna(close_price) or open_price <= 0:
                    continue
                
                eligible_stocks.append(stock_code)
                market_caps.append(mcap_value)
                
            except Exception as e:
                continue
        
        n_eligible = len(eligible_stocks)
        total_market_cap = sum(market_caps)
        
        # 统计信息
        # 计算总股票数（有价格数据的）
        total_stocks_with_price = sum(1 for stock in price_index.keys() if week in price_index.get(stock, set()))
        
        # 计算有市值数据的股票数
        stocks_with_mcap = sum(1 for stock in mcap_index.keys() if week in mcap_index.get(stock, set()))
        
        # 计算有FSFP数据的股票数（不管值是多少）
        stocks_with_fsfp = 0
        stocks_fsfp_zero = 0
        for stock_code in price_index.keys():
            if week not in price_index.get(stock_code, set()):
                continue
            key = (stock_code, week)
            if key in fsfp_exists_dict:
                stocks_with_fsfp += 1
                if fsfp_dict[key] == 0:
                    stocks_fsfp_zero += 1
        
        if n_eligible == 0 or total_market_cap == 0:
            # 没有符合条件的股票，资金闲置
            cumulative_return = (current_capital - initial_capital) / initial_capital
            results.append({
                'Week': week,
                'Total_Stocks_Price': total_stocks_with_price,
                'Stocks_With_Mcap': stocks_with_mcap,
                'Stocks_With_FSFP': stocks_with_fsfp,
                'FSFP_NonZero': stocks_with_fsfp - stocks_fsfp_zero,
                'FSFP_Zero_Total': stocks_fsfp_zero,
                'Eligible_Stocks': 0,
                'Total_Market_Cap': 0,
                'Start_Capital': current_capital,
                'End_Capital': current_capital,
                'Weekly_Return': 0,
                'Cumulative_Return': cumulative_return
            })
            continue
        
        # 市值加权投资
        week_end_capital = 0
        
        for stock_code, mcap_value in zip(eligible_stocks, market_caps):
            # 计算该股票的投资金额
            weight = mcap_value / total_market_cap
            investment_amount = current_capital * weight
            
            # 根据开盘价计算购买股数
            open_price = price_data[stock_code].loc[week, 'Open']
            shares = investment_amount / open_price
            
            # 根据收盘价计算周末价值
            close_price = price_data[stock_code].loc[week, 'Close']
            week_end_capital += shares * close_price
        
        # 计算收益率
        weekly_return = (week_end_capital - current_capital) / current_capital if current_capital > 0 else 0
        
        # 更新资金
        current_capital = week_end_capital
        cumulative_return = (current_capital - initial_capital) / initial_capital
        
        results.append({
            'Week': week,
            'Total_Stocks_Price': total_stocks_with_price,
            'Stocks_With_Mcap': stocks_with_mcap,
            'Stocks_With_FSFP': stocks_with_fsfp,
            'FSFP_NonZero': stocks_with_fsfp - stocks_fsfp_zero,
            'FSFP_Zero_Total': stocks_fsfp_zero,
            'Eligible_Stocks': n_eligible,
            'Total_Market_Cap': total_market_cap,
            'Start_Capital': results[-1]['End_Capital'] if results else initial_capital,
            'End_Capital': current_capital,
            'Weekly_Return': weekly_return,
            'Cumulative_Return': cumulative_return
        })
    
    # 完成回测
    total_time = time.time() - start_time
    print(f"\n回测完成! 总用时: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    
    results_df = pd.DataFrame(results)
    
    # 计算最终指标
    if not results_df.empty:
        weekly_returns = results_df['Weekly_Return'].values
        final_capital = results_df['End_Capital'].iloc[-1]
        metrics = calculate_metrics(weekly_returns, initial_capital, final_capital)
    else:
        metrics = None
    
    return results_df, metrics

def verify_code_alignment(price_data, mcap_data, fsfp_exists_dict):
    """
    验证三个数据源的股票代码是否对齐
    """
    print("\n" + "=" * 60)
    print("股票代码对齐验证")
    print("=" * 60)
    
    price_codes = set(price_data.keys())
    mcap_codes = set(mcap_data.keys())
    fsfp_codes = set([k[0] for k in fsfp_exists_dict.keys()])
    
    print(f"价格数据股票代码数: {len(price_codes)}")
    print(f"市值数据股票代码数: {len(mcap_codes)}")
    print(f"FSFP数据股票代码数: {len(fsfp_codes)}")
    print(f"\n三数据交集股票代码数: {len(price_codes & mcap_codes & fsfp_codes)}")
    
    # 显示一些示例
    print(f"\n价格数据示例代码: {list(price_codes)[:5]}")
    print(f"市值数据示例代码: {list(mcap_codes)[:5]}")
    print(f"FSFP数据示例代码: {list(fsfp_codes)[:5]}")
    
    # 检查只存在于价格数据但不在市值数据中的代码
    only_price = price_codes - mcap_codes
    if only_price:
        print(f"\n警告: {len(only_price)} 个股票只有价格数据但没有市值数据")
        print(f"示例: {list(only_price)[:5]}")
    
    # 检查只存在于市值数据但不在价格数据中的代码
    only_mcap = mcap_codes - price_codes
    if only_mcap:
        print(f"\n警告: {len(only_mcap)} 个股票只有市值数据但没有价格数据")
        print(f"示例: {list(only_mcap)[:5]}")
    
    return price_codes & mcap_codes & fsfp_codes

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 配置路径
    price_folder = "4414公司单个公司周度开收盘数据"#"4414公司单个公司周度开收盘数据"
    mcap_folder = "市值"  # 请修改为您的市值数据文件夹路径
    fsfp_file ="sgal_aligned.csv" #"bandit_prediction.csv"#"fsfp_weekly_corrected_week.csv"
    
    # ==================== 回测参数配置 ====================
    START_YEAR = 2018      # 回测起始年份，设为 None 则从最早开始
    END_YEAR = None        # 回测结束年份，设为 None 则到最后结束
    INITIAL_CAPITAL = 3000 # 初始资金
    RISK_FREE_RATE = 0 # 无风险利率（2.5%）
    # ====================================================
    
    # 检查文件
    if not os.path.exists(price_folder):
        print(f"错误: 价格数据文件夹 '{price_folder}' 不存在!")
        exit(1)
    
    if not os.path.exists(mcap_folder):
        print(f"错误: 市值数据文件夹 '{mcap_folder}' 不存在!")
        exit(1)
    
    if not os.path.exists(fsfp_file):
        print(f"错误: FSFP文件 '{fsfp_file}' 不存在!")
        exit(1)
    
    # 加载数据
    print("=" * 60)
    print("加载周度价格数据...")
    price_data, price_index = load_weekly_price_data(price_folder)
    print(f"加载完成: 共 {len(price_data)} 只股票")
    
    print("\n" + "=" * 60)
    print("加载市值数据...")
    mcap_data, mcap_index = load_market_cap_data(mcap_folder)
    print(f"加载完成: 共 {len(mcap_data)} 只股票有市值数据")
    
    print("\n" + "=" * 60)
    print("加载周度FSFP数据...")
    fsfp_dict, fsfp_exists_dict, fsfp_df = load_weekly_fsfp_data(fsfp_file)
    print(f"加载完成: 共 {len(fsfp_df)} 条记录")
    
    if len(price_data) == 0:
        print("错误: 没有加载到价格数据!")
        exit(1)
    
    if len(mcap_data) == 0:
        print("错误: 没有加载到市值数据!")
        exit(1)
    
    if len(fsfp_df) == 0:
        print("错误: 没有加载到FSFP数据!")
        exit(1)
    
    # 验证代码对齐
    common_stocks = verify_code_alignment(price_data, mcap_data, fsfp_exists_dict)
    
    print(len(common_stocks))
    if len(common_stocks) == 0:
        print("\n错误: 三个数据源没有共同的股票代码!")
        print("请检查股票代码标准化函数是否正确处理了所有格式")
        exit(1)
    
    # 运行回测
    print("\n" + "=" * 60)
    print("运行市值加权周度回测...")
    print("策略说明: 只投资有FSFP数据且FSFP=0的股票")
    print("        资金按股票市值占总投资组合市值的比例分配")
    print("        没有FSFP数据的股票即使有价格和市值数据也不考虑")
    print(f"        共同股票数: {len(common_stocks)}")
    
    results_df, metrics = run_weekly_backtest_marketcap(
        price_data, price_index, mcap_data, mcap_index,
        fsfp_dict, fsfp_exists_dict,
        initial_capital=INITIAL_CAPITAL,
        start_year=START_YEAR,
        end_year=END_YEAR
    )
    
    # ==================== 输出结果 ====================
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)
    
    if results_df.empty:
        print("没有回测结果!")
        exit(0)
    
    # 每周详细结果（前20周示例）
    print("\n每周详细结果（前20周）:")
    print("-" * 160)
    print(f"{'周':<8} {'有价格':<8} {'有市值':<8} {'有FSFP':<8} {'FSFP≠0':<8} {'FSFP=0':<8} {'实际投资':<8} {'周初资金':<12} {'周末资金':<12} {'周收益率':<10} {'累积收益率':<12}")
    print("-" * 160)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        if i < 20:
            weekly_return = row['Weekly_Return'] * 100
            cumulative_return = row['Cumulative_Return'] * 100
            print(f"{row['Week']:<8} {row['Total_Stocks_Price']:<8} {row['Stocks_With_Mcap']:<8} "
                  f"{row['Stocks_With_FSFP']:<8} {row['FSFP_NonZero']:<8} {row['FSFP_Zero_Total']:<8} "
                  f"{row['Eligible_Stocks']:<8} {row['Start_Capital']:<12.2f} {row['End_Capital']:<12.2f} "
                  f"{weekly_return:<10.4f}% {cumulative_return:<12.4f}%")
    
    if len(results_df) > 20:
        print(f"... (共{len(results_df)}周)")
    
    # 整体指标
    print("\n" + "=" * 80)
    print("整体绩效指标")
    print("=" * 80)
    
    if metrics:
        print(f"回测期间: {START_YEAR}年 - {END_YEAR if END_YEAR else '数据结束'}")
        print(f"回测周数: {metrics['周数']}")
        print(f"初始资金: {INITIAL_CAPITAL:.2f}")
        print(f"最终资金: {results_df['End_Capital'].iloc[-1]:.2f}")
        print(f"总收益率: {metrics['总收益率']*100:.2f}%")
        print(f"年化收益率: {metrics['年化收益率']*100:.2f}%")
        print(f"年化波动率: {metrics['年化波动率']*100:.2f}%")
        print(f"夏普比率: {metrics['夏普比率']:.4f}")
        print(f"最大回撤: {metrics['最大回撤']*100:.2f}%")
    
    # 额外统计
    print("\n" + "=" * 80)
    print("交易统计")
    print("=" * 80)
    
    positive_weeks = len(results_df[results_df['Weekly_Return'] > 0])
    negative_weeks = len(results_df[results_df['Weekly_Return'] < 0])
    zero_weeks = len(results_df[results_df['Weekly_Return'] == 0])
    
    print(f"盈利周数: {positive_weeks} ({positive_weeks/len(results_df)*100:.1f}%)")
    print(f"亏损周数: {negative_weeks} ({negative_weeks/len(results_df)*100:.1f}%)")
    print(f"持平周数: {zero_weeks} ({zero_weeks/len(results_df)*100:.1f}%)")
    
    # 平均每周统计
    avg_total_price = results_df['Total_Stocks_Price'].mean()
    avg_with_mcap = results_df['Stocks_With_Mcap'].mean()
    avg_with_fsfp = results_df['Stocks_With_FSFP'].mean()
    avg_fsfp_zero = results_df['FSFP_Zero_Total'].mean()
    avg_eligible = results_df['Eligible_Stocks'].mean()
    
    print(f"\n平均每周统计:")
    print(f"  有价格数据的股票数: {avg_total_price:.1f}")
    print(f"  其中有市值数据的股票数: {avg_with_mcap:.1f} ({avg_with_mcap/avg_total_price*100:.1f}%)")
    print(f"  其中有FSFP数据的股票数: {avg_with_fsfp:.1f} ({avg_with_fsfp/avg_with_mcap*100:.1f}% of 有市值)")
    print(f"  其中FSFP=0的股票数: {avg_fsfp_zero:.1f}")
    print(f"  实际投资股票数（FSFP=0且有市值）: {avg_eligible:.1f}")
    print(f"  实际投资比例: {avg_eligible/avg_total_price*100:.1f}% (占全部有价格股票)")
    
    # 保存结果
    output_file = f'marketcap_weighted_backtest_{START_YEAR}_to_{END_YEAR if END_YEAR else "end"}_cap{INITIAL_CAPITAL}fsfp_sg_al_aligned_pred.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")
    
    # 保存指标到文本文件
    if metrics:
        with open(output_file.replace('.csv', '_metrics.txt'), 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("市值加权策略回测绩效指标\n")
            f.write("=" * 60 + "\n")
            f.write("策略说明: 市值加权投资，只投资有FSFP数据且FSFP=0的股票\n")
            f.write("        资金按股票市值占总投资组合市值的比例分配\n")
            f.write("        没有FSFP数据的股票即使有价格和市值数据也不考虑\n\n")
            f.write(f"回测起始年份: {START_YEAR}\n")
            f.write(f"回测结束年份: {END_YEAR if END_YEAR else '数据结束'}\n")
            f.write(f"初始资金: {INITIAL_CAPITAL:.2f}\n")
            f.write(f"最终资金: {results_df['End_Capital'].iloc[-1]:.2f}\n")
            f.write(f"总收益率: {metrics['总收益率']*100:.2f}%\n")
            f.write(f"年化收益率: {metrics['年化收益率']*100:.2f}%\n")
            f.write(f"年化波动率: {metrics['年化波动率']*100:.2f}%\n")
            f.write(f"夏普比率: {metrics['夏普比率']:.4f}\n")
            f.write(f"最大回撤: {metrics['最大回撤']*100:.2f}%\n")
            f.write(f"回测周数: {metrics['周数']}\n")
            f.write(f"盈利周数: {positive_weeks} ({positive_weeks/len(results_df)*100:.1f}%)\n")
            f.write(f"亏损周数: {negative_weeks} ({negative_weeks/len(results_df)*100:.1f}%)\n")
            f.write(f"\n平均每周统计:\n")
            f.write(f"  有价格数据的股票数: {avg_total_price:.1f}\n")
            f.write(f"  其中有市值数据的股票数: {avg_with_mcap:.1f} ({avg_with_mcap/avg_total_price*100:.1f}%)\n")
            f.write(f"  其中有FSFP数据的股票数: {avg_with_fsfp:.1f} ({avg_with_fsfp/avg_with_mcap*100:.1f}% of 有市值)\n")
            f.write(f"  实际投资股票数: {avg_eligible:.1f}\n")
        print(f"绩效指标已保存到: {output_file.replace('.csv', '_metricsfsfp_sg_al_aligned_pred.txt')}")