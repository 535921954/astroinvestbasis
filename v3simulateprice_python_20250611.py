import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
from scipy import stats
import requests

# 比特币创始信息（西五区时间2009年1月11日9:00）
BITCOIN_FOUNDATION_TIME = datetime(2009, 1, 11, 9, 0)
BITCOIN_LOCATION = "Unknown"

# 行星列表
PLANETS = ['太阳', '月亮', '水星', '金星', '火星', '木星', '土星', '天王星', '海王星', '冥王星']

# 行星移动速度分类
FAST_MOVING_PLANETS = ['月亮', '水星', '金星', '太阳', '火星']  # 短期影响（日/周）
SLOW_MOVING_PLANETS = ['木星', '土星', '天王星', '海王星', '冥王星']  # 长期影响（月/年）

# 相位类型
ASPECTS = ['合相', '六分相', '刑相', '拱相', '冲相']

# 相位吉凶映射
ASPECT_LUCK = {
    '合相': 0.5,    # 中性偏吉
    '六分相': 1.0,  # 吉
    '刑相': -0.8,   # 凶
    '拱相': 1.0,    # 吉
    '冲相': -1.0    # 凶
}

# 行星权重（根据占星学重要性）
PLANET_WEIGHTS = {
    '太阳': 1.0, '月亮': 0.9, '水星': 0.7,
    '金星': 0.8, '火星': 0.8, '木星': 0.9,
    '土星': 0.9, '天王星': 0.7, '海王星': 0.6, '冥王星': 0.6
}

# 时间影响因子
TIME_FACTORS = {
    'daily': 1.0,    # 当日影响
    'weekly': 0.7,   # 本周影响
    'monthly': 0.5   # 本月影响
}

def calculate_natal_chart():
    """计算比特币的本命星盘"""
    natal_positions = {}
    for planet in PLANETS:
        # 生成随机但合理的行星位置（0-360度）
        natal_positions[f"{planet}_经度"] = round(np.random.uniform(0, 360), 2)
        natal_positions[f"{planet}_纬度"] = round(np.random.uniform(-5, 5), 2)
    
    return {
        '资产': '比特币',
        '创始时间': BITCOIN_FOUNDATION_TIME.strftime("%Y-%m-%d %H:%M:%S") + " (西五区)",
        '地点': BITCOIN_LOCATION,
        '行星位置': natal_positions
    }

def calculate_daily_transit(date, natal_positions):
    """计算指定日期的行运盘"""
    transit_positions = {}
    for planet in PLANETS:
        # 基于本命盘位置生成每日变化
        base_lon = natal_positions[f"{planet}_经度"]
        base_lat = natal_positions[f"{planet}_纬度"]
        
        # 根据行星速度生成变化
        if planet in FAST_MOVING_PLANETS:
            lon_change = np.random.uniform(-1.5, 1.5)
            lat_change = np.random.uniform(-0.5, 0.5)
        else:  # 慢速行星
            lon_change = np.random.uniform(-0.3, 0.3)
            lat_change = np.random.uniform(-0.1, 0.1)
        
        transit_positions[f"{planet}_经度"] = round((base_lon + lon_change * 30) % 360, 2)
        transit_positions[f"{planet}_纬度"] = round(base_lat + lat_change, 2)
    
    return transit_positions

def calculate_aspects(natal_positions, transit_positions):
    """计算本命盘与行运盘之间的相位"""
    aspects = []
    for planet in PLANETS:
        # 计算经度角度差
        natal_lon = natal_positions[f"{planet}_经度"]
        transit_lon = transit_positions[f"{planet}_经度"]
        angle_diff = abs(natal_lon - transit_lon)
        
        # 确保角度差在0-180度范围内
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        # 根据角度差确定相位
        if angle_diff < 8:
            aspect_type = '合相'
        elif 52 < angle_diff < 68:
            aspect_type = '六分相'
        elif 82 < angle_diff < 98:
            aspect_type = '刑相'
        elif 112 < angle_diff < 128:
            aspect_type = '拱相'
        elif 172 < angle_diff < 188:
            aspect_type = '冲相'
        else:
            continue  # 没有主要相位
        
        # 计算相位分数（考虑行星权重和行星类型）
        aspect_score = ASPECT_LUCK[aspect_type] * PLANET_WEIGHTS[planet]
        
        # 分类短期/长期影响
        time_effect = 'short_term' if planet in FAST_MOVING_PLANETS else 'long_term'
        
        aspects.append({
            '行星': planet,
            '相位': aspect_type,
            '角度差': round(angle_diff, 2),
            '分数': round(aspect_score, 2),
            '影响类型': time_effect
        })
    
    return aspects

def calculate_daily_score(aspects):
    """计算每日总分（区分短期和长期影响）"""
    short_term_score = 0
    long_term_score = 0
    short_term_aspects = 0
    long_term_aspects = 0
    
    for aspect in aspects:
        if aspect['影响类型'] == 'short_term':
            short_term_score += aspect['分数']
            short_term_aspects += 1
        else:
            long_term_score += aspect['分数']
            long_term_aspects += 1
    
    # 计算平均分（避免除以零）
    avg_short = short_term_score / max(1, short_term_aspects)
    avg_long = long_term_score / max(1, long_term_aspects)
    
    # 综合每日分数
    daily_score = (avg_short * 0.6) + (avg_long * 0.4)
    
    # 确定吉凶等级
    if daily_score > 0.3:
        luck_label = "利好"
    elif daily_score < -0.3:
        luck_label = "利空"
    else:
        luck_label = "中性"
    
    return {
        '短期分数': round(avg_short, 2),
        '长期分数': round(avg_long, 2),
        '综合分数': round(daily_score, 2),
        '吉凶评级': luck_label,
        '短期相位数': short_term_aspects,
        '长期相位数': long_term_aspects
    }

def generate_bitcoin_report(start_date, end_date):
    """生成比特币星盘报告"""
    # 1. 计算本命星盘
    natal_chart = calculate_natal_chart()
    natal_positions = natal_chart['行星位置']
    
    print("生成比特币本命星盘...")
    
    # 2. 准备日期范围
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_data = []
    
    print(f"分析 {start_date} 至 {end_date} 的每日星盘相位...")
    
    # 3. 生成每日分析数据
    for date in date_range:
        # 计算行运
        transit = calculate_daily_transit(date, natal_positions)
        
        # 分析相位
        aspects = calculate_aspects(natal_positions, transit)
        
        # 计算分数和评级
        scores = calculate_daily_score(aspects)
        
        # 格式化相位信息
        aspect_desc = "; ".join(
            [f"{asp['行星']}{asp['相位']}({asp['角度差']}°)" 
             for asp in aspects]) if aspects else "无主要相位"
        
        # 分类短期/长期相位
        short_term_aspects = [asp for asp in aspects if asp['影响类型'] == 'short_term']
        long_term_aspects = [asp for asp in aspects if asp['影响类型'] == 'long_term']
        
        daily_data.append({
            "日期": date.strftime("%Y-%m-%d"),
            "相位": aspect_desc,
            "短期相位": "; ".join([f"{asp['行星']}{asp['相位']}" for asp in short_term_aspects]),
            "长期相位": "; ".join([f"{asp['行星']}{asp['相位']}" for asp in long_term_aspects]),
            "短期分数": scores['短期分数'],
            "长期分数": scores['长期分数'],
            "综合分数": scores['综合分数'],
            "吉凶评级": scores['吉凶评级'],
            "短期相位数": scores['短期相位数'],
            "长期相位数": scores['长期相位数']
        })
    
    print(f"共分析 {len(daily_data)} 天的星盘数据")
    
    return natal_chart, daily_data, date_range

def create_excel_report(natal_chart, daily_data, start_date, end_date):
    """创建Excel报告"""
    # 创建本命星盘数据框
    natal_df = pd.DataFrame([{
        "资产": natal_chart['资产'],
        "创始时间": natal_chart['创始时间'],
        "地点": natal_chart['地点']
    }])
    
    # 添加行星位置
    for planet in PLANETS:
        natal_df[f"{planet}经度"] = natal_chart['行星位置'][f"{planet}_经度"]
        natal_df[f"{planet}纬度"] = natal_chart['行星位置'][f"{planet}_纬度"]
    
    # 创建每日数据框
    daily_df = pd.DataFrame(daily_data)
    
    # 统计吉凶分布
    luck_stats = daily_df['吉凶评级'].value_counts()
    luck_summary = pd.DataFrame({
        "吉凶评级": luck_stats.index,
        "天数": luck_stats.values
    })
    
    # 创建输出目录
    os.makedirs("reports", exist_ok=True)
    
    # 生成文件名
    report_id = f"BTC_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    filename = f"reports/比特币星盘分析_{report_id}.xlsx"
    
    # 写入Excel
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # 本命星盘
        natal_df.to_excel(writer, sheet_name='本命星盘', index=False)
        
        # 每日分析
        daily_df.to_excel(writer, sheet_name='每日分析', index=False)
        
        # 吉凶统计
        luck_summary.to_excel(writer, sheet_name='吉凶统计', index=False)
        
        # 获取工作簿和工作表
        workbook = writer.book
        daily_sheet = writer.sheets['每日分析']
        
        # 设置列宽
        daily_sheet.set_column('A:A', 12)  # 日期
        daily_sheet.set_column('B:B', 50)  # 相位
        daily_sheet.set_column('C:C', 30)  # 短期相位
        daily_sheet.set_column('D:D', 30)  # 长期相位
        daily_sheet.set_column('E:E', 12)  # 短期分数
        daily_sheet.set_column('F:F', 12)  # 长期分数
        daily_sheet.set_column('G:G', 12)  # 综合分数
        daily_sheet.set_column('H:H', 12)  # 吉凶评级
        daily_sheet.set_column('I:I', 12)  # 短期相位数
        daily_sheet.set_column('J:J', 12)  # 长期相位数
        
        # 添加条件格式
        format_good = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        format_bad = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        format_neutral = workbook.add_format({'bg_color': '#FFEB9C', 'font_color': '#9C6500'})
        
        # 应用条件格式
        daily_sheet.conditional_format('H2:H1000', {
            'type': 'text',
            'criteria': 'containing',
            'value': '利好',
            'format': format_good
        })
        daily_sheet.conditional_format('H2:H1000', {
            'type': 'text',
            'criteria': 'containing',
            'value': '利空',
            'format': format_bad
        })
        daily_sheet.conditional_format('H2:H1000', {
            'type': 'text',
            'criteria': 'containing',
            'value': '中性',
            'format': format_neutral
        })
    
    print(f"报告已生成: {os.path.abspath(filename)}")
    return filename, daily_df

def simulate_price_data(date_range):
    """模拟比特币价格数据（实际应用应使用真实数据）"""
    prices = []
    current_price = 60000  # 起始价格
    
    for date in date_range:
        # 生成每日价格变化 (-5% 到 +5%)
        daily_change = random.uniform(-0.05, 0.05)
        current_price *= (1 + daily_change)
        
        prices.append({
            "日期": date.strftime("%Y-%m-%d"),
            "价格": round(current_price, 2),
            "日涨跌幅": round(daily_change * 100, 2)
        })
    
    return pd.DataFrame(prices)

def get_real_price_data(start_date, end_date):
    """从CoinGecko获取真实价格数据"""
    url = f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': int(pd.Timestamp(start_date).timestamp()),
        'to': int(pd.Timestamp(end_date).timestamp())
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    # 处理价格数据
    prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
    prices['date'] = pd.to_datetime(prices['timestamp'], unit='ms').apply(lambda x:x.strftime("%Y-%m-%d"))
    prices['daily_change'] = prices['price'].pct_change() * 100
    prices['daily_change'] = prices['daily_change'].apply(lambda x:round(x, 2))
    prices['price'] = prices['price'].apply(lambda x:round(x, 2))
    
    print(f'{prices.columns}')
    print(f'{prices.index}')
    print(f'{prices.shape}')

    prices.rename(columns={'date': '日期', 'price': '价格', 'daily_change': '日涨跌幅'}, inplace=True)

    return prices[['日期', '价格', '日涨跌幅']]

    

def analyze_correlation(astrology_df, price_df):
    """分析星盘评分与价格变动的相关性"""
    # 合并数据
    merged_df = pd.merge(astrology_df, price_df, on="日期")
    
    # 计算相关性
    corr_score = merged_df['综合分数'].corr(merged_df['日涨跌幅'])
    corr_short = merged_df['短期分数'].corr(merged_df['日涨跌幅'])
    corr_long = merged_df['长期分数'].corr(merged_df['日涨跌幅'])
    
    # 预测准确率分析
    merged_df['预测方向'] = merged_df['综合分数'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    merged_df['实际方向'] = merged_df['日涨跌幅'].apply(lambda x: 1 if x > 1 else (-1 if x < -1 else 0))
    
    # 只考虑有明确预测的日子
    valid_days = merged_df[merged_df['预测方向'] != 0]
    accuracy = sum(valid_days['预测方向'] == valid_days['实际方向']) / len(valid_days)
    
    # 绘制图表
    plt.figure(figsize=(12, 8))
    
    # 价格走势与综合分数
    plt.subplot(2, 1, 1)
    plt.plot(merged_df['日期'], merged_df['价格'], 'b-', label='价格')
    plt.ylabel('价格 (USD)')
    plt.legend(loc='upper left')
    
    ax2 = plt.twinx()
    plt.plot(merged_df['日期'], merged_df['综合分数'], 'r-', label='综合分数')
    plt.ylabel('星盘分数')
    plt.title('比特币价格与星盘综合分数对比')
    plt.legend(loc='upper right')
    
    # 分数与涨跌幅相关性
    plt.subplot(2, 1, 2)
    plt.scatter(merged_df['综合分数'], merged_df['日涨跌幅'], alpha=0.6)
    plt.xlabel('星盘综合分数')
    plt.ylabel('日涨跌幅 (%)')
    plt.title(f'星盘分数与价格涨跌幅相关性 (r = {corr_score:.2f})')
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('reports/价格与星盘相关性分析.png')
    
    # 相关性统计检验
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        merged_df['综合分数'], merged_df['日涨跌幅'])
    
    return {
        "相关性_综合分数": corr_score,
        "相关性_短期分数": corr_short,
        "相关性_长期分数": corr_long,
        "预测准确率": accuracy,
        "有效预测天数": len(valid_days),
        "总天数": len(merged_df),
        "回归方程": f"涨跌幅 = {slope:.4f} * 分数 + {intercept:.4f}",
        "p值": p_value,
        "图表路径": os.path.abspath('reports/价格与星盘相关性分析.png')
    }

def main():
    # 设置分析时间段
    start_date = "2025-05-01"
    end_date = "2025-06-01"
    
    print("=" * 70)
    print(f"比特币星盘分析报告生成")
    print(f"创始时间: {BITCOIN_FOUNDATION_TIME.strftime('%Y-%m-%d %H:%M:%S')} (西五区)")
    print(f"分析时段: {start_date} 至 {end_date}")
    print("=" * 70)
    
    # 生成报告
    natal_chart, daily_data, date_range = generate_bitcoin_report(start_date, end_date)
    
    # 创建Excel报告
    report_file, astrology_df = create_excel_report(natal_chart, daily_data, start_date, end_date)
    
    # 模拟价格数据（实际应用应使用真实数据）
    price_df = get_real_price_data(start_date, end_date)

    
    # 分析相关性
    correlation_results = analyze_correlation(astrology_df, price_df)
    
    # 显示摘要
    print("\n" + "=" * 70)
    print("分析摘要:")
    print(f"总天数: {len(astrology_df)}")
    print(f"利好天数: {len(astrology_df[astrology_df['吉凶评级'] == '利好'])}")
    print(f"利空天数: {len(astrology_df[astrology_df['吉凶评级'] == '利空'])}")
    print(f"中性天数: {len(astrology_df[astrology_df['吉凶评级'] == '中性'])}")
    
    print("\n价格相关性分析:")
    print(f"综合分数与价格涨跌幅相关性: {correlation_results['相关性_综合分数']:.4f}")
    print(f"短期分数与价格涨跌幅相关性: {correlation_results['相关性_短期分数']:.4f}")
    print(f"长期分数与价格涨跌幅相关性: {correlation_results['相关性_长期分数']:.4f}")
    print(f"预测准确率: {correlation_results['预测准确率']:.2%}")
    print(f"回归方程: {correlation_results['回归方程']}")
    print(f"统计显著性(p值): {correlation_results['p值']:.4f}")
    
    print("\n报告已成功生成!")
    print(f"1. 星盘分析报告: {report_file}")
    print(f"2. 相关性分析图表: {correlation_results['图表路径']}")

if __name__ == "__main__":
    main()