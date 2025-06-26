import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
from scipy import stats
import requests
import swisseph as swe


# 设置 ephemeris 文件路径（首次运行会自动下载）
swe.set_ephe_path('/tmp')  # 或本地任意可写目录

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
    """用真实天文数据计算比特币的本命星盘"""
    natal_positions = {}
    jd = swe.julday(2009, 1, 11, 9.0, swe.GREG_CAL)  # 比特币创始时间
    for planet, pid in zip(PLANETS, [
        swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS,
        swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO
    ]):
        result, ret = swe.calc_ut(jd, pid)
        lon, lat, dist = result[0:3]
        natal_positions[f"{planet}_经度"] = round(lon, 2)
        natal_positions[f"{planet}_纬度"] = round(lat, 2)
    return {
        '资产': '比特币',
        '创始时间': BITCOIN_FOUNDATION_TIME.strftime("%Y-%m-%d %H:%M:%S") + " (西五区)",
        '地点': BITCOIN_LOCATION,
        '行星位置': natal_positions
    }

def calculate_daily_transit(date, natal_positions):
    """用真实天文数据计算指定日期的行运盘"""
    transit_positions = {}
    jd = swe.julday(date.year, date.month, date.day, 12.0, swe.GREG_CAL)  # 用中午12点
    for planet, pid in zip(PLANETS, [
        swe.SUN, swe.MOON, swe.MERCURY, swe.VENUS, swe.MARS,
        swe.JUPITER, swe.SATURN, swe.URANUS, swe.NEPTUNE, swe.PLUTO
    ]):
        result, ret = swe.calc_ut(jd, pid)
        lon, lat, dist = result[0:3]
        transit_positions[f"{planet}_经度"] = round(lon, 2)
        transit_positions[f"{planet}_纬度"] = round(lat, 2)
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

def create_excel_report(natal_chart, daily_data, start_date, end_date, price_df=None):
    """创建Excel报告，支持合并价格数据和预测准确性标记"""
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
    
    # 合并价格数据
    if price_df is not None:
        daily_df = pd.merge(daily_df, price_df, on="日期", how="left")
    # 填充缺失值，避免KeyError
        daily_df['价格'] = daily_df['价格'].fillna(0)
        daily_df['日涨跌幅'] = daily_df['日涨跌幅'].fillna(0)
        daily_df['预测方向'] = daily_df['吉凶评级'].map({'利好': 1, '利空': -1, '中性': 0})
        daily_df['实际方向'] = daily_df['日涨跌幅'].apply(lambda x: 1 if x > 1 else (-1 if x < -1 else 0))
        daily_df['预测是否正确'] = daily_df.apply(
            lambda row: '正确' if row['预测方向'] == row['实际方向'] and row['预测方向'] != 0 else (
                '中性' if row['预测方向'] == 0 else '错误'
            ), axis=1
        )
    
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
        
        # 设置列宽（新增价格和涨跌幅两列）
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
        daily_sheet.set_column('K:K', 14)  # 价格
        daily_sheet.set_column('L:L', 14)  # 日涨跌幅
        daily_sheet.set_column('M:M', 12)  # 预测方向
        daily_sheet.set_column('N:N', 12)  # 实际方向
        daily_sheet.set_column('O:O', 14)  # 预测是否正确
        
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
        # 预测是否正确高亮
        daily_sheet.conditional_format('O2:O1000', {
            'type': 'text',
            'criteria': 'containing',
            'value': '正确',
            'format': format_good
        })
        daily_sheet.conditional_format('O2:O1000', {
            'type': 'text',
            'criteria': 'containing',
            'value': '错误',
            'format': format_bad
        })
    
    print(f"报告已生成: {os.path.abspath(filename)}")
    return filename, daily_df




def get_real_price_data(start_date, end_date):
    """从CoinGecko获取真实比特币价格数据"""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"
    # 转为int秒级时间戳
    from_ts = int(pd.Timestamp(start_date).timestamp())
    to_ts = int(pd.Timestamp(end_date).timestamp())
    params = {
        'vs_currency': 'usd',
        'from': from_ts,
        'to': to_ts
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("API请求失败:", response.status_code, response.text)
        return None
    data = response.json()
    # 解析价格数据
    prices = pd.DataFrame(data['prices'], columns=['timestamp', '价格'])
    prices['日期'] = pd.to_datetime(prices['timestamp'], unit='ms').dt.strftime('%Y-%m-%d')
    prices['价格'] = prices['价格'].astype(float)
    # 计算日涨跌幅
    prices['日涨跌幅'] = prices['价格'].pct_change() * 100
    prices['日涨跌幅'] = prices['日涨跌幅'].fillna(0)
    # 去重（有时API会返回同一天多条数据，保留每日最后一条）
    prices = prices.groupby('日期').last().reset_index()
    return prices[['日期', '价格', '日涨跌幅']]
    



def analyze_correlation(astrology_df, price_df):
    """分析星盘评分与价格变动的相关性"""
    
    # 合并数据如果astrology_df未合并
    if '日涨跌幅' not in astrology_df.columns:
        merged_df = pd.merge(astrology_df, price_df, on="日期", how="left")
        # 填充缺失，确保不会KeyError
        merged_df['日涨跌幅'] = merged_df['日涨跌幅'].fillna(0)
        merged_df['价格'] = merged_df['价格'].fillna(0)
        # 调试输出
        # print("merged_df columns:", merged_df.columns)
    else:
        merged_df = astrology_df
    
    # 计算相关性
    corr_score = merged_df['综合分数'].corr(merged_df['日涨跌幅'])
    corr_short = merged_df['短期分数'].corr(merged_df['日涨跌幅'])
    corr_long = merged_df['长期分数'].corr(merged_df['日涨跌幅'])
    
    # 预测准确率分析
    merged_df['预测方向'] = merged_df['综合分数'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    merged_df['实际方向'] = merged_df['日涨跌幅'].apply(lambda x: 1 if x > 1 else (-1 if x < -1 else 0))
    
    # 只考虑有明确预测的日子
    valid_days = merged_df[merged_df['预测方向'] != 0]
    accuracy = sum(valid_days['预测方向'] == valid_days['实际方向']) / len(valid_days) if len(valid_days) > 0 else 0
    
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
    start_date = "2024-11-01"
    end_date = "2025-07-01"
    
    print("=" * 70)
    print(f"比特币星盘分析报告生成")
    print(f"创始时间: {BITCOIN_FOUNDATION_TIME.strftime('%Y-%m-%d %H:%M:%S')} (西五区)")
    print(f"分析时段: {start_date} 至 {end_date}")
    print("=" * 70)

    
    # 生成报告
    natal_chart, daily_data, date_range = generate_bitcoin_report(start_date, end_date)

     #      获得并使用真实数据
    price_df = get_real_price_data(start_date, end_date)
    print(price_df)
    
    # 创建Excel报告
    report_file, astrology_df = create_excel_report(natal_chart, daily_data, start_date, end_date, price_df=price_df)
    
    
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