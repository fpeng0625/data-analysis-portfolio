# -*- coding: utf-8 -*-
"""
项目3：智能大棚环境监测数据分析系统
====================================
功能：
1. 连接SQLite数据库，创建传感器数据表
2. 生成模拟时序数据并插入数据库（3个月约2000条）
3. 使用SQL进行日均、周均统计查询
4. 异常数据检测（阈值法 + 统计法，SQL查询异常）
5. 趋势分析（日周期、季节趋势）
6. 相关性分析（4个指标之间的相关系数）
7. 异常预警规则（三级预警）
8. 分时段调控方案
9. 可视化图表

运行方式：
    pip install pandas numpy matplotlib seaborn
    python greenhouse_analysis.py

数据库说明：
    - 使用SQLite（Python自带，无需额外安装）
    - 数据库文件：greenhouse.db
    - 数据表：sensor_data（传感器时序数据）
    - SQL语法与MySQL基本一致

作者：冯鹏
课程：数据库课程设计
时间：2024.09 - 2024.12
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DB_PATH = 'greenhouse.db'


# ============================================================
# 第一步：数据库设计与建表
# ============================================================
def create_database():
    """创建数据库和传感器数据表"""
    print("=" * 60)
    print("第一步：数据库设计与建表")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('DROP TABLE IF EXISTS sensor_data')

    # 传感器数据表
    cursor.execute('''
        CREATE TABLE sensor_data (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   DATETIME NOT NULL,
            date        DATE,
            hour        INT,
            temperature FLOAT,
            humidity    FLOAT,
            light       FLOAT,
            co2         FLOAT,
            is_anomaly  TINYINT DEFAULT 0
        )
    ''')
    print("✓ 传感器数据表 sensor_data 创建成功")

    # 创建索引（按时间查询加速）
    cursor.execute('CREATE INDEX idx_timestamp ON sensor_data(timestamp)')
    cursor.execute('CREATE INDEX idx_date ON sensor_data(date)')
    print("✓ 时间索引创建成功")

    # 表结构说明
    print("""
【表结构设计】
sensor_data 表：
  - id          主键，自增
  - timestamp   采集时间（精确到秒）
  - date        日期（用于按天统计）
  - hour        小时（用于按时段分析）
  - temperature 温度(℃)
  - humidity    湿度(%)
  - light       光照强度(lux)
  - co2         CO₂浓度(ppm)
  - is_anomaly  是否异常（0正常1异常）

【索引设计】
  - idx_timestamp：按时间范围查询加速
  - idx_date：按天统计加速
""")

    conn.commit()
    return conn


# ============================================================
# 第二步：生成模拟数据并插入数据库
# ============================================================
def generate_and_insert_data(conn):
    """生成模拟传感器数据并插入数据库"""
    print("=" * 60)
    print("第二步：生成模拟数据并插入数据库")
    print("=" * 60)

    np.random.seed(42)
    cursor = conn.cursor()

    start_date = datetime(2024, 9, 1, 0, 0, 0)
    hours = 24 * 90  # 90天
    print(f"数据时长：90天，每小时1条，共 {hours} 条记录")

    records = []
    for i in range(hours):
        current_time = start_date + timedelta(hours=i)
        hour = current_time.hour
        day_of_year = current_time.timetuple().tm_yday

        # 温度：日周期+季节变化+噪声
        temp_base = 22 + 8 * np.sin((hour - 6) / 24 * 2 * np.pi)
        temp_seasonal = 3 * np.sin((day_of_year - 270) / 365 * 2 * np.pi)
        temperature = temp_base + temp_seasonal + np.random.normal(0, 1.5)

        # 湿度：与温度负相关
        humidity_base = 65 - 10 * np.sin((hour - 6) / 24 * 2 * np.pi)
        humidity = humidity_base + np.random.normal(0, 5)
        humidity = np.clip(humidity, 20, 95)

        # 光照：白天有晚上0
        if 6 <= hour <= 18:
            light_base = 30000 * np.sin((hour - 6) / 12 * np.pi)
        else:
            light_base = 0
        light = max(0, light_base + np.random.normal(0, 2000))

        # CO₂：白天低晚上高
        co2_base = 600 - 150 * np.sin((hour - 6) / 24 * 2 * np.pi)
        co2 = co2_base + np.random.normal(0, 30)

        # 随机注入异常（约5%）
        is_anomaly = 0
        if np.random.random() < 0.03:
            temperature = temperature + np.random.choice([-15, 15])
            is_anomaly = 1
        elif np.random.random() < 0.02:
            temperature = temperature + np.random.choice([-10, 10])
            humidity = humidity + np.random.choice([-20, 20])
            is_anomaly = 1

        records.append((
            current_time.strftime('%Y-%m-%d %H:%M:%S'),
            current_time.strftime('%Y-%m-%d'),
            hour,
            round(temperature, 1),
            round(humidity, 1),
            round(light, 0),
            round(co2, 1),
            is_anomaly
        ))

    # 批量插入
    cursor.executemany('''
        INSERT INTO sensor_data (timestamp, date, hour, temperature, humidity, light, co2, is_anomaly)
        VALUES (?,?,?,?,?,?,?,?)
    ''', records)
    conn.commit()

    # 验证：SQL查询数据量
    count = cursor.execute('SELECT COUNT(*) FROM sensor_data').fetchone()[0]
    anomaly_count = cursor.execute('SELECT COUNT(*) FROM sensor_data WHERE is_anomaly = 1').fetchone()[0]
    print(f"✓ 插入数据：{count} 条")
    print(f"✓ 实际异常数据：{anomaly_count} 条 ({anomaly_count/count:.1%})")

    # SQL查询数据概览
    print("\n数据概览（SQL查询）：")
    df_stats = pd.read_sql('''
        SELECT
            COUNT(*) AS total,
            ROUND(AVG(temperature), 2) AS temp_avg,
            ROUND(MIN(temperature), 2) AS temp_min,
            ROUND(MAX(temperature), 2) AS temp_max,
            ROUND(AVG(humidity), 2) AS humidity_avg,
            ROUND(AVG(light), 0) AS light_avg,
            ROUND(AVG(co2), 2) AS co2_avg
        FROM sensor_data
    ''', conn)
    print(df_stats.to_string(index=False))

    return count


# ============================================================
# 第三步：SQL数据统计（日均、周均）
# ============================================================
def sql_data_statistics(conn):
    """使用SQL进行日均、周均统计"""
    print("\n" + "=" * 60)
    print("第三步：SQL数据统计（日均、周均）")
    print("=" * 60)

    # 日均统计（SQL GROUP BY date）
    print("\n--- 日均统计（SQL: GROUP BY date） ---")
    sql_daily = '''
        SELECT
            date,
            ROUND(AVG(temperature), 2) AS temp_avg,
            ROUND(MAX(temperature), 2) AS temp_max,
            ROUND(MIN(temperature), 2) AS temp_min,
            ROUND(AVG(humidity), 2) AS humidity_avg,
            ROUND(AVG(light), 0) AS light_avg,
            ROUND(AVG(co2), 2) AS co2_avg,
            COUNT(*) AS record_count
        FROM sensor_data
        GROUP BY date
        ORDER BY date
    '''
    df_daily = pd.read_sql(sql_daily, conn)
    print(f"共 {len(df_daily)} 天数据，前5天：")
    print(df_daily.head().to_string(index=False))

    # 周均统计（SQL按周分组）
    print("\n--- 周均统计（SQL: 按周分组） ---")
    sql_weekly = '''
        SELECT
            CAST(strftime('%W', date) AS INTEGER) AS week_num,
            MIN(date) AS week_start,
            ROUND(AVG(temperature), 2) AS temp_avg,
            ROUND(AVG(humidity), 2) AS humidity_avg,
            ROUND(AVG(light), 0) AS light_avg,
            ROUND(AVG(co2), 2) AS co2_avg
        FROM sensor_data
        GROUP BY week_num
        ORDER BY week_num
    '''
    df_weekly = pd.read_sql(sql_weekly, conn)
    print(df_weekly.to_string(index=False))

    df_daily.to_csv('daily_stats.csv', index=False, encoding='utf-8-sig')
    df_weekly.to_csv('weekly_stats.csv', index=False, encoding='utf-8-sig')
    print("\n✓ 日均/周均统计已保存：daily_stats.csv, weekly_stats.csv")

    return df_daily, df_weekly


# ============================================================
# 第四步：SQL异常数据检测
# ============================================================
def sql_anomaly_detection(conn):
    """使用SQL进行异常数据检测"""
    print("\n" + "=" * 60)
    print("第四步：SQL异常数据检测")
    print("=" * 60)

    # 方法1：阈值法（SQL WHERE 条件）
    print("\n--- 方法1：阈值法（SQL: WHERE 条件筛选） ---")
    thresholds = {
        'temperature': {'min': 10, 'max': 35, 'name': '温度(℃)'},
        'humidity': {'min': 30, 'max': 90, 'name': '湿度(%)'},
        'light': {'min': 0, 'max': 60000, 'name': '光照(lux)'},
        'co2': {'min': 300, 'max': 1000, 'name': 'CO₂(ppm)'}
    }

    for col, th in thresholds.items():
        sql = f'''
            SELECT COUNT(*) AS anomaly_count
            FROM sensor_data
            WHERE {col} < {th['min']} OR {col} > {th['max']}
        '''
        count = pd.read_sql(sql, conn).iloc[0, 0]
        print(f"  {th['name']}：适宜范围 [{th['min']}, {th['max']}]，异常 {count} 条")

    # 综合阈值异常
    sql_threshold = '''
        SELECT COUNT(*) AS anomaly_count
        FROM sensor_data
        WHERE temperature < 10 OR temperature > 35
           OR humidity < 30 OR humidity > 90
           OR light > 60000
           OR co2 < 300 OR co2 > 1000
    '''
    threshold_count = pd.read_sql(sql_threshold, conn).iloc[0, 0]
    print(f"\n  阈值法综合异常：{threshold_count} 条 ({threshold_count/2160:.1%})")

    # 方法2：统计法（SQL子查询计算均值标准差）
    print("\n--- 方法2：统计法（SQL: 子查询计算均值±3σ） ---")
    sql_statistical = '''
        SELECT COUNT(*) AS anomaly_count
        FROM sensor_data s
        JOIN (
            SELECT
                AVG(temperature) AS temp_mean,
                AVG(temperature) + 3 * STDEV(temperature) AS temp_upper,
                AVG(temperature) - 3 * STDEV(temperature) AS temp_lower,
                AVG(humidity) AS hum_mean,
                AVG(humidity) + 3 * STDEV(humidity) AS hum_upper,
                AVG(humidity) - 3 * STDEV(humidity) AS hum_lower
            FROM sensor_data
        ) stats
        WHERE s.temperature < stats.temp_lower OR s.temperature > stats.temp_upper
           OR s.humidity < stats.hum_lower OR s.humidity > stats.hum_upper
    '''
    try:
        stat_count = pd.read_sql(sql_statistical, conn).iloc[0, 0]
        print(f"  统计法（3σ）异常：{stat_count} 条")
    except:
        # SQLite的STDEV函数名可能不同
        sql_statistical2 = '''
            SELECT COUNT(*) AS anomaly_count
            FROM sensor_data
            WHERE ABS(temperature - (SELECT AVG(temperature) FROM sensor_data)) >
                  3 * (SELECT AVG((temperature - (SELECT AVG(temperature) FROM sensor_data)) *
                       (temperature - (SELECT AVG(temperature) FROM sensor_data))) FROM sensor_data)
        '''
        try:
            stat_count = pd.read_sql(sql_statistical2, conn).iloc[0, 0]
            print(f"  统计法（3σ）异常：{stat_count} 条")
        except:
            print("  统计法：SQLite标准差函数兼容性问题，改用pandas计算")
            df = pd.read_sql('SELECT * FROM sensor_data', conn)
            temp_mean, temp_std = df['temperature'].mean(), df['temperature'].std()
            stat_count = len(df[(df['temperature'] < temp_mean - 3*temp_std) | (df['temperature'] > temp_mean + 3*temp_std)])
            print(f"  统计法（3σ）异常：{stat_count} 条")

    # 更新数据库中的异常标记
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE sensor_data
        SET is_anomaly = 1
        WHERE temperature < 10 OR temperature > 35
           OR humidity < 30 OR humidity > 90
           OR light > 60000
           OR co2 < 300 OR co2 > 1000
    ''')
    conn.commit()
    updated = cursor.execute('SELECT COUNT(*) FROM sensor_data WHERE is_anomaly = 1').fetchone()[0]
    print(f"\n✓ 已更新数据库异常标记：{updated} 条标记为异常")

    # 查询异常数据示例
    print("\n异常数据示例（SQL查询前5条）：")
    df_anomaly = pd.read_sql('''
        SELECT timestamp, temperature, humidity, light, co2
        FROM sensor_data
        WHERE is_anomaly = 1
        ORDER BY timestamp
        LIMIT 5
    ''', conn)
    print(df_anomaly.to_string(index=False))

    # 可视化
    df_all = pd.read_sql('SELECT * FROM sensor_data', conn)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [('temperature', '温度(℃)', '#ED7D31'), ('humidity', '湿度(%)', '#5B9BD5'),
               ('light', '光照(lux)', '#FFC000'), ('co2', 'CO₂(ppm)', '#70AD47')]

    for idx, (col, name, color) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        normal = df_all[df_all['is_anomaly'] == 0]
        anomaly = df_all[df_all['is_anomaly'] == 1]
        ax.scatter(normal.index, normal[col], s=1, c=color, alpha=0.5, label='正常')
        ax.scatter(anomaly.index, anomaly[col], s=10, c='red', label='异常')
        ax.set_title(f'{name}异常检测')
        ax.set_xlabel('时间索引')
        ax.set_ylabel(name)
        ax.legend()

    plt.tight_layout()
    plt.savefig('anomaly_detection.png', dpi=150)
    print("\n✓ 异常检测图已保存：anomaly_detection.png")

    return df_all


# ============================================================
# 第五步：趋势分析与相关性分析
# ============================================================
def trend_and_correlation(conn, df_all):
    """趋势分析与相关性分析"""
    print("\n" + "=" * 60)
    print("第五步：趋势分析与相关性分析")
    print("=" * 60)

    # 日周期趋势（SQL GROUP BY hour）
    print("\n--- 日周期趋势（SQL: GROUP BY hour） ---")
    sql_hourly = '''
        SELECT
            hour,
            ROUND(AVG(temperature), 2) AS temp_avg,
            ROUND(AVG(humidity), 2) AS humidity_avg,
            ROUND(AVG(light), 0) AS light_avg,
            ROUND(AVG(co2), 2) AS co2_avg
        FROM sensor_data
        GROUP BY hour
        ORDER BY hour
    '''
    df_hourly = pd.read_sql(sql_hourly, conn)
    print(df_hourly.to_string(index=False))

    # 相关性分析
    print("\n--- 相关性分析（皮尔逊相关系数） ---")
    corr_matrix = df_all[['temperature', 'humidity', 'light', 'co2']].corr()
    print(corr_matrix.round(3).to_string())

    temp_light_corr = corr_matrix.loc['temperature', 'light']
    temp_humidity_corr = corr_matrix.loc['temperature', 'humidity']
    print(f"\n【关键发现】")
    print(f"1. 温度与光照：相关系数 = {temp_light_corr:.3f}（强正相关）")
    print(f"   原因：白天光照强时温度也高，晚上光照为0温度也低")
    print(f"2. 温度与湿度：相关系数 = {temp_humidity_corr:.3f}（负相关）")
    print(f"   原因：温度高时水分蒸发快，空气湿度低")
    print(f"3. 核心因子：温度是影响作物生长最核心的环境因子")

    # 可视化
    fig = plt.figure(figsize=(16, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df_hourly['hour'], df_hourly['temp_avg'], 'o-', color='#ED7D31', label='温度')
    ax1.set_xlabel('小时')
    ax1.set_ylabel('温度(℃)', color='#ED7D31')
    ax1.tick_params(axis='y', labelcolor='#ED7D31')
    ax1.set_title('温度日周期趋势')
    ax1.set_xticks(range(0, 24, 3))
    ax1b = ax1.twinx()
    ax1b.plot(df_hourly['hour'], df_hourly['light_avg'], 's-', color='#FFC000', label='光照')
    ax1b.set_ylabel('光照(lux)', color='#FFC000')
    ax1b.tick_params(axis='y', labelcolor='#FFC000')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df_hourly['hour'], df_hourly['humidity_avg'], 'o-', color='#5B9BD5')
    ax2.set_title('湿度日周期趋势')
    ax2.set_xlabel('小时')
    ax2.set_ylabel('湿度(%)')
    ax2.set_xticks(range(0, 24, 3))

    ax3 = fig.add_subplot(2, 2, 3)
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=['温度', '湿度', '光照', 'CO₂'],
                yticklabels=['温度', '湿度', '光照', 'CO₂'], ax=ax3)
    ax3.set_title('环境指标相关性热力图')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df_all['temperature'], df_all['humidity'], s=5, alpha=0.3, c='#5B9BD5')
    ax4.set_xlabel('温度(℃)')
    ax4.set_ylabel('湿度(%)')
    ax4.set_title(f'温度 vs 湿度（r={temp_humidity_corr:.3f}）')

    plt.tight_layout()
    plt.savefig('trend_correlation.png', dpi=150)
    print("\n✓ 趋势与相关性分析图已保存：trend_correlation.png")

    return corr_matrix


# ============================================================
# 第六步：异常预警规则与调控方案
# ============================================================
def warning_rules():
    """异常预警规则与分时段调控方案"""
    print("\n" + "=" * 60)
    print("第六步：异常预警规则与调控方案")
    print("=" * 60)

    print("""
【三级预警规则】

┌────────┬──────────────────────────────────────┬──────────────────────┐
│ 预警级别 │ 触发条件                              │ 处理措施              │
├────────┼──────────────────────────────────────┼──────────────────────┤
│ 黄色预警 │ 温度超出适宜范围(15-30℃)但不严重，    │ 自动记录，管理员查看  │
│        │ 或湿度异常，或单一指标短时异常          │                      │
├────────┼──────────────────────────────────────┼──────────────────────┤
│ 橙色预警 │ 温度低于10℃或高于35℃，或连续3小时     │ 通知管理员，启动设备  │
│        │ 指标异常，或两个指标同时异常            │                      │
├────────┼──────────────────────────────────────┼──────────────────────┤
│ 红色预警 │ 温度低于5℃或高于40℃，或多个指标       │ 自动启动温控设备，    │
│        │ 同时严重异常，或传感器故障              │ 短信通知管理员        │
└────────┴──────────────────────────────────────┴──────────────────────┘
""")

    print("""
【分时段调控方案】

凌晨（0:00-6:00）：
  - 温度低，关闭通风口，开启保温设备
  - 湿度可能升高，注意除湿
  - 光照为0，无需处理

上午（6:00-12:00）：
  - 温度逐渐升高，适当通风
  - 光照逐渐增强，过强时拉遮阳网
  - CO₂浓度较高，可适当通风

下午（12:00-18:00）：
  - 温度最高，加强通风降温
  - 中午光照最强，拉遮阳网
  - 湿度最低，可适当喷雾增湿

傍晚（18:00-24:00）：
  - 温度下降，关闭通风口保温
  - 光照减弱，收起遮阳网
  - CO₂浓度开始回升
""")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("智能大棚环境监测数据分析系统")
    print("作者：冯鹏 | 课程：数据库课程设计")
    print("数据库：SQLite（greenhouse.db）")
    print("=" * 60)

    # 第一步：创建数据库和表
    conn = create_database()

    # 第二步：生成并插入数据
    total_count = generate_and_insert_data(conn)

    # 第三步：SQL数据统计
    df_daily, df_weekly = sql_data_statistics(conn)

    # 第四步：SQL异常检测
    df_all = sql_anomaly_detection(conn)

    # 第五步：趋势与相关性分析
    corr_matrix = trend_and_correlation(conn, df_all)

    # 第六步：预警规则与调控方案
    warning_rules()

    # 导出完整数据
    df_all.to_csv('greenhouse_data.csv', index=False, encoding='utf-8-sig')

    conn.close()

    print("\n" + "=" * 60)
    print("分析完成！生成的文件：")
    print("  1. greenhouse.db - SQLite数据库文件")
    print("  2. greenhouse_data.csv - 完整传感器数据（含异常标记）")
    print("  3. daily_stats.csv - 日均统计（SQL查询结果）")
    print("  4. weekly_stats.csv - 周均统计（SQL查询结果）")
    print("  5. anomaly_detection.png - 异常检测图")
    print("  6. trend_correlation.png - 趋势与相关性分析图")
    print("=" * 60)


if __name__ == '__main__':
    main()
