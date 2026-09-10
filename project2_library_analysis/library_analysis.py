# -*- coding: utf-8 -*-
"""
项目2：校园图书借阅运营数据分析系统
====================================
功能：
1. 连接SQLite数据库，创建三张核心表（用户表、图书表、借阅流水表）
2. 生成模拟数据并插入数据库（8000+条）
3. 使用SQL进行多表联查、分组统计等数据查询
4. 多维度分析（用户年级、图书品类、借阅时段）
5. 核心指标计算（图书流通率、用户活跃度）
6. 业务结论与优化建议
7. 可视化图表

运行方式：
    pip install pandas numpy matplotlib
    python library_analysis.py

数据库说明：
    - 使用SQLite（Python自带，无需额外安装）
    - 数据库文件：library.db
    - 三张表：user、book、borrow_record
    - SQL语法与MySQL基本一致，可直接迁移到MySQL

作者：冯鹏
课程：软件工程课程设计
时间：2025.09 - 2025.12
"""

import pandas as pd
import numpy as np
import random
import sqlite3
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DB_PATH = 'library.db'


# ============================================================
# 第一步：数据库设计与建表
# ============================================================
def create_database():
    """创建数据库和三张核心表"""
    print("=" * 60)
    print("第一步：数据库设计与建表")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 删除已存在的表（重新运行时清空）
    cursor.execute('DROP TABLE IF EXISTS borrow_record')
    cursor.execute('DROP TABLE IF EXISTS book')
    cursor.execute('DROP TABLE IF EXISTS user')

    # 1. 用户表
    cursor.execute('''
        CREATE TABLE user (
            user_id      VARCHAR(10) PRIMARY KEY,
            name         VARCHAR(50),
            grade        VARCHAR(10),
            major        VARCHAR(50),
            register_date DATE
        )
    ''')
    print("✓ 用户表 user 创建成功")

    # 2. 图书表
    cursor.execute('''
        CREATE TABLE book (
            book_id      VARCHAR(10) PRIMARY KEY,
            title        VARCHAR(200),
            author       VARCHAR(50),
            category     VARCHAR(20),
            publisher    VARCHAR(100),
            publish_date DATE,
            stock        INT
        )
    ''')
    print("✓ 图书表 book 创建成功")

    # 3. 借阅流水表
    cursor.execute('''
        CREATE TABLE borrow_record (
            record_id    VARCHAR(10) PRIMARY KEY,
            user_id      VARCHAR(10),
            book_id      VARCHAR(10),
            borrow_time  DATETIME,
            return_time  DATETIME,
            borrow_days  INT,
            is_overdue   TINYINT,
            FOREIGN KEY (user_id) REFERENCES user(user_id),
            FOREIGN KEY (book_id) REFERENCES book(book_id)
        )
    ''')
    print("✓ 借阅流水表 borrow_record 创建成功")

    # 表关系说明
    print("""
【表关系设计】
- user 1:N borrow_record（一个用户可以有多条借阅记录）
- book 1:N borrow_record（一本图书可以被多次借阅）
- borrow_record 通过 user_id 关联 user，通过 book_id 关联 book
""")

    conn.commit()
    return conn


# ============================================================
# 第二步：生成模拟数据并插入数据库
# ============================================================
def generate_and_insert_data(conn):
    """生成模拟数据并插入数据库"""
    print("=" * 60)
    print("第二步：生成模拟数据并插入数据库")
    print("=" * 60)

    np.random.seed(42)
    random.seed(42)
    cursor = conn.cursor()

    # ---- 1. 生成用户数据 ----
    grades = ['大一', '大二', '大三', '大四']
    majors = ['计算机', '机械', '电子', '经管', '外语', '材料', '自动化', '法学']

    users = []
    for i in range(1, 501):
        grade = np.random.choice(grades, p=[0.3, 0.25, 0.25, 0.2])
        users.append((
            f'U{i:04d}',
            f'用户{i}',
            grade,
            random.choice(majors),
            '2023-09-01'
        ))

    cursor.executemany('INSERT INTO user VALUES (?,?,?,?,?)', users)
    print(f"✓ 插入用户数据：{len(users)} 条")

    # ---- 2. 生成图书数据 ----
    categories = {
        '计算机': 200, '文学': 150, '历史': 100, '经济': 80,
        '哲学': 50, '农业': 40, '军事': 30, '艺术': 50
    }

    books = []
    book_id = 1
    for cat, count in categories.items():
        for i in range(count):
            books.append((
                f'B{book_id:05d}',
                f'{cat}类图书{i+1}',
                f'作者{random.randint(1, 100)}',
                cat,
                f'出版社{random.randint(1, 20)}',
                f'202{random.randint(0, 3)}-{random.randint(1,12):02d}-01',
                random.randint(1, 5)
            ))
            book_id += 1

    cursor.executemany('INSERT INTO book VALUES (?,?,?,?,?,?,?)', books)
    print(f"✓ 插入图书数据：{len(books)} 条，共 {len(categories)} 个品类")

    # ---- 3. 生成借阅流水数据 ----
    # 先从数据库读取用户和图书（模拟真实场景）
    df_users = pd.read_sql('SELECT * FROM user', conn)
    df_books = pd.read_sql('SELECT * FROM book', conn)

    grade_borrow_prob = {'大一': 0.35, '大二': 0.25, '大三': 0.15, '大四': 0.25}
    cat_weights = {'计算机': 0.25, '文学': 0.20, '历史': 0.15, '经济': 0.12,
                   '哲学': 0.05, '农业': 0.03, '军事': 0.05, '艺术': 0.15}

    records = []
    record_id = 1
    start_date = datetime(2024, 9, 1)
    end_date = datetime(2025, 6, 30)
    total_days = (end_date - start_date).days
    target_records = 8500

    while len(records) < target_records:
        user = df_users.sample(1).iloc[0]
        if random.random() > grade_borrow_prob[user['grade']]:
            continue

        cat = np.random.choice(list(cat_weights.keys()), p=list(cat_weights.values()))
        cat_books = df_books[df_books['category'] == cat]
        if len(cat_books) == 0:
            continue
        book = cat_books.sample(1).iloc[0]

        day_offset = random.randint(0, total_days)
        borrow_date = start_date + timedelta(days=day_offset)
        month = borrow_date.month
        if month in [12, 6] and random.random() > 0.5:
            continue

        borrow_days = random.randint(15, 60)
        return_date = borrow_date + timedelta(days=borrow_days)
        if return_date > end_date:
            return_time = None
            is_overdue = 0
        else:
            return_time = return_date.strftime('%Y-%m-%d %H:%M:%S')
            is_overdue = 1 if borrow_days > 30 else 0

        hour = np.random.choice([8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20],
                                p=[0.05, 0.08, 0.10, 0.08, 0.15, 0.18, 0.15, 0.10, 0.05, 0.03, 0.03])
        borrow_datetime = borrow_date.replace(hour=int(hour), minute=random.randint(0, 59))

        records.append((
            f'R{record_id:06d}',
            user['user_id'],
            book['book_id'],
            borrow_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            return_time,
            borrow_days,
            is_overdue
        ))
        record_id += 1

    cursor.executemany('INSERT INTO borrow_record VALUES (?,?,?,?,?,?,?)', records)
    conn.commit()
    print(f"✓ 插入借阅流水数据：{len(records)} 条")

    # 验证：用SQL查询数据量
    count = cursor.execute('SELECT COUNT(*) FROM borrow_record').fetchone()[0]
    print(f"\n数据库验证：借阅流水表共 {count} 条记录")

    return df_users, df_books


# ============================================================
# 第三步：使用SQL进行多维度分析
# ============================================================
def sql_multi_dimension_analysis(conn):
    """使用SQL进行多维度分析"""
    print("\n" + "=" * 60)
    print("第三步：使用SQL进行多维度分析")
    print("=" * 60)

    # ---- 维度1：用户年级分析（SQL多表联查+分组） ----
    print("\n--- 维度1：用户年级分析（SQL: JOIN + GROUP BY） ---")
    sql_grade = '''
        SELECT
            u.grade,
            COUNT(*) AS borrow_count,
            COUNT(DISTINCT r.user_id) AS user_count
        FROM borrow_record r
        JOIN user u ON r.user_id = u.user_id
        GROUP BY u.grade
        ORDER BY borrow_count DESC
    '''
    df_grade = pd.read_sql(sql_grade, conn)
    df_grade['borrow_rate'] = df_grade['borrow_count'] / df_grade['borrow_count'].sum()
    print(df_grade.to_string(index=False))

    # ---- 维度2：图书品类分析（SQL多表联查+分组） ----
    print("\n--- 维度2：图书品类分析（SQL: JOIN + GROUP BY） ---")
    sql_category = '''
        SELECT
            b.category,
            COUNT(*) AS borrow_count
        FROM borrow_record r
        JOIN book b ON r.book_id = b.book_id
        GROUP BY b.category
        ORDER BY borrow_count DESC
    '''
    df_category = pd.read_sql(sql_category, conn)
    df_category['borrow_rate'] = df_category['borrow_count'] / df_category['borrow_count'].sum()
    print(df_category.to_string(index=False))

    # ---- 维度3：借阅时段分析 ----
    print("\n--- 维度3：借阅时段分析 ---")
    sql_month = '''
        SELECT
            CAST(strftime('%m', borrow_time) AS INTEGER) AS borrow_month,
            COUNT(*) AS borrow_count
        FROM borrow_record
        GROUP BY borrow_month
        ORDER BY borrow_month
    '''
    df_month = pd.read_sql(sql_month, conn)
    print("按月统计：")
    print(df_month.to_string(index=False))

    # 按时段
    sql_period = '''
        SELECT
            CASE
                WHEN CAST(strftime('%H', borrow_time) AS INTEGER) BETWEEN 8 AND 11 THEN '上午(8-12点)'
                WHEN CAST(strftime('%H', borrow_time) AS INTEGER) BETWEEN 12 AND 13 THEN '中午(12-14点)'
                WHEN CAST(strftime('%H', borrow_time) AS INTEGER) BETWEEN 14 AND 17 THEN '下午(14-18点)'
                ELSE '晚上(18-21点)'
            END AS time_period,
            COUNT(*) AS borrow_count
        FROM borrow_record
        GROUP BY time_period
        ORDER BY borrow_count DESC
    '''
    df_period = pd.read_sql(sql_period, conn)
    df_period['rate'] = df_period['borrow_count'] / df_period['borrow_count'].sum()
    print("\n按时段统计：")
    print(df_period.to_string(index=False))

    # ---- 可视化 ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    grade_order = ['大一', '大二', '大三', '大四']
    df_grade['grade'] = pd.Categorical(df_grade['grade'], categories=grade_order, ordered=True)
    df_grade = df_grade.sort_values('grade')

    axes[0, 0].bar(df_grade['grade'].astype(str), df_grade['borrow_count'], color='#5B9BD5')
    axes[0, 0].set_title('各年级借阅量分布')
    axes[0, 0].set_ylabel('借阅次数')
    for i, v in enumerate(df_grade['borrow_count']):
        axes[0, 0].text(i, v + 50, f'{v}', ha='center')

    axes[0, 1].barh(df_category['category'], df_category['borrow_count'], color='#ED7D31')
    axes[0, 1].set_title('各品类借阅量分布')
    axes[0, 1].set_xlabel('借阅次数')
    axes[0, 1].invert_yaxis()

    axes[1, 0].plot(df_month['borrow_month'], df_month['borrow_count'], marker='o', color='#70AD47', linewidth=2)
    axes[1, 0].set_title('月度借阅量趋势')
    axes[1, 0].set_xlabel('月份')
    axes[1, 0].set_ylabel('借阅次数')
    axes[1, 0].set_xticks(range(1, 13))

    axes[1, 1].pie(df_period['borrow_count'], labels=df_period['time_period'],
                    autopct='%1.1f%%', colors=['#5B9BD5', '#ED7D31', '#70AD47', '#FFC000'])
    axes[1, 1].set_title('借阅时段分布')

    plt.tight_layout()
    plt.savefig('multi_dimension_analysis.png', dpi=150)
    print("\n✓ 多维度分析图已保存：multi_dimension_analysis.png")

    return df_grade, df_category, df_month, df_period


# ============================================================
# 第四步：核心指标计算（SQL子查询+聚合）
# ============================================================
def sql_core_metrics(conn, df_users):
    """使用SQL计算核心指标"""
    print("\n" + "=" * 60)
    print("第四步：核心指标计算（SQL子查询+聚合）")
    print("=" * 60)

    # ---- 指标1：图书流通率（SQL子查询） ----
    print("\n--- 指标1：图书流通率 ---")
    print("流通率 = 某品类借阅次数 / 该品类馆藏总数")
    sql_circulation = '''
        SELECT
            b.category,
            COUNT(r.record_id) AS borrow_count,
            cat_stock.total_stock,
            ROUND(CAST(COUNT(r.record_id) AS FLOAT) / cat_stock.total_stock, 2) AS circulation_rate
        FROM borrow_record r
        JOIN book b ON r.book_id = b.book_id
        JOIN (
            SELECT category, SUM(stock) AS total_stock
            FROM book
            GROUP BY category
        ) cat_stock ON b.category = cat_stock.category
        GROUP BY b.category
        ORDER BY circulation_rate DESC
    '''
    df_circulation = pd.read_sql(sql_circulation, conn)
    print(df_circulation.to_string(index=False))

    # ---- 指标2：用户活跃度（SQL CASE WHEN 分档） ----
    print("\n--- 指标2：用户活跃度 ---")
    print("活跃度分档：0次(不活跃)、1-3次(低活跃)、4-10次(中活跃)、10次以上(高活跃)")
    sql_activity = '''
        SELECT
            activity_level,
            COUNT(*) AS user_count,
            SUM(borrow_count) AS total_borrow
        FROM (
            SELECT
                u.user_id,
                COUNT(r.record_id) AS borrow_count,
                CASE
                    WHEN COUNT(r.record_id) = 0 THEN '不活跃(0次)'
                    WHEN COUNT(r.record_id) <= 3 THEN '低活跃(1-3次)'
                    WHEN COUNT(r.record_id) <= 10 THEN '中活跃(4-10次)'
                    ELSE '高活跃(10次以上)'
                END AS activity_level
            FROM user u
            LEFT JOIN borrow_record r ON u.user_id = r.user_id
            GROUP BY u.user_id
        )
        GROUP BY activity_level
        ORDER BY total_borrow DESC
    '''
    df_activity = pd.read_sql(sql_activity, conn)
    df_activity['user_rate'] = df_activity['user_count'] / df_activity['user_count'].sum()
    df_activity['borrow_rate'] = df_activity['total_borrow'] / df_activity['total_borrow'].sum()
    print(df_activity.to_string(index=False))

    # 二八定律验证
    sql_28 = '''
        SELECT
            ROUND(CAST(SUM(top20_borrow) AS FLOAT) / SUM(total_borrow) * 100, 1) AS top20_contribution
        FROM (
            SELECT
                SUM(borrow_count) AS total_borrow,
                SUM(CASE WHEN rn <= total_cnt * 0.2 THEN borrow_count ELSE 0 END) AS top20_borrow
            FROM (
                SELECT
                    borrow_count,
                    ROW_NUMBER() OVER (ORDER BY borrow_count DESC) AS rn,
                    COUNT(*) OVER () AS total_cnt
                FROM (
                    SELECT user_id, COUNT(*) AS borrow_count
                    FROM borrow_record
                    GROUP BY user_id
                )
            )
        )
    '''
    try:
        df_28 = pd.read_sql(sql_28, conn)
        print(f"\n二八定律验证：前20%用户贡献了 {df_28.iloc[0]['top20_contribution']}% 的借阅量")
    except:
        # SQLite窗口函数可能不支持，用pandas算
        user_borrow = pd.read_sql('SELECT user_id, COUNT(*) as cnt FROM borrow_record GROUP BY user_id', conn)
        user_borrow = user_borrow.sort_values('cnt', ascending=False)
        top20 = int(len(user_borrow) * 0.2)
        top20_borrow = user_borrow.head(top20)['cnt'].sum()
        total_borrow = user_borrow['cnt'].sum()
        print(f"\n二八定律验证：前20%用户贡献了 {top20_borrow/total_borrow:.1%} 的借阅量")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(df_circulation['category'], df_circulation['circulation_rate'], color='#5B9BD5')
    axes[0].set_title('各品类图书流通率')
    axes[0].set_xlabel('流通率（次/本）')
    axes[0].invert_yaxis()
    for i, v in enumerate(df_circulation['circulation_rate']):
        axes[0].text(v + 0.1, i, f'{v:.1f}', va='center')

    level_order = ['高活跃(10次以上)', '中活跃(4-10次)', '低活跃(1-3次)', '不活跃(0次)']
    df_activity['activity_level'] = pd.Categorical(df_activity['activity_level'], categories=level_order, ordered=True)
    df_activity = df_activity.sort_values('activity_level')

    x = np.arange(len(df_activity))
    width = 0.35
    axes[1].bar(x - width/2, df_activity['user_rate'], width, label='用户占比', color='#5B9BD5')
    axes[1].bar(x + width/2, df_activity['borrow_rate'], width, label='借阅量占比', color='#ED7D31')
    axes[1].set_ylabel('占比')
    axes[1].set_title('用户活跃度分布（二八定律）')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_activity['activity_level'].astype(str), rotation=15)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('core_metrics.png', dpi=150)
    print("\n✓ 核心指标图已保存：core_metrics.png")

    return df_circulation, df_activity


# ============================================================
# 第五步：业务结论与优化建议
# ============================================================
def business_insights(df_circulation, df_activity, df_grade):
    """业务结论与优化建议"""
    print("\n" + "=" * 60)
    print("第五步：业务结论与优化建议")
    print("=" * 60)

    print("\n【核心发现】")
    cold_cats = df_circulation[df_circulation['circulation_rate'] < 1]
    print(f"1. 冷门品类（流通率<1）：{', '.join(cold_cats['category'].tolist())}")
    hot_cats = df_circulation[df_circulation['circulation_rate'] > 5]
    print(f"2. 热门品类（流通率>5）：{', '.join(hot_cats['category'].tolist())}")
    junior_rate = df_grade[df_grade['grade'] == '大三']['borrow_rate'].values[0]
    print(f"3. 大三学生借阅量最低，仅占总量的 {junior_rate:.1%}")
    inactive_row = df_activity[df_activity['activity_level'] == '不活跃(0次)']
    if len(inactive_row) > 0:
        print(f"4. 不活跃用户占比：{inactive_row['user_rate'].values[0]:.1%}")

    print("\n【优化建议】")
    suggestions = [
        "建议1（馆藏优化）：减少农业、哲学等冷门品类的新书采购，将预算向计算机、文学等热门品类倾斜，增加热门品类复购；对冷门书做主题推荐和书架陈列优化，提高曝光率",
        "建议2（用户运营）：针对大三学生推出考研资料专区、专业必读书单推荐，精准触达高流失群体；推出借阅积分、打卡活动，提高用户活跃度，降低不活跃用户比例"
    ]
    for s in suggestions:
        print(f"  {s}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("校园图书借阅运营数据分析系统")
    print("作者：冯鹏 | 课程：软件工程课程设计")
    print("数据库：SQLite（library.db）")
    print("=" * 60)

    # 第一步：创建数据库和表
    conn = create_database()

    # 第二步：生成并插入数据
    df_users, df_books = generate_and_insert_data(conn)

    # 第三步：SQL多维度分析
    df_grade, df_category, df_month, df_period = sql_multi_dimension_analysis(conn)

    # 第四步：SQL核心指标
    df_circulation, df_activity = sql_core_metrics(conn, df_users)

    # 第五步：业务结论
    business_insights(df_circulation, df_activity, df_grade)

    # 导出查询结果
    df_grade.to_csv('grade_analysis.csv', index=False, encoding='utf-8-sig')
    df_category.to_csv('category_analysis.csv', index=False, encoding='utf-8-sig')
    df_circulation.to_csv('circulation_rate.csv', index=False, encoding='utf-8-sig')
    df_activity.to_csv('user_activity.csv', index=False, encoding='utf-8-sig')

    conn.close()

    print("\n" + "=" * 60)
    print("分析完成！生成的文件：")
    print("  1. library.db - SQLite数据库文件（含三张表）")
    print("  2. grade_analysis.csv - 年级分析结果")
    print("  3. category_analysis.csv - 品类分析结果")
    print("  4. circulation_rate.csv - 流通率计算结果")
    print("  5. user_activity.csv - 用户活跃度分析结果")
    print("  6. multi_dimension_analysis.png - 多维度分析图")
    print("  7. core_metrics.png - 核心指标图")
    print("=" * 60)


if __name__ == '__main__':
    main()
