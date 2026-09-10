# -*- coding: utf-8 -*-
"""
项目1：美食平台评论情感分析系统
================================
功能：
1. 连接SQLite数据库，创建评论数据表
2. 生成模拟美食评论数据并插入数据库（1.2万条）
3. 从数据库读取数据，进行清洗与预处理
4. 情感分类模型（朴素贝叶斯 vs 随机森林，对比准确率）
5. 多维度情感分析（口味、服务、环境三个维度）
6. 业务结论与优化建议
7. 可视化图表

运行方式：
    pip install pandas numpy scikit-learn jieba matplotlib
    python food_review_analysis.py

数据库说明：
    - 使用SQLite（Python自带，无需额外安装）
    - 数据库文件：food_review.db
    - 数据表：review（评论原始数据）、review_clean（清洗后数据）
    - SQL语法与MySQL基本一致

作者：冯鹏
课程：模式识别课程设计
时间：2026.03 - 2026.06
"""

import pandas as pd
import numpy as np
import re
import random
import sqlite3
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

DB_PATH = 'food_review.db'


# ============================================================
# 第一步：创建数据库和表
# ============================================================
def create_database():
    """创建数据库和评论表"""
    print("=" * 60)
    print("第一步：创建数据库和表")
    print("=" * 60)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 删除已存在的表
    cursor.execute('DROP TABLE IF EXISTS review')
    cursor.execute('DROP TABLE IF EXISTS review_clean')

    # 评论原始数据表
    cursor.execute('''
        CREATE TABLE review (
            review_id     INTEGER PRIMARY KEY,
            text          TEXT,
            rating        INT,
            label         INT,
            restaurant_id INT
        )
    ''')
    print("✓ 评论表 review 创建成功")

    # 清洗后数据表
    cursor.execute('''
        CREATE TABLE review_clean (
            review_id      INTEGER PRIMARY KEY,
            text           TEXT,
            text_clean     TEXT,
            text_tokenized TEXT,
            rating         INT,
            label          INT
        )
    ''')
    print("✓ 清洗后数据表 review_clean 创建成功")

    # 表结构说明
    print("""
【表结构设计】
review 表（原始评论数据）：
  - review_id     评论ID（主键）
  - text          评论文本
  - rating        评分（1-5星）
  - label         情感标签（0负面，1正面）
  - restaurant_id 餐厅ID

review_clean 表（清洗后数据）：
  - review_id      评论ID（主键）
  - text           原始文本
  - text_clean     清洗后文本
  - text_tokenized 分词后文本
  - rating         评分
  - label          情感标签
""")

    conn.commit()
    return conn


# ============================================================
# 第二步：生成模拟数据并插入数据库
# ============================================================
def generate_and_insert_data(conn, n_samples=12000):
    """生成模拟美食评论数据并插入数据库"""
    print("\n" + "=" * 60)
    print("第二步：生成模拟数据并插入数据库")
    print("=" * 60)

    random.seed(42)
    np.random.seed(42)

    positive_templates = [
        "这家店真的太好吃了，{taste}，{service}，{environment}，强烈推荐！",
        "今天和朋友来吃，{taste}，{service}，{environment}，下次还来！",
        "慕名而来，果然没让我失望，{taste}，{service}，{environment}，五星好评！",
        "{taste}，而且{service}，{environment}，性价比很高，会再来的。",
        "第一次来这家店，{taste}，{service}，{environment}，体验非常好！",
        "朋友推荐来的，{taste}，{service}，{environment}，果然名不虚传！",
        "周末和家人来聚餐，{taste}，{service}，{environment}，大家都很满意！",
        "已经是第三次来了，{taste}，{service}，{environment}，每次都有惊喜！",
        "公司团建选的这里，{taste}，{service}，{environment}，同事们都说好！",
        "路过随便进来的，没想到{experience}，{taste}，{service}，{environment}，意外之喜！",
    ]

    negative_templates = [
        "这家店太让人失望了，{taste}，{service}，{environment}，再也不来了！",
        "等了半天终于吃上，结果{taste}，服务员{service}，环境{environment}，差评！",
        "{taste}，而且{service}，{environment}，完全不值这个价，后悔来了。",
        "再也不会来了，{taste}，{service}，{environment}，体验很差！",
        "踩雷了，{taste}，{service}，{environment}，不推荐大家来。",
        "网上评价那么好，实际{taste}，{service}，{environment}，名不副实！",
        "等了一个小时才上菜，{taste}，{service}，{environment}，太影响心情了！",
        "团购来的，结果{taste}，{service}，{environment}，区别对待太明显了！",
        "朋友说好吃才来的，没想到{taste}，{service}，{environment}，太失望了！",
        "环境看起来不错，实际{taste}，{service}，{environment}，中看不中用！",
    ]

    taste_pos = [
        "菜品非常新鲜", "味道超级棒", "招牌菜名不虚传", "口味很地道", "每道菜都很好吃",
        "食材很新鲜", "调味恰到好处", "色香味俱全", "口感很丰富", "分量很足",
        "菜品很精致", "味道很正宗", "厨师水平很高", "创意菜很有特色", "汤底很浓郁"
    ]
    taste_neg = [
        "菜品不新鲜", "味道太咸了", "招牌菜也就那样", "口味一般般", "菜很难吃",
        "食材不新鲜", "调味太重了", "卖相很差", "口感很单一", "分量很少",
        "菜品很粗糙", "味道不正宗", "厨师水平一般", "创意菜很奇怪", "汤底很寡淡"
    ]

    service_pos = [
        "服务员态度特别好", "上菜速度很快", "服务很周到", "服务员很热情", "响应很及时",
        "有求必应", "主动帮忙换骨碟", "介绍菜品很详细", "服务很专业", "笑容很亲切"
    ]
    service_neg = [
        "服务员态度很差", "上菜速度特别慢", "服务很不周到", "服务员爱答不理", "叫半天没人理",
        "没人管", "骨碟堆成山也不换", "问什么都不知道", "服务很不专业", "全程黑脸"
    ]

    env_pos = [
        "环境很干净", "装修很有特色", "氛围很好", "空间很大", "很安静适合聊天",
        "灯光很温馨", "音乐很好听", "座位很舒服", "卫生做得很好", "很有格调"
    ]
    env_neg = [
        "环境很脏", "装修很老旧", "氛围很差", "空间很小很挤", "太吵了",
        "灯光很昏暗", "音乐很吵", "座位很硬", "卫生做得很差", "很没格调"
    ]

    experience_pos = ["体验超出预期", "整体感觉很棒", "各方面都不错", "性价比超高"]
    experience_neg = ["体验低于预期", "整体感觉很差", "各方面都不行", "性价比超低"]

    prefixes = ["", "", "", "说实话，", "讲真，", "不得不说，", "客观评价，", "个人觉得，"]
    suffixes = ["", "", "", "！", "。", "～", "👍", "下次还会再来的！", "推荐给大家！"]

    data = []
    for i in range(n_samples):
        if random.random() < 0.55:
            template = random.choice(positive_templates)
            if '{experience}' in template:
                text = template.format(
                    experience=random.choice(experience_pos),
                    taste=random.choice(taste_pos),
                    service=random.choice(service_pos),
                    environment=random.choice(env_pos)
                )
            else:
                text = template.format(
                    taste=random.choice(taste_pos),
                    service=random.choice(service_pos),
                    environment=random.choice(env_pos)
                )
            label = 1
            rating = random.randint(4, 5)
            if random.random() < 0.1:
                text = text + " 就是" + random.choice(["价格稍贵", "人有点多", "位置不太好找"]) + "。"
        else:
            template = random.choice(negative_templates)
            text = template.format(
                taste=random.choice(taste_neg),
                service=random.choice(service_neg),
                environment=random.choice(env_neg)
            )
            label = 0
            rating = random.randint(1, 3)
            if random.random() < 0.1:
                text = text + " 不过" + random.choice(["环境还可以", "位置挺好找的", "价格倒是不贵"]) + "。"

        text = random.choice(prefixes) + text + random.choice(suffixes)

        if random.random() < 0.05:
            text = "<p>" + text + "</p>"
        if random.random() < 0.03:
            text = ""

        data.append((i + 1, text, rating, label, random.randint(1, 50)))

    # 批量插入数据库
    cursor = conn.cursor()
    cursor.executemany('INSERT INTO review VALUES (?,?,?,?,?)', data)
    conn.commit()

    # 验证：SQL查询数据量
    count = cursor.execute('SELECT COUNT(*) FROM review').fetchone()[0]
    pos_count = cursor.execute('SELECT COUNT(*) FROM review WHERE label = 1').fetchone()[0]
    neg_count = cursor.execute('SELECT COUNT(*) FROM review WHERE label = 0').fetchone()[0]
    print(f"✓ 插入数据：{count} 条")
    print(f"  正面评论：{pos_count} 条 ({pos_count/count*100:.1f}%)")
    print(f"  负面评论：{neg_count} 条 ({neg_count/count*100:.1f}%)")

    # SQL查询示例
    print("\n数据预览（SQL查询前5条）：")
    df_preview = pd.read_sql('SELECT review_id, rating, label, substr(text,1,30) as text_preview FROM review LIMIT 5', conn)
    print(df_preview.to_string(index=False))

    return count


# ============================================================
# 第三步：从数据库读取数据并清洗
# ============================================================
def clean_data_from_db(conn):
    """从数据库读取数据并清洗"""
    print("\n" + "=" * 60)
    print("第三步：从数据库读取数据并清洗")
    print("=" * 60)

    # 从数据库读取原始数据
    df = pd.read_sql('SELECT * FROM review', conn)
    original_count = len(df)
    print(f"从数据库读取：{original_count} 条")

    # 1. 删除空评论
    df = df[df['text'].str.strip() != '']
    print(f"删除空评论后：{len(df)} 条（删除 {original_count - len(df)} 条）")

    # 2. 删除重复评论
    df = df.drop_duplicates(subset=['text'])
    print(f"删除重复评论后：{len(df)} 条")

    # 3. 去除HTML标签
    df['text_clean'] = df['text'].apply(lambda x: re.sub(r'<[^>]+>', '', x))

    # 4. 去除特殊符号和表情
    df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', x))

    # 5. 去除多余空格
    df['text_clean'] = df['text_clean'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # 6. jieba中文分词
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
                 '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
                 '自己', '这', '那', '他', '她', '它', '们', '这个', '那个', '什么', '怎么',
                 '为什么', '哪', '哪里', '谁', '多少', '几', '啊', '呀', '吧', '呢', '嘛', '哦'}

    def tokenize(text):
        words = jieba.lcut(text)
        words = [w for w in words if w not in stopwords and len(w) > 1]
        return ' '.join(words)

    df['text_tokenized'] = df['text_clean'].apply(tokenize)

    print(f"清洗完成，最终数据量：{len(df)} 条")
    print(f"数据清洗示例：")
    print(f"  原始：{df.iloc[0]['text'][:50]}...")
    print(f"  清洗：{df.iloc[0]['text_clean'][:50]}...")
    print(f"  分词：{df.iloc[0]['text_tokenized'][:50]}...")

    # 将清洗后的数据存入数据库
    cursor = conn.cursor()
    cursor.execute('DELETE FROM review_clean')
    clean_data = [(row['review_id'], row['text'], row['text_clean'], row['text_tokenized'],
                    row['rating'], row['label']) for _, row in df.iterrows()]
    cursor.executemany('INSERT INTO review_clean VALUES (?,?,?,?,?,?)', clean_data)
    conn.commit()
    print(f"\n✓ 清洗后数据已存入数据库 review_clean 表，共 {len(clean_data)} 条")

    return df


# ============================================================
# 第四步：情感分类模型
# ============================================================
def train_model(df):
    """训练情感分类模型"""
    print("\n" + "=" * 60)
    print("第四步：情感分类模型训练")
    print("=" * 60)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text_tokenized'])
    y = df['label']

    print(f"特征维度：{X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"训练集：{X_train.shape[0]} 条，测试集：{X_test.shape[0]} 条")

    results = {}

    print("\n--- 模型1：朴素贝叶斯 ---")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    results['朴素贝叶斯'] = {
        'accuracy': accuracy_score(y_test, y_pred_nb),
        'precision': precision_score(y_test, y_pred_nb),
        'recall': recall_score(y_test, y_pred_nb),
        'f1': f1_score(y_test, y_pred_nb)
    }
    print(f"准确率：{results['朴素贝叶斯']['accuracy']:.4f}")
    print(f"精确率：{results['朴素贝叶斯']['precision']:.4f}")
    print(f"召回率：{results['朴素贝叶斯']['recall']:.4f}")
    print(f"F1值：{results['朴素贝叶斯']['f1']:.4f}")

    print("\n--- 模型2：随机森林 ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results['随机森林'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf)
    }
    print(f"准确率：{results['随机森林']['accuracy']:.4f}")
    print(f"精确率：{results['随机森林']['precision']:.4f}")
    print(f"召回率：{results['随机森林']['recall']:.4f}")
    print(f"F1值：{results['随机森林']['f1']:.4f}")

    best_model = rf
    print(f"\n最优模型：随机森林，准确率 {results['随机森林']['accuracy']:.2%}")

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, [results['朴素贝叶斯'][m] for m in metrics], width, label='朴素贝叶斯', color='#5B9BD5')
    ax.bar(x + width/2, [results['随机森林'][m] for m in metrics], width, label='随机森林', color='#ED7D31')
    ax.set_ylabel('分数')
    ax.set_title('模型性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(['准确率', '精确率', '召回率', 'F1值'])
    ax.legend()
    ax.set_ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
    print("模型对比图已保存：model_comparison.png")

    return best_model, vectorizer, results


# ============================================================
# 第五步：多维度情感分析
# ============================================================
def dimension_analysis(df, model, vectorizer):
    """多维度情感分析（口味、服务、环境）"""
    print("\n" + "=" * 60)
    print("第五步：多维度情感分析")
    print("=" * 60)

    dimension_keywords = {
        '口味': ['好吃', '难吃', '咸', '淡', '新鲜', '味道', '菜品', '菜', '调味', '食材', '招牌', '口味', '地道'],
        '服务': ['服务员', '服务', '态度', '热情', '慢', '快', '排队', '响应', '周到', '上菜', '叫', '理'],
        '环境': ['环境', '装修', '干净', '脏', '嘈杂', '安静', '空间', '氛围', '特色', '老旧', '挤']
    }

    dimension_results = {}

    for dim, keywords in dimension_keywords.items():
        mask = df['text_clean'].apply(lambda x: any(kw in x for kw in keywords))
        dim_df = df[mask].copy()

        if len(dim_df) == 0:
            continue

        X_dim = vectorizer.transform(dim_df['text_tokenized'])
        dim_df['predicted_label'] = model.predict(X_dim)

        pos_count = len(dim_df[dim_df['predicted_label'] == 1])
        neg_count = len(dim_df[dim_df['predicted_label'] == 0])
        total = len(dim_df)
        pos_rate = pos_count / total
        neg_rate = neg_count / total

        dimension_results[dim] = {
            'total': total,
            'positive': pos_count,
            'negative': neg_count,
            'pos_rate': pos_rate,
            'neg_rate': neg_rate
        }

        print(f"\n{dim}维度：")
        print(f"  相关评论：{total} 条")
        print(f"  正面：{pos_count} 条 ({pos_rate:.1%})")
        print(f"  负面：{neg_count} 条 ({neg_rate:.1%})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (dim, res) in enumerate(dimension_results.items()):
        axes[i].pie([res['positive'], res['negative']],
                    labels=['正面', '负面'],
                    colors=['#70AD47', '#C00000'],
                    autopct='%1.1f%%',
                    startangle=90)
        axes[i].set_title(f'{dim}维度情感分布\n（共{res["total"]}条）')
    plt.tight_layout()
    plt.savefig('dimension_sentiment.png', dpi=150)
    print("\n维度情感分布图已保存：dimension_sentiment.png")

    return dimension_results


# ============================================================
# 第六步：业务结论与优化建议
# ============================================================
def business_insights(dimension_results):
    """输出业务结论与优化建议"""
    print("\n" + "=" * 60)
    print("第六步：业务结论与优化建议")
    print("=" * 60)

    worst_dim = max(dimension_results.items(), key=lambda x: x[1]['neg_rate'])
    print(f"\n【核心发现】")
    print(f"1. 负面评价占比最高的维度是：{worst_dim[0]}（负面率 {worst_dim[1]['neg_rate']:.1%}）")

    sorted_dims = sorted(dimension_results.items(), key=lambda x: x[1]['neg_rate'], reverse=True)
    print(f"2. 各维度负面率排序：")
    for i, (dim, res) in enumerate(sorted_dims, 1):
        print(f"   {i}. {dim}：{res['neg_rate']:.1%}")

    print(f"\n【优化建议】")
    suggestions = [
        "建议1（服务优化）：优化排队叫号系统，高峰时段增加服务人员，加强服务员培训，提高响应速度",
        "建议2（口味优化）：后厨统一调味标准，提供口味清淡/偏重选项，定期收集顾客口味反馈",
        "建议3（环境优化）：优化座位布局，增加隔音措施，设置安静用餐区，改善高峰期嘈杂问题"
    ]
    for s in suggestions:
        print(f"  {s}")

    fig, ax = plt.subplots(figsize=(8, 5))
    dims = list(dimension_results.keys())
    neg_rates = [dimension_results[d]['neg_rate'] for d in dims]
    colors = ['#C00000' if r == max(neg_rates) else '#ED7D31' for r in neg_rates]
    ax.bar(dims, neg_rates, color=colors)
    ax.set_ylabel('负面率')
    ax.set_title('各维度负面评价占比')
    ax.set_ylim(0, 0.6)
    for i, v in enumerate(neg_rates):
        ax.text(i, v + 0.01, f'{v:.1%}', ha='center')
    plt.tight_layout()
    plt.savefig('negative_rate.png', dpi=150)
    print("\n负面率柱状图已保存：negative_rate.png")


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "=" * 60)
    print("美食平台评论情感分析系统")
    print("作者：冯鹏 | 课程：模式识别课程设计")
    print("数据库：SQLite（food_review.db）")
    print("=" * 60)

    # 第一步：创建数据库和表
    conn = create_database()

    # 第二步：生成数据并插入数据库
    total_count = generate_and_insert_data(conn, n_samples=12000)

    # 第三步：从数据库读取并清洗
    df_clean = clean_data_from_db(conn)

    # 第四步：训练模型
    best_model, vectorizer, model_results = train_model(df_clean)

    # 第五步：多维度分析
    dim_results = dimension_analysis(df_clean, best_model, vectorizer)

    # 第六步：业务结论
    business_insights(dim_results)

    # 导出CSV
    df_clean.to_csv('cleaned_reviews.csv', index=False, encoding='utf-8-sig')
    print("\n清洗后的数据已导出：cleaned_reviews.csv")

    conn.close()

    print("\n" + "=" * 60)
    print("分析完成！生成的文件：")
    print("  1. food_review.db - SQLite数据库文件（含review和review_clean两张表）")
    print("  2. cleaned_reviews.csv - 清洗后的评论数据")
    print("  3. model_comparison.png - 模型性能对比图")
    print("  4. dimension_sentiment.png - 维度情感分布图")
    print("  5. negative_rate.png - 各维度负面率柱状图")
    print("=" * 60)


if __name__ == '__main__':
    main()
