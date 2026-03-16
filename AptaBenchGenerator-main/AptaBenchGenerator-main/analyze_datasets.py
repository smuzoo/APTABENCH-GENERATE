import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import Counter

# Определение функций метрик
def gc_content(seq: str) -> float:
    seq = seq.upper()
    if not seq:
        return 0.0
    g = seq.count("G")
    c = seq.count("C")
    return (g + c) / len(seq)

def longest_homopolymer(seq: str) -> int:
    if not seq:
        return 0
    max_run = 1
    run = 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            run += 1
        else:
            max_run = max(max_run, run)
            run = 1
    return max(max_run, run)

def shannon_entropy(seq: str) -> float:
    counts = Counter(seq.upper())
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

# Загрузка данных
train_df = pd.read_csv('aptamer_model/data/AptaBench_dataset_v2.csv')
gen_df = pd.read_csv('all_generated_aptamers.csv')  # Измените на 'all_generated_aptamers_v2.csv' для prompt v2

# Вычисление метрик для тренировочного датасета
train_df['length'] = train_df['sequence'].str.len()
train_df['gc_content'] = train_df['sequence'].apply(gc_content)
train_df['entropy'] = train_df['sequence'].apply(shannon_entropy)
train_df['longest_homopolymer'] = train_df['sequence'].apply(longest_homopolymer)

# Вычисление метрик для сгенерированных аптамеров
gen_df['length'] = gen_df['sequence'].str.len()
gen_df['gc_content'] = gen_df['sequence'].apply(gc_content)
gen_df['entropy'] = gen_df['sequence'].apply(shannon_entropy)
gen_df['longest_homopolymer'] = gen_df['sequence'].apply(longest_homopolymer)

# Построение графиков
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Распределение proba (для генерированных) и pKd_value (для тренировочных)
sns.histplot(gen_df['proba'], ax=axes[0,0], label='Generated Proba', alpha=0.5, color='red')
sns.histplot(train_df['pKd_value'], ax=axes[0,0], label='Train pKd_value', alpha=0.5, color='blue')
axes[0,0].set_title('Proba / pKd Distribution')
axes[0,0].legend()

# Распределение label
sns.countplot(x='label', data=train_df, ax=axes[0,1], alpha=0.5, color='blue', label='Train')
sns.countplot(x='label', data=gen_df, ax=axes[0,1], alpha=0.5, color='red', label='Generated')
axes[0,1].set_title('Label Distribution')
axes[0,1].legend()

# Распределение энтропии
sns.histplot(train_df['entropy'], ax=axes[0,2], label='Train', alpha=0.5, color='blue')
sns.histplot(gen_df['entropy'], ax=axes[0,2], label='Generated', alpha=0.5, color='red')
axes[0,2].set_title('Entropy Distribution')
axes[0,2].legend()

# Распределение GC-содержания
sns.histplot(train_df['gc_content'], ax=axes[1,0], label='Train', alpha=0.5, color='blue')
sns.histplot(gen_df['gc_content'], ax=axes[1,0], label='Generated', alpha=0.5, color='red')
axes[1,0].set_title('GC Content Distribution')
axes[1,0].legend()

# Распределение длины
sns.histplot(train_df['length'], ax=axes[1,1], label='Train', alpha=0.5, color='blue')
sns.histplot(gen_df['length'], ax=axes[1,1], label='Generated', alpha=0.5, color='red')
axes[1,1].set_title('Length Distribution')
axes[1,1].legend()

# Распределение longest homopolymer
sns.histplot(train_df['longest_homopolymer'], ax=axes[1,2], label='Train', alpha=0.5, color='blue')
sns.histplot(gen_df['longest_homopolymer'], ax=axes[1,2], label='Generated', alpha=0.5, color='red')
axes[1,2].set_title('Longest Homopolymer Distribution')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('dataset_comparison.png')  # Сохранить график
plt.show()

# Вывод статистики
print("Статистика тренировочного датасета:")
print(train_df[['pKd_value', 'label', 'length', 'gc_content', 'entropy', 'longest_homopolymer']].describe())

print("\nСтатистика сгенерированных аптамеров:")
print(gen_df[['proba', 'label', 'length', 'gc_content', 'entropy', 'longest_homopolymer']].describe())

# Корреляции
print("\nКорреляции в тренировочном датасете:")
print(train_df[['pKd_value', 'length', 'gc_content', 'entropy', 'longest_homopolymer']].corr())

print("\nКорреляции в сгенерированных:")
print(gen_df[['proba', 'length', 'gc_content', 'entropy', 'longest_homopolymer']].corr())