# %% [markdown]
# # Практична робота №3: Дослідження ефективності TF-IDF зважених ембеддингів для детекції текстів, згенерованих великими мовними моделями
#
# **Мета роботи:**
# Засвоїти принципи побудови векторних представлень текстів на основі субсловних ембеддингів (FastText), дослідити вплив TF-IDF зважування на якість агрегації документних векторів, порівняти ефективність лінійних та нелінійних методів класифікації (SVM, Logistic Regression, KNN) у задачі бінарної детекції штучно згенерованих текстів, а також проаналізувати геометричні властивості отриманих векторних просторів.
#
# %% [markdown]
# # Датасет: AI vs Human Text Classifcation Dataset 2026
#
# **Характеристики датасету:**
# - **Цільова змінна:** `generated` - вказує чи текст було згенеровано моделлю.
# - **Кількість спостережень:** 487235 зразків.
# - **Ознаки:** text - власне текст написаний людиною або моделлю

# %%
import os
from collections.abc import Callable
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

script_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
data_dir = os.path.join(script_dir, "data")
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# %% [markdown]
# ### Допоміжні функції для типізації Pandas


# %%
def get_col(df: pd.DataFrame, col_name: str) -> pd.Series:
    """Type-safe column accessor. Guarantees pd.Series return type."""
    return cast(pd.Series, df[col_name])


def filter_col(
    df: pd.DataFrame | pd.Series, col_name: str, cond: Callable[[pd.Series], pd.Series]
) -> pd.Series:
    if isinstance(df, pd.DataFrame):
        return cast(pd.Series, df[col_name][cond])
    else:
        return cast(pd.Series, df[cond])


def filter_df(
    df: pd.DataFrame, col_name: str, cond: Callable[[pd.Series], pd.Series]
) -> pd.DataFrame:
    """Filter rows of a DataFrame based on a column condition lambda."""
    col = cast(pd.Series, df[col_name])
    return cast(pd.DataFrame, df.loc[cond(col)])


# %% [markdown]
# # Етапи виконання

# %% [markdown]
# ## Етап 1. Дослідження даних та попередня обробка

# %%

df_raw = pd.read_parquet(os.path.join(data_dir, "AI_vs_Human.parquet"))
print(df_raw.head())

missing_values = df_raw.isnull().sum()
total_missing = int(missing_values.sum())
print(f"Загальна кількість пропущених значень: {total_missing}")

# %% [markdown]
# ### 1.2 Описові статистики та розподіл цільової змінної

# %%

target_col = "generated"
target_series = get_col(df_raw, target_col)
bin_counts = target_series.value_counts()
bin_percentages = target_series.value_counts(normalize=True) * 100

bin_dist_df = pd.DataFrame(
    {"Кількість": bin_counts, "Частка (%)": bin_percentages.round(2)}
)
print("Розподіл класів у вихідному датасеті:")
print(bin_dist_df)

plt.figure(figsize=(9, 5))
palette = sns.color_palette("viridis", len(bin_counts))
ax = sns.barplot(
    x=bin_counts.index,
    y=bin_counts.values,
    hue=bin_counts.index,
    palette=palette,
    legend=False,
)
plt.title("Розподіл походження тексту", fontsize=13, fontweight="bold")
plt.xlabel("Згенеровано мовною моделлю? (Так/Ні)")
plt.ylabel("Кількість зразків")
plt.tight_layout()
target_dist_plot = os.path.join(plots_dir, "01_target_distribution.png")
plt.savefig(target_dist_plot, dpi=150)
plt.show()

# %% [markdown]
# ### 1.3 Аналіз текстових ознак та виявлення артефактів
#
# Обчислюємо довжину кожного тексту у символах (`char_count`) та словах (`word_count`).
# Для оптимізації пам'яті підрахунок слів здійснюється через генератор у компактний масив `np.int32`.
#
# Перевіряємо наявність артефактів:
# 1. Порожні тексти або тексти, що містять лише пробіли.
# 2. Повні дублікати текстів.
# 3. Аномально короткі тексти (< 10 слів), які не мають достатнього семантичного наповнення.

# %%
# Calculate text lengths with minimal RAM overhead using generator
text_series = get_col(df_raw, "text")
df_raw["char_count"] = text_series.str.len().astype(np.int32)
df_raw["word_count"] = np.fromiter(
    (len(t.split()) for t in text_series),
    dtype=np.int32,
    count=len(df_raw),
)

# Identify artifacts and duplicates
total_duplicates = int(text_series.duplicated().sum())
empty_mask = (get_col(df_raw, "word_count") == 0) | (text_series.str.strip() == "")
short_mask = (get_col(df_raw, "word_count") > 0) & (get_col(df_raw, "word_count") < 10)

total_empty = int(empty_mask.sum())
total_short = int(short_mask.sum())

print("--- Звіт щодо виявлених артефактів ---")
print(f"Загальна кількість текстів: {len(df_raw):,}")
print(f"Повні дублікати текстів: {total_duplicates:,}")
print(f"Порожні/пробільні тексти: {total_empty:,}")
print(f"Надмірно короткі тексти (< 10 слів): {total_short:,}")

# Filter out empty and degenerate texts (< 10 words)
df_clean = filter_df(df_raw, "word_count", lambda s: s >= 10).copy()
df_clean.reset_index(drop=True, inplace=True)
print(f"\nРозмірність датасету після фільтрації артефактів: {len(df_clean):,} зразків")

# %% [markdown]
# ### 1.4 Порівняльний статистичний аналіз та візуалізація довжин текстів
#
# Порівняємо розподіли довжини текстів між класами:
# `0.0` — Людина (Human), `1.0` — Згенеровано мовною моделлю (AI).

# %%
# Compute summary statistics by class
length_stats = df_clean.groupby("generated")[["word_count", "char_count"]].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
)
print("Описові статистики довжин текстів за класами (0 = Human, 1 = AI):")
print(length_stats.T)

# Plot comparative distributions of word count and character count
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

labels = {0.0: "Human", 1.0: "AI"}
palette_dict = {0.0: "#1f77b4", 1.0: "#2ca02c"}

for class_val, label_text in labels.items():
    class_data = filter_df(df_clean, "generated", lambda s, v=class_val: s == v)
    words = get_col(class_data, "word_count")
    chars = get_col(class_data, "char_count")

    sns.kdeplot(
        words,  # type: ignore
        ax=axes[0],
        label=f"{label_text} (медіана: {words.median():.0f})",
        color=palette_dict[class_val],
        clip=(0, 1500),
        linewidth=2,
    )
    sns.kdeplot(
        chars,  # type: ignore
        ax=axes[1],
        label=f"{label_text} (медіана: {chars.median():.0f})",
        color=palette_dict[class_val],
        clip=(0, 8000),
        linewidth=2,
    )

axes[0].set_title("Розподіл кількості слів у тексті", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Кількість слів (Word count)")
axes[0].set_ylabel("Густина ймовірності (Density)")
axes[0].set_xlim(0, 1200)
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.5)

axes[1].set_title(
    "Розподіл кількості символів у тексті", fontsize=12, fontweight="bold"
)
axes[1].set_xlabel("Кількість символів (Character count)")
axes[1].set_ylabel("Густина ймовірності (Density)")
axes[1].set_xlim(0, 7000)
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle(
    "Порівняльний аналіз довжин текстів: Human vs AI",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
length_dist_plot = os.path.join(plots_dir, "02_text_length_distribution.png")
plt.savefig(length_dist_plot, dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Аналітичні інсайти з первинного аналізу даних (EDA):
#
# 1. **Баланс класів та цілісність**:
#    - У вихідному датасеті спостерігається помірний дисбаланс: **62.76%** текстів написано людьми (`generated = 0.0`) та **37.24%** згенеровано ШІ (`generated = 1.0`).
#    - Виявлено та вилучено 4 повністю порожніх рядки та 18 вироджених записів довжиною менше 10 слів, які не несуть семантичної інформації. Повних дублікатів серед текстів не виявлено.
#
# 2. **Характер розподілу довжин текстів**:
#    - Тексти людей мають суттєво більшу варіативність ($\sigma = 186.9$ слів проти $\sigma = 117.0$ у ШІ) та вищу середню довжину (медіана **389** слів проти **337** слів). Розподіл має виражений "важкий" правий хвіст (максимум до 1,668 слів).
#    - Тексти ШІ мають значно більш однорідну та концентровану довжину навколо діапазону 250–400 слів. Це пояснюється внутрішніми лімітами довжини генерації (max tokens), притаманними інтерфейсам та системним промптам мовних моделей.

# %% [markdown]
# ### 1.5 Стратифікована вибірка для подальшого моделювання
#
# Оскільки датасет налічує 487,213 спостережень, а подальший пайплайн передбачає лематизацію SpaCy,
# навчання FastText моделі, розрахунок матриць TF-IDF, а також навчання нелінійних моделей класифікації,
# як **SVM з RBF ядром** (складність тренування $O(N^2) - O(N^3)$) та **KNN**,
# повне навчання на 487 тис. зразків призведе до вичерпання оперативної пам'яті та вимагатиме декілька днів обробки.
#
# Для забезпечення статистичної репрезентативності та ідеального балансу класів відбираємо збалансовану
# стратифіковану підвибірку розміром $N = 10\,000$ (по 5,000 спостережень для кожного класу) з фіксованим `random_state=42`.

# %%
# Form balanced stratified sample for downstream NLP processing and model training
SAMPLE_SIZE = 10_000
SAMPLES_PER_CLASS = SAMPLE_SIZE // 2

human_subset = filter_df(df_clean, "generated", lambda s: s == 0.0).sample(
    n=SAMPLES_PER_CLASS, random_state=42
)
ai_subset = filter_df(df_clean, "generated", lambda s: s == 1.0).sample(
    n=SAMPLES_PER_CLASS, random_state=42
)
df_sample = (
    pd.concat([human_subset, ai_subset], ignore_index=True)
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

print(f"Розмірність стратифікованої вибірки: {df_sample.shape[0]} рядків")
print("Розподіл класів у підвибірці:")
print(df_sample["generated"].value_counts())

# %% [markdown]
# ## Етап 2. Попередня обробка тексту (SpaCy)
#
# Реалізуємо пайплайн обробки текстових даних за допомогою моделі `en_core_web_sm` бібліотеки `spacy`.
#
# **Порядок обробки:**
# 1. Лематизація - приведення кожного слова до базової морфологічної форми.
# 2. Видалення стоп-слів - усунення високочастотних слів загального вжитку.
# 3. Видалення небуквених символів - очищення від пунктуації, чисел та спеціальних знаків.
# 4. Зведення до нижнього регістру.
#
# **Оптимізація швидкодії:**
# Для опрацювання масиву текстів використовуємо конвеєр `nlp.pipe()` з пакетною обробкою (`batch_size=500`).
# Щоб уникнути надлишкових обчислень, відключаємо синтаксичний парсер (`parser`) та модуль розпізнавання
# іменованих сутностей (`ner`), залишаючи лише необхідні для лематизації компоненти (`tok2vec`, `tagger`, `attribute_ruler`, `lemmatizer`).

# %%
import gc

import spacy

# Release raw uncleaned dataframe to minimize memory footprint
del df_raw
gc.collect()

# Load SpaCy pipeline with unnecessary components disabled
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Process texts in batches through nlp.pipe
print(f"Початок попередньої обробки {len(df_sample):,} текстів через SpaCy...")
tokenized_corpus: list[list[str]] = []
sample_texts = get_col(df_sample, "text").tolist()

for doc in nlp.pipe(sample_texts, batch_size=500):
    processed_doc = []
    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue
        processed_doc.append(token.lemma_.lower())
    tokenized_corpus.append(processed_doc)

df_sample["tokens"] = tokenized_corpus
df_sample["clean_text"] = [" ".join(tokens) for tokens in tokenized_corpus]
print("Попередня обробка тексту успішно завершена!")

# Compute basic corpus vocabulary statistics
all_tokens_count = sum(len(toks) for toks in tokenized_corpus)
unique_tokens_set = {t for toks in tokenized_corpus for t in toks}
print(f"Загальна кількість токенів у вибірці: {all_tokens_count:,}")
print(f"Розмір унікального словника (vocabulary size): {len(unique_tokens_set):,}")
print(
    f"Середня кількість токенів на документ: {all_tokens_count / len(tokenized_corpus):.1f}"
)

# Display sample of preprocessed tokens
print("\nПриклад обробленого тексту (Людина):")
human_sample = filter_df(df_sample, "generated", lambda s: s == 0.0).iloc[0]
print("Оригінал (перші 150 симв.):", human_sample["text"][:150], "...")
print("Токени (перші 15 слів):", human_sample["tokens"][:15])

print("\nПриклад обробленого тексту (ШІ):")
ai_sample = filter_df(df_sample, "generated", lambda s: s == 1.0).iloc[0]
print("Оригінал (перші 150 симв.):", ai_sample["text"][:150], "...")
print("Токени (перші 15 слів):", ai_sample["tokens"][:15])

# %% [markdown]
# ### Висновки етапу попередньої обробки:
#
# Видалення стоп-слів і небуквених символів зменшило середню довжину тексту приблизно вдвічі, сконцентрувавши найбільш семантично значущу лексику.
#
# Лематизація звела різноманітні граматичні форми одного слова до єдиної леми (наприклад, форми множини, часові форми дієслів), що значно спрощує подальше знаходження спільних семантичних патернів для моделі FastText.

# %% [markdown]
# ## Етап 3. Навчання моделі FastText
#
# Навчаємо модель субсловних ембеддингів `FastText` з бібліотеки `gensim` на підготовленому корпусі токенів.
#
# **Особливості архітектури та параметри:**
# - `vector_size = 100`: розмірність векторного простору ознак.
# - `window = 5`: розмір контекстного вікна (по 5 слів ліворуч і праворуч).
# - `min_count = 2`: мінімальна частота входження токена для включення у словник.
# - `sg = 1` (Skip-gram): максимізує ймовірність контексту за заданим цільовим словом. Skip-gram краще репрезентує рідкісні слова у порівнянні з CBOW.
# - `workers = 4`: кількість потоків паралельного навчання.
# - `seed = 42`: фіксація випадкового стану для повної відтворюваності результатів.
#
# FastText розглядає кожне слово як сукупність символьних n-грам ($3 \le n \le 6$), що забезпечує стійкість до OOV слів (Out-Of-Vocabulary) та морфологічних похідних.

# %%
import time

from gensim.models import FastText

print("Початок навчання моделі FastText...")
ft_start_time = time.time()

# Train FastText Skip-gram model on the preprocessed corpus
ft_model = FastText(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,
    workers=4,
    seed=42,
)

ft_elapsed = time.time() - ft_start_time
print(f"Модель FastText успішно навчена за {ft_elapsed:.2f} с.")
print(f"Кількість слів у словнику моделі: {len(ft_model.wv.index_to_key):,}")

# Check semantic relationships and vector similarity
test_word = "technology"
if test_word in ft_model.wv:
    similar_words = ft_model.wv.most_similar(test_word, topn=5)
    print(f"\nСлова, найбільш схожі на '{test_word}':")
    for word, score in similar_words:
        print(f"  - {word}: {score:.4f}")

# Demonstrate subword OOV handling
oov_word = "technologization"
oov_in_vocab = oov_word in ft_model.wv.key_to_index
print(f"\nЧи міститься слово '{oov_word}' у базовому словнику? {oov_in_vocab}")
oov_vector = ft_model.wv[oov_word]
print(
    f"Форма OOV-вектора: {oov_vector.shape}, L2-норма: {np.linalg.norm(oov_vector):.4f}"
)
print("Топ-3 схожих слів для OOV-терміна:")
for word, score in ft_model.wv.most_similar([oov_vector], topn=3):
    print(f"  - {word}: {score:.4f}")

# %% [markdown]
# ### Аналітичні інсайти з навчання FastText:
#
# 1. **Семантична когерентність простору**:
#    - Модель FastText успішно виділила стійкі семантичні кластери: для слова `technology` найближчими векторами є релевантні поняття комп'ютерної та технологічної сфери.
# 2. **Ефективність субсловних n-грам (Out-Of-Vocabulary)**:
#    - Для терміна `technologization`, який відсутній у тренувальному словнику, модель зуміла побудувати повноцінний ненульовий вектор завдяки виділенню спільних n-грам (`techno`, `log`, `tion`), знайшовши відповідні семантичні асоціації. Це критично для детекції ШІ-текстів, де моделі генерації часто створюють рідкісні або складні морфологічні форми.

# %% [markdown]
# ## Етап 4. Векторизація документів (Mean Pooling vs TF-IDF Weighted Mean)
#
# Реалізуємо два альтернативні підходи до агрегації векторів слів у єдиний документний вектор:
#
# 1. **Mean Pooling (базовий підхід)**:
#    Обчислює просте середнє арифметичне векторів усіх токенів документа:
#    $$\mathbf{v}_d^{\text{mean}} = \frac{1}{m} \sum_{i=1}^m \mathbf{v}_{w_i}$$
#
# 2. **TF-IDF Weighted Mean (досліджуваний підхід)**:
#    Зважує кожен вектор слова на його статистичну вагу $\text{TF-IDF}(w_i, d, D)$:
#    $$\mathbf{v}_d^{\text{tfidf}} = \frac{\sum_{i=1}^m \text{tfidf}(w_i, d, D) \cdot \mathbf{v}_{w_i}}{\sum_{i=1}^m \text{tfidf}(w_i, d, D)}$$
#    Цей підхід пригнічує загальновживані слова та підсилює специфічні терміни й маркери класів.

# %%
from sklearn.feature_extraction.text import TfidfVectorizer


def mean_pooling(tokens: list[str], model: FastText) -> np.ndarray:
    """Aggregate token vectors via simple arithmetic mean."""
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0, dtype=np.float32)


# Fit TF-IDF Vectorizer directly on token lists
tfidf_vectorizer = TfidfVectorizer(
    analyzer=lambda doc: doc,
    min_df=2,
    sublinear_tf=True,
)
tfidf_matrix = tfidf_vectorizer.fit_transform(tokenized_corpus)
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"Розмірність матриці TF-IDF: {tfidf_matrix.shape}")


def tfidf_weighted_pooling(
    tokens: list[str],
    model: FastText,
    tfidf_row_weights: dict[str, float],
) -> np.ndarray:
    """Aggregate token vectors weighted by their document-specific TF-IDF scores."""
    vectors: list[np.ndarray] = []
    weights: list[float] = []

    for t in tokens:
        if t in model.wv and t in tfidf_row_weights:
            vectors.append(model.wv[t])
            weights.append(tfidf_row_weights[t])

    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)

    weights_arr = np.array(weights, dtype=np.float32)
    weights_sum = weights_arr.sum()
    if weights_sum == 0:
        return np.mean(vectors, axis=0, dtype=np.float32)

    weights_arr /= weights_sum
    return np.average(vectors, axis=0, weights=weights_arr).astype(np.float32)


# Vectorize all documents using both aggregation methods
print("Побудова матриць ознак X_mean та X_tfidf для всіх документів...")
num_docs = len(tokenized_corpus)
vector_size = ft_model.vector_size

X_mean = np.zeros((num_docs, vector_size), dtype=np.float32)
X_tfidf = np.zeros((num_docs, vector_size), dtype=np.float32)

for i in range(num_docs):
    doc_tokens = tokenized_corpus[i]
    X_mean[i] = mean_pooling(doc_tokens, ft_model)

    # Extract non-zero TF-IDF weights for document i
    row_start = tfidf_matrix.indptr[i]
    row_end = tfidf_matrix.indptr[i + 1]
    row_indices = tfidf_matrix.indices[row_start:row_end]
    row_data = tfidf_matrix.data[row_start:row_end]
    doc_tfidf_dict = {
        feature_names[col]: val for col, val in zip(row_indices, row_data)
    }

    X_tfidf[i] = tfidf_weighted_pooling(doc_tokens, ft_model, doc_tfidf_dict)

y = get_col(df_sample, "generated").to_numpy(dtype=np.float32)

mean_norms = np.linalg.norm(X_mean, axis=1)
tfidf_norms = np.linalg.norm(X_tfidf, axis=1)

print(
    f"Матриця X_mean: {X_mean.shape}, середній діапазон L2-норм: [{mean_norms.min():.3f}, {mean_norms.max():.3f}] (середнє: {mean_norms.mean():.3f})"
)
print(
    f"Матриця X_tfidf: {X_tfidf.shape}, середній діапазон L2-норм: [{tfidf_norms.min():.3f}, {tfidf_norms.max():.3f}] (середнє: {tfidf_norms.mean():.3f})"
)
print(
    f"Вектор міток y: {y.shape}, баланс класів: 0.0 -> {(y == 0.0).sum()}, 1.0 -> {(y == 1.0).sum()}"
)

# %% [markdown]
# ### Висновки щодо геометричних властивостей агрегованих просторів:
#
# 1. **Вплив TF-IDF зважування на документні вектори**:
#    - При використанні **Mean Pooling** кожен токен має однакову вагу $1/m$, тому вектор документа тяжіє до центру хмари високочастотних слів загальної тематики. Це викликає згладжування векторів та стискання їхніх евклідових норм.
#    - При **TF-IDF зважуванні** вектори специфічних слів (з високим показником $\text{idf}$) отримують суттєво більшу вагу, тоді як загальна лексика майже нівелюється. Це розширює геометричну варіативність простору та дозволяє виразніше диференціювати специфічні лексичні патерни ШІ.

# %% [markdown]
# ## Етап 5. Навчання та порівняльна оцінка класифікаторів
#
# Проводимо повнофакторне дослідження $2 \times 3$:
# - **2 методи агрегації ознак**: Mean Pooling проти TF-IDF Weighted Mean;
# - **3 моделі класифікації**:
#   1. **Support Vector Machine (SVM)** з нелінійним RBF ядром (`SVC(kernel='rbf', probability=True, random_state=42)`);
#   2. **Logistic Regression (LogReg)** (`LogisticRegression(max_iter=1000, random_state=42)`);
#   3. **K-Nearest Neighbors (KNN)** (`KNeighborsClassifier(n_neighbors=5)`).
#
# **Методологія валідації:**
# - Вибірка розділяється на навчальну (80%, 8,000 зразків) та тестову (20%, 2,000 зразків).
# - Застосовується стратифіковане розбиття (`stratify=y`) з єдиним сидом `random_state=42`, що гарантує ідентичні індекси документів для обох просторів представлення.
# - Якість класифікації оцінюється за метриками:
#   - **Accuracy** (загальна точність);
#   - **Balanced Accuracy** (збалансована точність з урахуванням класів);
#   - **F1-Score (macro)** (гармонійне середнє precision та recall);
#   - **ROC-AUC** (площа під ROC-кривою ймовірнісних оцінок).

# %%
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Split indices to ensure identical train/test splits for both vector spaces
indices = np.arange(len(y))
train_idx, test_idx, y_train, y_test = train_test_split(
    indices,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(
    f"Розмір навчальної вибірки: {len(train_idx)} зразків (по {(y_train == 0).sum()} Human / {(y_train == 1).sum()} AI)"
)
print(
    f"Розмір тестової вибірки:   {len(test_idx)} зразків (по {(y_test == 0).sum()} Human / {(y_test == 1).sum()} AI)"
)

# Prepare dataset splits for both feature spaces
datasets = {
    "Mean Pooling": {
        "X_train": X_mean[train_idx],
        "X_test": X_mean[test_idx],
    },
    "TF-IDF Weighted": {
        "X_train": X_tfidf[train_idx],
        "X_test": X_tfidf[test_idx],
    },
}

# Define models to train and evaluate
model_factories = {
    "SVM (RBF)": lambda: CalibratedClassifierCV(
        estimator=SVC(kernel="rbf", random_state=42),
        ensemble=False,
    ),
    "LogReg": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "KNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
}

# Containers for experimental results and predictions
results_list: list[dict[str, Any]] = []
predictions_cache: dict[str, dict[str, Any]] = {}

print("\n--- Початок навчання та оцінки класифікаторів ---")
for rep_name, split_data in datasets.items():
    X_tr = split_data["X_train"]
    X_te = split_data["X_test"]
    predictions_cache[rep_name] = {}

    for clf_name, clf_builder in model_factories.items():
        print(f"Тренування {clf_name} на представленні [{rep_name}]...")
        clf = clf_builder()

        # Measure training duration
        t_start = time.time()
        clf.fit(X_tr, y_train)
        fit_duration = time.time() - t_start

        # Inference and probability estimation
        y_pred = clf.predict(X_te)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_te)[:, 1]
        else:
            y_prob = clf.decision_function(X_te)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_prob)

        # Store results
        results_list.append(
            {
                "Агрегація": rep_name,
                "Класифікатор": clf_name,
                "Accuracy": acc,
                "Balanced Accuracy": bal_acc,
                "F1 (macro)": f1_macro,
                "ROC-AUC": roc_auc,
                "Час навчання (с)": round(fit_duration, 2),
            }
        )

        predictions_cache[rep_name][clf_name] = {
            "model": clf,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

# Build summary comparison DataFrame
df_results = pd.DataFrame(results_list)
print("\n" + "=" * 80)
print("ЗВЕДЕНА ТАБЛИЦЯ ПОРІВНЯННЯ ЕФЕКТИВНОСТІ КЛАСИФІКАТОРІВ")
print("=" * 80)
print(df_results.to_string(index=False))

# %% [markdown]
# ### Аналітичні інсайти з результатів класифікації:
#
# 1. **Порівняння ефективності методів агрегації (Mean Pooling vs TF-IDF)**:
#    - Використання **TF-IDF зважування** суттєво покращує роздільність класів для лінійних та ядерних моделей (Logistic Regression та SVM). Це підтверджує вихідну гіпотезу: зважування підсилює унікальні лексичні маркери генерації ШІ, пригнічуючи шум частотних слів загального контексту.
#
# 2. **Порівняння архітектур класифікаторів**:
#    - **SVM з ядром RBF** демонструє найвищу узагальнюючу здатність, оскільки успішно будує нелінійну роздільну гіперповерхню у 100-вимірному просторі ембеддингів FastText.
#    - **Logistic Regression** забезпечує чудовий компроміс між якістю та швидкістю: навчається за лічені частки секунди і показує результат, близький до нелінійного SVM, що свідчить про високу лінійну роздільність після TF-IDF агрегації.
#    - **KNN (k=5)** виявляється найбільш чутливим до метрики відстані. При переході до TF-IDF зважування спостерігається специфічна динаміка: нерівномірне розтягування осей може призводити до спотворення локальних околів через ефект "прокляття розмірності", через що приріст якості KNN є менш вираженим.

# %% [markdown]
# ## Етап 6. Візуалізація та аналіз векторних просторів
#
# На цьому етапі реалізуємо:
# 1. **Двовимірну проекцію t-SNE** для візуалізації структури просторів Mean Pooling та TF-IDF Weighted Mean.
# 2. **Матриці плутанини (Confusion Matrices)** для всіх 6 навчених моделей.
# 3. **Лінгвістичний аналіз ключових TF-IDF маркерів**, що відрізняють тексти ШІ від текстів людей.

# %%
from sklearn.manifold import TSNE

# Subsample test set vectors for t-SNE visualization to ensure responsive computation
tsne_sample_size = min(1500, len(test_idx))
rng = np.random.RandomState(42)
tsne_sub_idx = rng.choice(len(test_idx), size=tsne_sample_size, replace=False)

print(
    f"Обчислення 2D проекцій t-SNE для {tsne_sample_size} зразків тестової вибірки..."
)
tsne_model = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)

X_mean_sub = datasets["Mean Pooling"]["X_test"][tsne_sub_idx]
X_tfidf_sub = datasets["TF-IDF Weighted"]["X_test"][tsne_sub_idx]
y_sub = y_test[tsne_sub_idx]

mean_tsne_2d = tsne_model.fit_transform(X_mean_sub)
tfidf_tsne_2d = tsne_model.fit_transform(X_tfidf_sub)

# Plot side-by-side t-SNE projections
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

scatter_labels = {0.0: "Human", 1.0: "AI"}
scatter_colors = {0.0: "#1f77b4", 1.0: "#e377c2"}

for cls_val, cls_label in scatter_labels.items():
    mask = y_sub == cls_val
    axes[0].scatter(
        mean_tsne_2d[mask, 0],
        mean_tsne_2d[mask, 1],
        label=cls_label,
        c=scatter_colors[cls_val],
        alpha=0.6,
        edgecolors="none",
        s=25,
    )
    axes[1].scatter(
        tfidf_tsne_2d[mask, 0],
        tfidf_tsne_2d[mask, 1],
        label=cls_label,
        c=scatter_colors[cls_val],
        alpha=0.6,
        edgecolors="none",
        s=25,
    )

axes[0].set_title("t-SNE: Mean Pooling Простору", fontsize=13, fontweight="bold")
axes[0].set_xlabel("t-SNE Вимір 1")
axes[0].set_ylabel("t-SNE Вимір 2")
axes[0].legend(loc="upper right")
axes[0].grid(True, linestyle="--", alpha=0.4)

axes[1].set_title(
    "t-SNE: TF-IDF Weighted Mean Простору", fontsize=13, fontweight="bold"
)
axes[1].set_xlabel("t-SNE Вимір 1")
axes[1].set_ylabel("t-SNE Вимір 2")
axes[1].legend(loc="upper right")
axes[1].grid(True, linestyle="--", alpha=0.4)

plt.suptitle(
    "Геометричний аналіз векторних просторів (t-SNE)",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)
plt.tight_layout()
tsne_plot_path = os.path.join(plots_dir, "03_tsne_projections.png")
plt.savefig(tsne_plot_path, dpi=150, bbox_inches="tight")
plt.show()

# %%
# Plot normalized confusion matrices for all evaluated models (2x3 grid)
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
rep_names_list = list(datasets.keys())
model_names_list = list(model_factories.keys())

class_names = ["Human", "AI"]

for r_idx, rep_name in enumerate(rep_names_list):
    for c_idx, clf_name in enumerate(model_names_list):
        ax = axes[r_idx, c_idx]
        y_pred_model = predictions_cache[rep_name][clf_name]["y_pred"]
        cm_norm = confusion_matrix(y_test, y_pred_model, normalize="true")

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2%",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
        )
        ax.set_title(f"{clf_name} ({rep_name})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Передбачений клас")
        ax.set_ylabel("Фактичний клас")

plt.suptitle(
    "Матриці плутанини класифікаторів (Нормалізовані за рядками)",
    fontsize=15,
    fontweight="bold",
    y=1.00,
)
plt.tight_layout()
cm_plot_path = os.path.join(plots_dir, "04_confusion_matrices.png")
plt.savefig(cm_plot_path, dpi=150, bbox_inches="tight")
plt.show()

# %%
# Extract and analyze top TF-IDF words characterizing AI vs Human classes
human_doc_indices = [i for i, doc_y in enumerate(y) if doc_y == 0.0]
ai_doc_indices = [i for i, doc_y in enumerate(y) if doc_y == 1.0]

mean_tfidf_human = np.asarray(tfidf_matrix[human_doc_indices].mean(axis=0)).flatten()
mean_tfidf_ai = np.asarray(tfidf_matrix[ai_doc_indices].mean(axis=0)).flatten()

tfidf_diff_ai = mean_tfidf_ai - mean_tfidf_human
tfidf_diff_human = mean_tfidf_human - mean_tfidf_ai

top_ai_idx = np.argsort(tfidf_diff_ai)[::-1][:15]
top_human_idx = np.argsort(tfidf_diff_human)[::-1][:15]

df_top_markers = pd.DataFrame(
    {
        "Топ маркерів AI": [feature_names[i] for i in top_ai_idx],
        "Різниця TF-IDF (AI - Human)": [
            round(float(tfidf_diff_ai[i]), 5) for i in top_ai_idx
        ],
        "Топ маркерів Human": [feature_names[i] for i in top_human_idx],
        "Різниця TF-IDF (Human - AI)": [
            round(float(tfidf_diff_human[i]), 5) for i in top_human_idx
        ],
    }
)

print("\n" + "=" * 80)
print("ТОП ЛЕКСИЧНИХ МАРКЕРІВ ТЕКСТІВ (НА ОСНОВІ РІЗНИЦІ TF-IDF)")
print("=" * 80)
print(df_top_markers.to_string(index=False))

# %% [markdown]
# # Відповіді на дослідницькі питання
#
# ## 1. Вплив агрегації: Як TF-IDF зважування змінює геометрію векторного простору порівняно з Mean Pooling?
# - **Механізм трансформації**: Mean Pooling усереднює вектори слів з рівними вагами $1/m$. Через високу частоту загальновживаних слів документні вектори зсуваються до єдиного "центроїда мови", що зменшує кутову та евклідову відстань між документами різних класів (ефект згладжування семантики).
# - **Зміна геометрії**: TF-IDF зважування пригнічує спільну міжкласову лексику завдяки високій частоті в документах корпусу ($\text{idf} \to 0$) та експоненційно підсилює рідкісні, специфічні для класу слова. Вектори документів розсуваються у просторі ознак за специфічними семантичними напрямками, збільшуючи міжкласову дисперсію та покращуючи лінійну роздільність.
#
# ## 2. Порівняння класифікаторів: Який класифікатор демонструє найкращі результати для кожного методу агрегації?
# - **Найкраща модель**: Нелінійний **SVM з ядром RBF** показує найкращі результати як на Mean Pooling, так і на TF-IDF просторі, досягаючи найвищих значень F1-macro та ROC-AUC. RBF ядро проектує 100-вимірні вектори у нескінченновимірний гільбертів простір, ефективно моделюючи складні межі рішень.
# - **Лінійна роздільність**: **Logistic Regression** отримує найбільший відносний приріст якості при переході до TF-IDF Weighted Mean. Це доводить, що TF-IDF зважування робить класи суттєво більш лінійно роздільними, наближаючи якість простої лінійної моделі до складного нелінійного SVM при багаторазово менших обчислювальних витратах.
#
# ## 3. Чутливість KNN: Чому KNN демонструє специфічну динаміку / деградацію при TF-IDF зважуванні?
# - **Прокляття розмірності (Curse of Dimensionality)**: KNN опирається на локальну евклідову метрику у 100-вимірному просторі. При Mean Pooling вектори зосереджені в компактній гіперсфері, де відносні відстані між найближчим і найвіддаленішим сусідами є стійкими.
# - **Спотворення метрики через ваги**: TF-IDF зважування сильно видовжує окремі осі простору залежно від наявності специфічних рідкісних слів у документі. Це спричиняє нерівномірну деформацію векторного простору: евклідова відстань починає визначатися одиничними домінуючими словами, а не загальним контекстом, що знижує щільність корисної локальної інформації для $k$ найближчих сусідів.
#
# ## 4. Лінгвістична інтерпретація: Які слова мають найбільші TF-IDF значення для класу AI?
# - **Маркери штучного тексту (AI)**: Серед виділених маркерів домінують структуруючі зв'язки, формальні терміни, вставні конструкції та узагальнюючі поняття (наприклад, слова, що формують логічний каркас відповіді: структуровані аргументи, переліки, формальний академічний тон).
# - **Маркери людського тексту (Human)**: Тексти людей характеризуються більшою кількістю суб'єктивних дієслів, емоційно забарвлених слів, розмовної лексики та специфічних орфографічних або стилістичних нюансів, які рідко зустрічаються у стандартизованих відповідях LLM.
#
# ## 5. Геометричний аналіз: Чи утворюють класи компактні кластери на t-SNE візуалізації?
# - **Кластеризація на проекціях**: На 2D проекції t-SNE для **Mean Pooling** спостерігається значне перекриття двох хмар точок, де тексти людей і ШІ перемішані у центральній області через спільну тематичну лексику.
# - **Покращення на TF-IDF**: У просторі **TF-IDF Weighted Mean** взаємне перекриття класів зменшується: формуються виразніші локальні підкластери та периферичні зони, збагачені виключно текстами ШІ або текстами людей. Це візуально підтверджує аналітичний висновок про зростання роздільності класів у трансформованому просторі.
