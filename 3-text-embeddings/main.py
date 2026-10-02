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
from typing import cast

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
#    - **Тексти людей (Human)** демонструють суттєво більшу варіативність ($\sigma = 186.9$ слів проти $\sigma = 117.0$ у ШІ) та вищу середню довжину (медіана **389** слів проти **337** слів). Розподіл має виражений "важкий" правий хвіст (максимум до 1,668 слів).
#    - **Тексти ШІ (AI)** мають значно більш однорідну та концентровану довжину навколо діапазону 250–400 слів. Це пояснюється внутрішніми лімітами довжини генерації (max tokens), притаманними інтерфейсам та системним промптам мовних моделей.

# %% [markdown]
# ### 1.5 Стратифікована вибірка для подальшого моделювання
#
# Оскільки датасет налічує 487,213 спостережень, а подальший пайплайн передбачає лематизацію SpaCy,
# навчання FastText моделі, розрахунок матриць TF-IDF, а також навчання нелінійних моделей класифікації:
# **SVM з RBF ядром** (складність тренування $O(N^2) - O(N^3)$) та **KNN**, повне навчання на 487 тис. зразків
# призведе до вичерпання оперативної пам'яті та багатодобового часу очікування.
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
# **Вимоги до обробки згідно з інструкцією:**
# 1. **Лематизація** — приведення кожного слова до базової морфологічної форми (`lemma_`).
# 2. **Видалення стоп-слів** — усунення високочастотних слів загального вжитку (`not token.is_stop`).
# 3. **Видалення небуквених символів** — очищення від пунктуації, чисел та спеціальних знаків (`token.is_alpha`).
# 4. **Зведення до нижнього регістру** (`token.lemma_.lower()`).
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


def preprocess_tokens(doc: spacy.tokens.Doc) -> list[str]:
    """Extract lemmatized, lowercase alpha tokens excluding stopwords."""
    return [
        token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop
    ]


# Process texts in batches through nlp.pipe
print(f"Початок попередньої обробки {len(df_sample):,} текстів через SpaCy...")
tokenized_corpus: list[list[str]] = []
sample_texts = get_col(df_sample, "text").tolist()

for doc in nlp.pipe(sample_texts, batch_size=500):
    tokenized_corpus.append(preprocess_tokens(doc))

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
# 1. **Фільтрація шуму та лематизація**:
#    - Видалення стоп-слів і небуквених символів зменшило середню довжину тексту приблизно вдвічі, сконцентрувавши найбільш семантично значущу лексику.
#    - Лематизація звела різноманітні граматичні форми одного слова до єдиної леми (наприклад, форми множини, часові форми дієслів), що значно спрощує подальше знаходження спільних семантичних патернів для моделі FastText.

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
