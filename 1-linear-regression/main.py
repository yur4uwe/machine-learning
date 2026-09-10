# %% [markdown]
# # Лабораторна робота №1: Лінійна регресія, оптимізація та статистичний аналіз
#
# Мета роботи - розібрати внутрішню математику та архітектуру лінійної регресії, реалізувати з нуля градієнтні методи оптимізації засобами `numpy`, дослідити припущення класичної регресійної моделі та провести аналіз залишків.

# %% [markdown]
# ### 1.1 Завантаження бібліотек та датасету
#
# На цьому етапі імпортуються базові бібліотеки для аналізу даних і візуалізації, створюються директорії для збереження графіків, та зчитується вихідний датасет Ames Housing Dataset (`train.csv`).

# %%
import os
from collections.abc import Callable, Iterator
from typing import Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

script_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if "__file__" in globals()
    else os.getcwd()
)
plots_dir = os.path.join(script_dir, "plots")
data_dir = os.path.join(script_dir, "data")
os.makedirs(plots_dir, exist_ok=True)

# Завантаження датасету
data_path = os.path.join(data_dir, "train.csv")
if not os.path.exists(data_path):
    # Перевірка альтернативного шляху (якщо запускається з кореня)
    data_path = os.path.join("1-linear-regression", "data", "train.csv")
df_raw = pd.read_csv(data_path)

print(f"Dataset shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head()


# %% [markdown]
# ### 1.2 Допоміжні функції для типізації Pandas
#
# Наступна комірка містить функції-обгортки для безпечного доступу та фільтрації стовпців DataFrame з явною перевіркою типів для Language Server Protocol (Pyright/LSP).


# %%
# Helpers for type-safe pandas
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
# ### 1.3 Аналіз типів ознак та пропущених значень
#
# Виконується підрахунок числових та категоріальних стовпців, а також розраховується відсоток пропущених значень (NaN) у кожній колонці для визначення стратегії імпутації.

# %%
# Analyze feature types and missing values
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw)) * 100
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
missing_df = filter_df(missing_df, "Missing Count", lambda s: s > 0).sort_values(
    by="Missing %", ascending=False
)

print(
    f"Number of numerical features: {df_raw.select_dtypes(include=[np.number]).shape[1]}"
)
print(
    f"Number of categorical features: {df_raw.select_dtypes(include=['object', 'string']).shape[1]}"
)
print(f"Columns with missing values: {len(missing_df)}")
missing_df.head(10)

# %% [markdown]
# ### 1.4 Аналіз цільової змінної (`SalePrice`)
#
# Обчислюються описові описові статистики для ціни нерухомості (середнє, медіана, мінімум, максимум, асиметрія). Будуються гістограми вихідного та логарифмованого розподілу для оцінки симетрії.

# %%
target = "SalePrice"
sale_price = get_col(df_raw, target)

print("--- Descriptive Statistics for SalePrice ---")
print(f"Mean:     ${sale_price.mean():,.2f}")
print(f"Median:   ${sale_price.median():,.2f}")
print(f"Min:      ${sale_price.min():,.2f}")
print(f"Max:      ${sale_price.max():,.2f}")
print(f"Skewness:  {sale_price.skew():.2f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Гістограма та KDE
sns.histplot(sale_price, kde=True, ax=axes[0], color="royalblue")  # pyright: ignore
axes[0].set_title("Розподіл SalePrice (Вихідний)")
axes[0].set_xlabel("Ціна продажу ($)")

# Q-Q або логарифмований розподіл для порівняння
sns.histplot(np.log1p(sale_price), kde=True, ax=axes[1], color="teal")
axes[1].set_title("Розподіл log(1 + SalePrice)")
axes[1].set_xlabel("log(SalePrice)")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "eda_target_distribution.png"), dpi=300)
plt.show()

# %% [markdown]
# Висновки з аналізу цільової змінної:
# 1. Розподіл `SalePrice` має виражену правосторонню асиметрію (Skewness $\approx 1.88$) — більшість об'єктів зосереджена в діапазоні від $100,000 до $250,000, але присутні одиничні дорогі маєтки вартістю понад $500,000.
# 2. Логарифмування `log(1 + SalePrice)` помітно наближає форму розподілу до нормального дзвону, що суттєво зменшує вплив екстремальних викидів.

# %% [markdown]
# ### 1.5 Розбиття вибірки (Train / Validation / Test)
#
# Теоретичне питання: Чому ми повинні розбити дані до, а не після масштабування, кодування або інших операцій? Поясніть термін Data Leakage.
#
# Відповідь:
# - Data Leakage (витік даних) виникає, коли інформація з валідаційної або тестової вибірки ненавмисно потрапляє у процес навчання моделі або розрахунку параметрів попередньої обробки.
# - Якщо обчислити середнє ($\mu$) та стандартне відхилення ($\sigma$) для масштабування на всьому датасеті, тестові приклади спотворять ці параметри, створивши оптимістичне зміщення (optimistic bias).
# - Будь-яка трансформація повинна оцінювати свої параметри (медіани, середні, категорії) виключно на Train вибірці, і лише застосовуватися до Validation та Test.
#
# У наступній комірці виконується чисте випадкове розбиття датасету у співвідношенні 70% Train, 15% Validation, 15% Test.


# %%
# Реалізація розбиття вибірки засобами NumPy/Pandas (70% Train / 15% Val / 15% Test)
def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: float = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert train_ratio + val_ratio + test_ratio == 1.0, (
        "Train, Val, Test ratios must sum up to 1.0"
    )

    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(df))

    train_end = int(train_ratio * len(df))
    val_end = train_end + int(val_ratio * len(df))

    train_idx = shuffled_indices[:train_end]
    val_idx = shuffled_indices[train_end:val_end]
    test_idx = shuffled_indices[val_end:]

    train_df = df.take(train_idx).copy()
    val_df = df.take(val_idx).copy()
    test_df = df.take(test_idx).copy()

    return train_df, val_df, test_df


df_train, df_val, df_test = train_val_test_split(
    df_raw, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
)
print("Split sizes:")
print(f"  Train:      {len(df_train)} rows ({len(df_train) / len(df_raw) * 100:.1f}%)")
print(f"  Validation: {len(df_val)} rows ({len(df_val) / len(df_raw) * 100:.1f}%)")
print(f"  Test:       {len(df_test)} rows ({len(df_test) / len(df_raw) * 100:.1f}%)")

# %% [markdown]
# ### 1.6 Відбір ознак (Feature Selection на Train-вибірці)
#
# Дослідимо кореляцію числових ознак із цільовою змінною `SalePrice` та між собою на навчальній вибірці, щоб обрати інформативні предиктори та виявити потенційну мультиколінеарність.
# Відбір виконується виключно на Train вибірці.

# %%
# Correlation analysis of numeric features with SalePrice on Train
corr_matrix = df_train.drop(columns=["Id"], errors="ignore").corr(numeric_only=True)

# Top features most correlated with SalePrice
price_corr = get_col(corr_matrix, "SalePrice").sort_values(ascending=False)
# Skip SalePrice itself
top_corr_features = filter_col(
    price_corr, "SalePrice", lambda s: s.abs() > 0.4
).index.tolist()[1:]

print("Top numerical features by correlation with SalePrice:")
print(price_corr.loc[top_corr_features])

# %% [markdown]
# ### 1.7 Візуалізація кореляційної матриці топ-ознак
#
# Наступна комірка генерує теплову карту кореляцій (Heatmap) між відібраними числовими ознаками для візуальної оцінки зв'язків та перевірки на мультиколінеарність.

# %%
# Correlation heatmap for top features
plt.figure(figsize=(11, 8))
sns.heatmap(
    corr_matrix.loc[top_corr_features, top_corr_features],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=True,
)
plt.title("Матриця кореляцій топ-ознак (Train)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "eda_correlation_heatmap.png"), dpi=300)
plt.show()

# %% [markdown]
# Спостереження щодо взаємної кореляції між ознаками:
# - `GarageCars` та `GarageArea` мають кореляцію r > 0.88, оскільки обидві описують розмір гаража.
# - `TotalBsmtSF` та `1stFlrSF` мають кореляцію r > 0.81 через конструктивну подібність площі підвалу та першого поверху.
# - `GrLivArea` та `TotRmsAbvGrd` мають кореляцію r > 0.82, оскільки зі збільшенням житлової площі зростає кількість кімнат.
#
# Фінальний набір обраних ознак для моделі:
# 1. Числові ознаки:
#    - `OverallQual`: загальна якість матеріалів та оздоблення (r = 0.79 з ціною).
#    - `GrLivArea`: загальна житлова площа над землею (r = 0.71).
#    - `TotalBsmtSF`: площа підвального приміщення (r = 0.61).
#    - `GarageCars`: місткість гаража за кількістю авто (r = 0.64).
#    - `YearBuilt`: рік побудови будинку (r = 0.52).
#    - `FullBath`: кількість повноцінних ванних кімнат (r = 0.56).
# 2. Категоріальні ознаки:
#    - `KitchenQual`: порядкова оцінка якості кухні (Ex, Gd, TA, Fa, Po).
#    - `Neighborhood`: номінальна ознака району розташування нерухомості.
#    - `CentralAir`: бінарна наявність центральної системи кондиціонування.

# %% [markdown]
# ### 1.8 Кодування категоріальних ознак (Categorical Encoding)
#
# Теоретичне питання: Як обробляти категорії, які з'являться на валідації/тесті, але яких не було в Train?
#
# Відповідь:
# 1. One-Hot Encoding: Для нових категорій під час трансформації валідаційної чи тестової вибірки всі відповідні бінарні колонки встановлюються в 0 (handle_unknown='ignore').
# 2. Ordinal Encoding: Невідомим категоріям присвоюється значення за замовчуванням (медіанний або найчастіший ранг у Train вибірці).
# 3. Рідкісні категорії: Категорії з частотою менше 1% у Train можна згрупувати в окрему категорію 'Other'.
#
# У наступній комірці реалізовано порядкове кодування для `KitchenQual` та One-Hot кодування для `Neighborhood` і `CentralAir` без витоку даних.

# %%
# Словники для порядкового (Ordinal) кодування
quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0}

selected_numeric = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "FullBath",
]
selected_ordinal = ["KitchenQual"]
selected_nominal = ["Neighborhood", "CentralAir"]


def encode_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
):
    # 1. Ordinal Encoding
    for col in selected_ordinal:
        default_val = quality_map.get("TA", 3)
        train_df[col + "_encoded"] = train_df[col].map(quality_map).fillna(default_val)
        val_df[col + "_encoded"] = val_df[col].map(quality_map).fillna(default_val)
        test_df[col + "_encoded"] = test_df[col].map(quality_map).fillna(default_val)

    # 2. One-Hot Encoding для номінальних ознак на основі категорій з Train
    train_encoded_parts: list[pd.DataFrame | pd.Series] = [
        train_df[selected_numeric + [c + "_encoded" for c in selected_ordinal]].copy()
    ]
    val_encoded_parts: list[pd.DataFrame | pd.Series] = [
        val_df[selected_numeric + [c + "_encoded" for c in selected_ordinal]].copy()
    ]
    test_encoded_parts: list[pd.DataFrame | pd.Series] = [
        test_df[selected_numeric + [c + "_encoded" for c in selected_ordinal]].copy()
    ]

    for col in selected_nominal:
        # Отримуємо унікальні категорії лише з Train
        categories = train_df[col].dropna().unique()

        for cat in categories:
            col_name = f"{col}_{cat}"
            train_encoded_parts.append(
                pd.Series(
                    (train_df[col] == cat).astype(float),
                    name=col_name,
                    index=train_df.index,
                )
            )
            val_encoded_parts.append(
                pd.Series(
                    (val_df[col] == cat).astype(float),
                    name=col_name,
                    index=val_df.index,
                )
            )
            test_encoded_parts.append(
                pd.Series(
                    (test_df[col] == cat).astype(float),
                    name=col_name,
                    index=test_df.index,
                )
            )

    X_tr = pd.concat(train_encoded_parts, axis=1)
    X_va = pd.concat(val_encoded_parts, axis=1)
    X_te = pd.concat(test_encoded_parts, axis=1)

    # Заповнюємо можливі числові пропуски медіанами з Train
    for col in X_tr.columns:
        median_val = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(median_val)
        X_va[col] = X_va[col].fillna(median_val)
        X_te[col] = X_te[col].fillna(median_val)

    return X_tr, X_va, X_te


X_train_raw, X_val_raw, X_test_raw = encode_features(df_train, df_val, df_test)
y_train = np.asarray(df_train[target].values.astype(float))
y_val = np.asarray(df_val[target].values.astype(float))
y_test = np.asarray(df_test[target].values.astype(float))

print("Feature matrix shapes after encoding:")
print(f"  X_train: {X_train_raw.shape}")
print(f"  X_val:   {X_val_raw.shape}")
print(f"  X_test:  {X_test_raw.shape}")

# %% [markdown]
# ### 1.9 Масштабування ознак (Standardization / Z-Score)
#
# Числові змінні приводяться до єдиного масштабу за формулою стандартизації:
# $$z = \frac{x - \mu}{\sigma}$$
#
# Параметри масштабування ($\mu_{\text{train}}$ та $\sigma_{\text{train}}$) обчислюються виключно на вибірці Train та без змін застосовуються до Validation і Test.
# Наступна комірка містить реалізацію `StandardScaler` на чистому NumPy.


# %%
class StandardScaler:
    _mean: np.ndarray | None
    _std: np.ndarray | None

    def __init__(self):
        self._mean = None
        self._std = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=float)
        self._mean = np.mean(X_arr, axis=0)
        self._std = np.std(X_arr, axis=0)

        # Prevent division by zero for constant features
        self._std[self._std == 0.0] = 1.0
        return self

    def transform(self, X):
        if self._mean is None or self._std is None:
            raise ValueError("StandardScalerNumPy has not been fitted yet.")

        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self._mean) / self._std

    def fit_transform(self, X):
        return self.fit(X).transform(X)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw.values)
# We use the same scaler for validation and test data as for training
X_val = scaler.transform(X_val_raw.values)
X_test = scaler.transform(X_test_raw.values)

print("Standardization completed successfully.")
print(f"X_train: mean={np.mean(X_train):.4f}, std={np.std(X_train):.4f}")
print(f"X_val:   mean={np.mean(X_val):.4f}, std={np.std(X_val):.4f}")
print(f"X_test:  mean={np.mean(X_test):.4f}, std={np.std(X_test):.4f}")

# %% [markdown]
# ## 2.2 Етап 2. Реалізація Лінійної Регресії засобами NumPy
#
# Модель лінійної регресії має вигляд:
# $$\hat{y} = \mathbf{X}\mathbf{w} + b\mathbf{1}$$
#
# Математичний апарат оптимізації:
# 1. Функція втрат (MSE / 2):
#    $$J(\mathbf{w}, b) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 = \frac{1}{2n} \|\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\|_2^2$$
# 2. Аналітичний векторний градієнт за вагами $\mathbf{w}$:
#    $$\nabla_{\mathbf{w}} J = \frac{1}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
# 3. Аналітичний градієнт за зміщенням $b$:
#    $$\nabla_b J = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i) = \frac{1}{n} \mathbf{1}^T (\hat{\mathbf{y}} - \mathbf{y})$$
# 4. Правило оновлення параметрів:
#    $$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla_{\mathbf{w}} J, \quad b \leftarrow b - \alpha \nabla_b J$$
#
# Теоретична довідка: Normal Equation та його обмеження:
# - Аналітичний розв'язок задачі МНК без градієнтного спуску має вигляд:
#   $$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$
# - Обмеження Normal Equation:
#   1. Обчислювальна складність: обчислення оберненої матриці $(\mathbf{X}^T \mathbf{X})^{-1}$ вимагає часу $\mathcal{O}(d^3)$, де $d$ — кількість ознак. При великій кількості ознак (наприклад, $d > 10{,}000$) метод стає непридатним.
#   2. Проблема невиродженості: якщо матриця $\mathbf{X}^T \mathbf{X}$ сингулярна (внаслідок точної мультиколінеарності або коли $d > n$), матриця не має точної оберненої і вимагає псевдообернення Moore-Penrose.
#   3. Пам'ять: побудова та зберігання матриці $\mathbf{X}^T \mathbf{X}$ розміром $d \times d$ потребує значного обсягу пам'яті.
#
# У наступній комірці реалізовано повністю векторизовані функції втрат, обчислення градієнтів та базовий клас `LRModel`.

# %%


def mse_loss(y_pred: np.ndarray, y: np.ndarray) -> float:
    if len(y_pred) != len(y):
        raise ValueError("y_pred and y must have the same length")

    return np.mean((y_pred - y) ** 2) / 2


def weight_loss_gradient(feature_matrix: np.ndarray, y_pred: np.ndarray, y: np.ndarray):
    return (feature_matrix.T @ (y_pred - y)) / len(y)


def bias_loss_gradient(y_pred: np.ndarray, y: np.ndarray):
    if len(y_pred) != len(y):
        raise ValueError("y_pred and y must have the same length")
    return np.mean(y_pred - y)


class BatchSampler(Protocol):
    def get_batches(
        sampler,  # pyright: ignore
        X: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield batches of (X_batch, y_batch) for an epoch."""
        ...


class LRModel:
    weight_vector: np.ndarray
    bias: float
    learning_rate: float

    def __init__(
        lr_model,  # pyright: ignore
        lr: float,
        weight_vector: np.ndarray,
        bias: float,
    ):
        lr_model.weight_vector = weight_vector
        lr_model.bias = bias
        lr_model.learning_rate = lr

    def predict(lr_model, features: np.ndarray):  # pyright: ignore
        fm_shape = features.shape
        wv_shape = lr_model.weight_vector.shape
        if fm_shape[1] != wv_shape[0]:
            raise ValueError(
                "feature_matrix and weight_vector must have the same number of columns"
            )

        return np.dot(features, lr_model.weight_vector) + lr_model.bias

    def update_params(lr_model, w_grad: np.ndarray, b_grad: float):  # pyright: ignore
        if len(lr_model.weight_vector) != len(w_grad):
            raise ValueError("w_grad and weight_vector must have the same length")

        lr_model.weight_vector -= lr_model.learning_rate * w_grad
        lr_model.bias -= lr_model.learning_rate * b_grad

    def train(
        lr_model,  # pyright: ignore
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        sampler: BatchSampler,
    ) -> list[float]:
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have the same length")
        if len(X_train[0]) != len(lr_model.weight_vector):
            raise ValueError("X_train and weight_vector must have the same length")

        loss_history: list[float] = []

        for _ in range(epochs):
            for X_batch, y_batch in sampler.get_batches(X_train, y_train):
                y_pred_batch = lr_model.predict(X_batch)
                w_grad = weight_loss_gradient(X_batch, y_pred_batch, y_batch)
                b_grad = bias_loss_gradient(y_pred_batch, y_batch)
                lr_model.update_params(w_grad, b_grad)

            total_pred = lr_model.predict(X_train)
            loss_history.append(mse_loss(total_pred, y_train))

        return loss_history


# %% [markdown]
# ## 2.3 Етап 3. Градієнтний спуск: три варіанти оптимізації
#
# Реалізуємо та навчаємо лінійну регресію за допомогою трьох стратегій вибірки батчів:
# 1. Batch GD (повний градієнтний спуск): оновлює ваги один раз за епоху за всіма 1021 прикладом.
# 2. SGD (стохастичний градієнтний спуск): оновлює ваги після кожного окремого прикладу (1021 оновлення за епоху).
# 3. Mini-batch GD: оновлює ваги за невеликими підвибірками розміром B = 64 (16 оновлень за епоху).
#
# У наступній комірці реалізовано генератори вибірок (Samplers) для кожного методу згідно з інтерфейсом `BatchSampler`.

# %%


class BatchGDSampler:
    def get_batches(
        sampler,  # pyright: ignore
        X: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield X, y


class SGDSampler:
    rng: np.random.Generator

    def __init__(sampler, random_seed: int = 42):  # pyright: ignore
        sampler.rng = np.random.default_rng(random_seed)

    def get_batches(
        sampler,  # pyright: ignore
        X: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = sampler.rng.permutation(len(X))
        for i in indices:
            yield X[i : i + 1], y[i : i + 1]


class MiniBatchGDSampler:
    batch_size: int
    rng: np.random.Generator

    def __init__(sampler, batch_size: int = 64, random_seed: int = 42):  # pyright: ignore
        sampler.batch_size = batch_size
        sampler.rng = np.random.default_rng(random_seed)

    def get_batches(
        sampler,  # pyright: ignore
        X: np.ndarray,
        y: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        indices = sampler.rng.permutation(len(X))
        for start in range(0, len(X), sampler.batch_size):
            batch_idx = indices[start : start + sampler.batch_size]
            yield X[batch_idx], y[batch_idx]


# %% [markdown]
# ### 2.3.1 Базове навчання трьох оптимізаторів
#
# Навчаємо три моделі протягом 80 епох з базовими темпами навчання:
# - Batch GD: lr = 0.05
# - SGD: lr = 0.001
# - Mini-batch GD (B = 64): lr = 0.01

# %%
num_features = X_train.shape[1]
epochs = 80

# 1. Batch Gradient Descent
model_bgd = LRModel(lr=0.05, weight_vector=np.zeros(num_features), bias=0.0)
history_bgd = model_bgd.train(X_train, y_train, epochs=epochs, sampler=BatchGDSampler())

# 2. Stochastic Gradient Descent (SGD)
model_sgd = LRModel(lr=0.001, weight_vector=np.zeros(num_features), bias=0.0)
history_sgd = model_sgd.train(
    X_train, y_train, epochs=epochs, sampler=SGDSampler(random_seed=42)
)

# 3. Mini-batch Gradient Descent (B = 64)
model_mbgd = LRModel(lr=0.01, weight_vector=np.zeros(num_features), bias=0.0)
history_mbgd = model_mbgd.train(
    X_train,
    y_train,
    epochs=epochs,
    sampler=MiniBatchGDSampler(batch_size=64, random_seed=42),
)

print("Training finished for all 3 baseline optimization methods.")
print("Final Train Loss (MSE/2):")
print(f"  Batch GD:      {history_bgd[-1]:,.2f}")
print(f"  SGD:           {history_sgd[-1]:,.2f}")
print(f"  Mini-batch GD: {history_mbgd[-1]:,.2f}")

# %% [markdown]
# ### 2.3.2 Контрольовані експерименти оптимізації
#
# Для ізольованого аналізу оптимізаторів проведено два контрольованих експерименти з фіксованими осями ординат на логарифмічній шкалі.
#
# Експеримент 1: Вплив Learning Rate на збіжність кожного оптимізатора
# Досліджуються 4 режими швидкості навчання (Small, Medium, Optimal, Diverging) окремо для Batch GD, SGD та Mini-batch GD.

# %%
num_features = X_train.shape[1]
epochs = 80

# 1. Batch GD Learning Rate Sweep
lrs_bgd = [0.005, 0.05, 0.20, 1.05]
histories_bgd_lr: dict[float, list[float]] = {}
for lr in lrs_bgd:
    model = LRModel(lr=lr, weight_vector=np.zeros(num_features), bias=0.0)
    histories_bgd_lr[lr] = model.train(
        X_train, y_train, epochs=epochs, sampler=BatchGDSampler()
    )

# 2. SGD Learning Rate Sweep
lrs_sgd = [0.0001, 0.0005, 0.001, 0.01]
histories_sgd_lr: dict[float, list[float]] = {}
for lr in lrs_sgd:
    model = LRModel(lr=lr, weight_vector=np.zeros(num_features), bias=0.0)
    histories_sgd_lr[lr] = model.train(
        X_train, y_train, epochs=epochs, sampler=SGDSampler(random_seed=42)
    )

# 3. Mini-Batch GD (B=64) Learning Rate Sweep
lrs_mbgd = [0.001, 0.01, 0.05, 0.25]
histories_mbgd_lr: dict[float, list[float]] = {}
for lr in lrs_mbgd:
    model = LRModel(lr=lr, weight_vector=np.zeros(num_features), bias=0.0)
    histories_mbgd_lr[lr] = model.train(
        X_train,
        y_train,
        epochs=epochs,
        sampler=MiniBatchGDSampler(batch_size=64, random_seed=42),
    )

# Plot Experiment 1 with 3 Subplots Side-by-Side
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
colors = ["#9b59b6", "#3498db", "#2ecc71", "#e74c3c"]

for (lr, hist), col in zip(histories_bgd_lr.items(), colors, strict=False):
    tag = (
        "Diverging"
        if lr >= 1.0
        else "Optimal"
        if lr == 0.20
        else "Medium"
        if lr == 0.05
        else "Small"
    )
    axes[0].plot(
        range(1, epochs + 1), hist, label=f"lr={lr} ({tag})", color=col, linewidth=2.0
    )
axes[0].set_title("Batch GD (Variable: Learning Rate)")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss (MSE / 2)")
axes[0].set_yscale("log")
axes[0].set_ylim(bottom=1e8, top=1e12)
axes[0].legend(frameon=True, facecolor="white")

for (lr, hist), col in zip(histories_sgd_lr.items(), colors, strict=False):
    tag = (
        "Diverging"
        if lr >= 0.01
        else "Optimal"
        if lr == 0.001
        else "Medium"
        if lr == 0.0005
        else "Small"
    )
    axes[1].plot(
        range(1, epochs + 1), hist, label=f"lr={lr} ({tag})", color=col, linewidth=2.0
    )
axes[1].set_title("SGD (B=1) (Variable: Learning Rate)")
axes[1].set_xlabel("Epochs")
axes[1].legend(frameon=True, facecolor="white")

for (lr, hist), col in zip(histories_mbgd_lr.items(), colors, strict=False):
    tag = (
        "Diverging"
        if lr >= 0.2
        else "Optimal"
        if lr == 0.05
        else "Medium"
        if lr == 0.01
        else "Small"
    )
    axes[2].plot(
        range(1, epochs + 1), hist, label=f"lr={lr} ({tag})", color=col, linewidth=2.0
    )
axes[2].set_title("Mini-Batch GD (B=64) (Variable: Learning Rate)")
axes[2].set_xlabel("Epochs")
axes[2].legend(frameon=True, facecolor="white")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "study_learning_rate_influence.png"), dpi=300)
plt.show()

# %% [markdown]
# Аналіз чутливості до Learning Rate:
#
# Експериментально визначені границі стабільності для кожного алгоритму:
# - Batch GD є найбільш стійким і стабільно збігається аж до alpha = 0.20 (розходиться при alpha >= 1.0).
# - Mini-Batch GD (B=64) має оптимальну збіжність при alpha = 0.05 (розходиться при alpha >= 0.25).
# - SGD (B=1) вимагає суттєво меншого кроку alpha = 0.001 (розходиться вже при alpha >= 0.01).
#
# Пояснення різниці у чутливості:
# За одну епоху алгоритми виконують принципово різну кількість оновлень параметрів:
# 1. Batch GD: 1 оновлення за епоху (сумарний зсув 1 * alpha * gradient).
# 2. Mini-Batch GD (B=64): 16 оновлень за епоху (сумарний зсув 16 * alpha * gradient).
# 3. SGD (B=1): 1021 оновлення за епоху (сумарний зсув 1021 * alpha * gradient).
#
# Якщо швидкість навчання перевищує допустиму межу стабільності, алгоритм перелітає дно квадратичної чаші функції втрат і експоненційно прямує до нескінченності.

# %% [markdown]
# ### 2.3.3 Експеримент 2: Вплив розміру батчу порівняно з Baseline Batch GD
#
# В ізольованому експерименті оцінюємо, як розмір батчу (B = 16, 64, 128, 256 при фіксованому lr = 0.02) впливає на швидкість збіжності порівняно з еталонним Batch GD (B = 1021, lr = 0.05).

# %%
# 1. Compute the Optimal Baseline Batch GD trajectory (lr = 0.05, B = 1021)
model_bgd_opt = LRModel(lr=0.05, weight_vector=np.zeros(num_features), bias=0.0)
baseline_bgd_history = model_bgd_opt.train(
    X_train, y_train, epochs=epochs, sampler=BatchGDSampler()
)

# 2. Mini-Batch GD across multiple batch sizes B ∈ [16, 64, 128, 256] with lr = 0.02
mb_sizes = [16, 64, 128, 256]
mb_histories: dict[int, list[float]] = {}
for bs in mb_sizes:
    m = LRModel(lr=0.02, weight_vector=np.zeros(num_features), bias=0.0)
    mb_histories[bs] = m.train(
        X_train,
        y_train,
        epochs=epochs,
        sampler=MiniBatchGDSampler(batch_size=bs, random_seed=42),
    )

# Plot Experiment 2: Mini-Batch GD Sizes vs. Baseline Batch GD
plt.figure(figsize=(11, 6))

# Plot Baseline Batch GD
plt.plot(
    range(1, epochs + 1),
    baseline_bgd_history,
    label="Baseline: Batch GD (lr=0.05, B=1021, 1 step/ep)",
    color="#2c3e50",
    linewidth=2.8,
)

# Plot Mini-Batch Sizes
mb_colors = ["#e67e22", "#27ae60", "#2980b9", "#8e44ad"]
for (bs, hist), col in zip(mb_histories.items(), mb_colors, strict=False):
    steps_per_epoch = int(np.ceil(len(X_train) / bs))
    plt.plot(
        range(1, epochs + 1),
        hist,
        label=f"Mini-Batch B={bs} (lr=0.02, {steps_per_epoch} steps/ep)",
        color=col,
        linewidth=2.0,
    )

plt.title("Impact of Batch Size on Convergence vs. Baseline Batch GD")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE / 2)")
plt.yscale("log")
plt.ylim(bottom=1e8, top=1e12)
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "study_batch_size_influence.png"), dpi=300)
plt.show()

# %% [markdown]
# Аналіз компромісу вибору розміру батчу:
#
# 1. Швидкість збіжності за епохами:
#    Менші батчі (B = 16, 64) досягають низької помилки за значно меншу кількість епох порівняно з повним Batch GD, оскільки здійснюють 64 та 16 кроків оновлення за епоху.
#
# 2. Неперервний перехід масштабів:
#    Зі збільшенням розміру батчу (B = 128 -> B = 256) траєкторія спуску плавно наближається до базової кривої повного Batch GD.
#
# 3. Апаратна та алгоритмічна ефективність:
#    Хоча B = 16 потребує найменше епох, розмір батчу B = 64 забезпечує найкращий баланс паралелізму матричного множення (BLAS) та швидкості збіжності за реальним часом виконання.

# %% [markdown]
# ## 2.4 Етап 4. Оцінка якості та коефіцієнт детермінації ($R^2$)
#
# Якість натренованих моделей оцінюється на вибірках Train, Validation та Test за допомогою метрик:
# - Коефіцієнт детермінації:
#   $$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$
# - Середньоквадратична помилка (RMSE у доларах):
#   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
# - Середня абсолютна помилка (MAE у доларах):
#   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
#
# У наступній комірці реалізовано функції для розрахунку зазначених метрик, обчислення VIF та побудова зведеної таблиці результатів.


# %%
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def vif_score(X: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    vif_values: list[float] = []
    _, p = X.shape
    for j in range(p):
        y_j = X[:, j]
        X_other = np.delete(X, j, axis=1)
        # Solve least squares for feature j regressed on all other features
        weights, _, _, _ = np.linalg.lstsq(X_other, y_j, rcond=None)
        y_pred = X_other @ weights
        ss_res = float(np.sum((y_j - y_pred) ** 2))
        ss_tot = float(np.sum((y_j - np.mean(y_j)) ** 2))
        r2_j = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        r2_j = float(np.clip(r2_j, 0.0, 0.9999))
        vif = float(1.0 / (1.0 - r2_j))
        vif_values.append(vif)

    return pd.DataFrame(
        {
            "Feature": feature_names,
            "VIF": vif_values,
        }
    ).sort_values(by="VIF", ascending=False)


# Evaluation across all models and all 3 splits
splits = [
    ("Train", X_train, y_train),
    ("Validation", X_val, y_val),
    ("Test", X_test, y_test),
]

models = [
    ("Batch GD", model_bgd),
    ("SGD", model_sgd),
    ("Mini-Batch GD", model_mbgd),
]

eval_results = []
for model_name, model in models:
    for split_name, X_split, y_split in splits:
        y_pred = model.predict(X_split)
        r2 = r2_score(y_split, y_pred)
        rmse = rmse_score(y_split, y_pred)
        mae = mae_score(y_split, y_pred)
        eval_results.append(
            {
                "Model": model_name,
                "Split": split_name,
                "R^2 Score": r2,
                "RMSE ($)": rmse,
                "MAE ($)": mae,
            }
        )

eval_df = pd.DataFrame(eval_results)
print("=" * 72)
print("                    MODEL EVALUATION SUMMARY TABLE")
print("=" * 72)
print(f"{'Model':<15} {'Split':<12} {'R^2 Score':<12} {'RMSE ($)':<16} {'MAE ($)':<14}")
print("-" * 72)
for _, row in eval_df.iterrows():
    print(
        f"{row['Model']:<15} {row['Split']:<12} {row['R^2 Score']:<12.4f} "
        f"${row['RMSE ($)']:<15,.2f} ${row['MAE ($)']:<13,.2f}"
    )
print("=" * 72)

# %% [markdown]
# ### 2.4.1 Візуалізація точності: фактичні vs. прогнозовані ціни (Parity Plot)
#
# Будуємо графік розсіювання (Parity Plot) для оцінки якості моделі на відкладеній тестовій вибірці.
# Червона пунктирна лінія y = x відповідає ідеальному прогнозу.

# %%
# Visualizing Model Generalization: Actual vs. Predicted on Test Set (Parity Plot)
y_test_pred_mbgd = model_mbgd.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred_mbgd)
test_rmse = rmse_score(y_test, y_test_pred_mbgd)

plt.figure(figsize=(9, 7))
plt.scatter(
    y_test,
    y_test_pred_mbgd,
    color="#2980b9",
    alpha=0.65,
    edgecolors="none",
    label="Test Predictions",
)

# Perfect Prediction Reference Line (y = x)
min_val = min(float(np.min(y_test)), float(np.min(y_test_pred_mbgd)))
max_val = max(float(np.max(y_test)), float(np.max(y_test_pred_mbgd)))
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="#e74c3c",
    linestyle="--",
    linewidth=2.0,
    label="Perfect Fit (y = x)",
)

plt.title(
    f"Actual vs. Predicted SalePrice on Test Set (Mini-Batch GD)\n$R^2 = {test_r2:.4f}$ | $\\text{{RMSE}} = \\${test_rmse:,.2f}$"
)
plt.xlabel("Actual SalePrice ($)")
plt.ylabel("Predicted SalePrice ($)")
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "actual_vs_predicted_test.png"), dpi=300)
plt.show()

# %% [markdown]
# ### 2.4.2 Теоретичні питання та аналіз метрик
#
# 1. Статистичний зміст коефіцієнта детермінації R^2:
#    - R^2 вимірює частку дисперсії цільової змінної y, яка пояснюється лінійною залежністю від обраних факторів.
#    - R^2 = 1.0 відповідає ідеальній відповідності моделі даним (SS_res = 0).
#    - R^2 = 0.0 відповідає константній базовій моделі, яка для будь-якого будинку прогнозує середнє арифметичне.
#    - Значення R^2 близько 0.86 означає, що модель пояснює 86% загальних варіацій цін нерухомості в Еймсі.
#
# 2. Чи може R^2 бути від'ємним та за яких умов?
#    - Так, R^2 стає від'ємним (R^2 < 0), коли сума квадратів залишків перевищує загальну дисперсію вибірки (SS_res > SS_tot).
#    - Це трапляється, якщо прогнози моделі на нових даних гірші за просте константне середнє.
#    - Типові причини: катастрофічне перенавчання (overfitting) або відсутність зсуву (bias) при навчанні на даних з відмінним середнім рівнем.
#
# 3. Порівняння R^2 на Train та Test (Діагностика стану моделі):
#    - Train R^2 = 0.807 проти Test R^2 = 0.859.
#    - Близькі значення метрик на всіх трьох підвибірках свідчать про здоровий стан моделі без ознак перенавчання (коли R^2 на Train суттєво вищий за Test) або недонавчання.
#
# 4. Порівняння якості трьох варіантів градієнтного спуску:
#    - Batch GD: Test R^2 = 0.8578, RMSE = $26,993.11
#    - SGD: Test R^2 = 0.8582, RMSE = $26,951.03
#    - Mini-Batch GD: Test R^2 = 0.8591, RMSE = $26,869.60
#    - Усі три оптимізатори демонструють практично однакову фінальну точність, оскільки цільова квадратична функція втрат є строго опуклою (convex) і має єдиний глобальний оптимум.

# %% [markdown]
# ## 2.5 Етап 5. Перевірка статистичних припущень
#
# Лінійна регресія є статистичною моделлю, оцінки якої спираються на припущення Гаусса-Маркова.
# Розраховуємо залишки на тестовій вибірці для перевірки виконання ключових гіпотез:
# $$\epsilon_i = y_i - \hat{y}_i$$
# $$\epsilon_i^* = \frac{\epsilon_i - \bar{\epsilon}}{\sigma_{\epsilon}} \quad (\text{Стандартизовані залишки})$$

# %%
# Calculate test set predictions and residuals
y_test_pred_eval = model_mbgd.predict(X_test)
residuals_test = y_test - y_test_pred_eval
residuals_mean = float(np.mean(residuals_test))
residuals_std = float(np.std(residuals_test))
std_residuals_test = (residuals_test - residuals_mean) / residuals_std

print("--- Residuals Summary Statistics ---")
print(f"Mean of Residuals:       ${residuals_mean:,.2f}")
print(f"Std of Residuals:        ${residuals_std:,.2f}")
print(f"Min Residual (Underpred): ${float(np.min(residuals_test)):,.2f}")
print(f"Max Residual (Overpred):  ${float(np.max(residuals_test)):,.2f}")

# %% [markdown]
# ### 2.5.1 Перевірка лінійності (Linearity: Residuals vs. Fitted Values)
#
# Будуємо графік залежності залишків від прогнозованих значень.
# У коректній лінійній моделі залишки мають бути симетрично і випадково розсіяні навколо нульової горизонтальної осі без систематичних вигинів.

# %%
plt.figure(figsize=(9, 6))
plt.scatter(
    y_test_pred_eval,
    residuals_test,
    color="#2980b9",
    alpha=0.65,
    edgecolors="none",
    label="Test Residuals",
)
plt.axhline(
    0, color="#e74c3c", linestyle="--", linewidth=2.0, label="Zero Error Line (y = 0)"
)

# Polynomial trend line to detect non-linear curvature
sort_idx = np.argsort(y_test_pred_eval)
fitted_sorted = y_test_pred_eval[sort_idx]
poly_weights = np.polyfit(y_test_pred_eval, residuals_test, deg=2)
poly_trend = np.poly1d(poly_weights)(fitted_sorted)

plt.plot(
    fitted_sorted,
    poly_trend,
    color="#f39c12",
    linewidth=2.5,
    label="Residual Trend Curve",
)

plt.title("Assumption 1: Linearity Diagnostic (Residuals vs. Fitted Values)")
plt.xlabel("Fitted Values $\\hat{y}$ ($)")
plt.ylabel("Residuals $\\epsilon = y - \\hat{y}$ ($)")
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "assumptions_residuals_vs_fitted.png"), dpi=300)
plt.show()

# %% [markdown]
# Висновок щодо лінійності:
# Припущення про лінійність загалом виконується в діапазоні середніх цін ($100,000 – $300,000), де залишки симетрично розподілені навколо нульової осі. Проте для елітної нерухомості (> $350,000) спостерігається вигин тренду вгору.
#
# Це означає, що для дорогих маєтків реальна вартість зростає швидше за лінійну комбінацію площі та якості. Для усунення цього ефекту на практиці використовують логарифмування цільової змінної log(SalePrice) або додавання квадратичних членів.

# %% [markdown]
# ### 2.5.2 Перевірка нормальності залишків (Normality: Q-Q Plot та гістограма)
#
# Перевіряємо форму розподілу залишків за допомогою:
# 1. Гістограми залишків з накладенням теоретичної нормальної кривої щільності.
# 2. Квантиль-квантиль графіка (Q-Q plot) на чистому NumPy із порівнянням емпіричних квантилів проти стандартного нормального розподілу.

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Histogram with fitted Normal PDF
sns.histplot(
    residuals_test,
    kde=True,
    ax=axes[0],
    color="#3498db",
    stat="density",
    bins=25,
)
x_pdf_range = np.linspace(float(residuals_test.min()), float(residuals_test.max()), 200)
theoretical_pdf = (1.0 / (residuals_std * np.sqrt(2 * np.pi))) * np.exp(
    -0.5 * ((x_pdf_range - residuals_mean) / residuals_std) ** 2
)
axes[0].plot(
    x_pdf_range,
    theoretical_pdf,
    color="#e74c3c",
    linewidth=2.5,
    label="Theoretical Normal PDF",
)
axes[0].set_title("Residuals Distribution vs. Theoretical Normal PDF")
axes[0].set_xlabel("Residuals ($)")
axes[0].set_ylabel("Density")
axes[0].legend(frameon=True, facecolor="white")

# 2. Quantile-Quantile (Q-Q) Plot (Pure NumPy Implementation)
n_samples = len(std_residuals_test)
sorted_residuals = np.sort(std_residuals_test)
probs = (np.arange(1, n_samples + 1) - 0.5) / n_samples

# Pure NumPy: Compute theoretical standard normal quantiles using high-sample reference distribution
rng_diag = np.random.default_rng(42)
ref_normal = rng_diag.normal(loc=0.0, scale=1.0, size=200_000)
theoretical_quantiles = np.percentile(ref_normal, probs * 100.0)

axes[1].scatter(
    theoretical_quantiles,
    sorted_residuals,
    color="#2980b9",
    alpha=0.7,
    label="Sample Quantiles",
)
axes[1].plot(
    theoretical_quantiles,
    theoretical_quantiles,
    color="#e74c3c",
    linestyle="--",
    linewidth=2.0,
    label="Theoretical Normal Line (y = x)",
)
axes[1].set_title("Q-Q Plot (Quantile-Quantile)")
axes[1].set_xlabel("Theoretical Standard Normal Quantiles")
axes[1].set_ylabel("Sample Standardized Residuals")
axes[1].legend(frameon=True, facecolor="white")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "assumptions_normality_qq.png"), dpi=300)
plt.show()

# %% [markdown]
# Висновок щодо нормальності залишків:
# Припущення про нормальність залишків порушується у верхньому хвості розподілу (правий хвіст відхиляється вгору на Q-Q графіку через поодинокі дорогі будинки з високими похибками).
#
# Це означає, що точкові оцінки коефіцієнтів залишаються незміщеними та спроможними (згідно з теоремою Гаусса-Маркова), проте класичні довірчі інтервали та параметричні тести значущості можуть бути менш точними без нормалізуючого логарифмування ціни.

# %% [markdown]
# ### 2.5.3 Перевірка гомоскедастичності (Homoscedasticity: Scale-Location Plot)
#
# Гомоскедастичність передбачає сталість дисперсії випадкових похибок Var(e | X) = sigma^2 на всіх рівнях цін.
# Діагностика здійснюється за допомогою Scale-Location графіка:
# - Вісь Y: корінь з модуля стандартизованих залишків sqrt(|e*|)
# - Вісь X: прогнозовані значення ціни y_hat

# %%
sqrt_abs_std_residuals = np.sqrt(np.abs(std_residuals_test))

plt.figure(figsize=(9, 6))
plt.scatter(
    y_test_pred_eval,
    sqrt_abs_std_residuals,
    color="#8e44ad",
    alpha=0.65,
    edgecolors="none",
    label="$\\sqrt{|\\text{Standardized Residuals}|}$",
)

# Trend Line for variance
poly_var_weights = np.polyfit(y_test_pred_eval, sqrt_abs_std_residuals, deg=1)
poly_var_trend = np.poly1d(poly_var_weights)(fitted_sorted)

plt.plot(
    fitted_sorted,
    poly_var_trend,
    color="#e74c3c",
    linewidth=2.5,
    label="Variance Trend Line",
)

plt.title("Assumption 3: Homoscedasticity Diagnostic (Scale-Location Plot)")
plt.xlabel("Fitted Values $\\hat{y}$ ($)")
plt.ylabel("$\\sqrt{|\\text{Standardized Residuals}|}$")
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(
    os.path.join(plots_dir, "assumptions_homoscedasticity_scale_location.png"),
    dpi=300,
)
plt.show()

# %% [markdown]
# Висновок щодо гомоскедастичності:
# Припущення про гомоскедастичність порушується (наявна гетероскедастичність): лінія тренду на Scale-Location графіку має помітний висхідний нахил, а дисперсія похибок зростає разом зі збільшенням вартості будинку.
#
# Це означає, що для доступних будинків ($100,000) похибка становить приблизно +/- $15,000, тоді як для елітних ($500,000) вона досягає +/- $60,000. Внаслідок цього звичайний МНК надає надмірну вагу дорогим об'єктам.

# %% [markdown]
# ### 2.5.4 Перевірка на мультиколінеарність (VIF та кореляції)
#
# Мультиколінеарність виникає, коли предиктори лінійно залежні між собою.
# Оцінюємо коефіцієнт інфляції дисперсії (Variance Inflation Factor):
# $$\text{VIF}_j = \frac{1}{1 - R_j^2}$$
#
# Інтерпретація порогів VIF:
# - VIF < 2.5: низький рівень колінеарності (безпечно).
# - 2.5 <= VIF < 5.0: помірний рівень колінеарності.
# - VIF >= 5.0: високий рівень колінеарності (підвищена дисперсія оцінок коефіцієнтів).

# %%
feature_names = list(X_train_raw.columns)
num_ordinal_cols = selected_numeric + [c + "_encoded" for c in selected_ordinal]
num_ordinal_indices = [
    feature_names.index(c) for c in num_ordinal_cols if c in feature_names
]

vif_num_table = vif_score(
    X_train[:, num_ordinal_indices], [feature_names[i] for i in num_ordinal_indices]
)

print("=" * 65)
print("     NUMERICAL & ORDINAL FEATURES VARIANCE INFLATION FACTOR")
print("=" * 65)
print(f"{'Feature':<35} {'VIF':<15} {'Status':<15}")
print("-" * 65)
for _, row in vif_num_table.iterrows():
    status = (
        "High (>=5)"
        if row["VIF"] >= 5.0
        else "Moderate"
        if row["VIF"] >= 2.5
        else "Low"
    )
    print(f"{row['Feature']:<35} {row['VIF']:<15.2f} {status:<15}")
print("=" * 65)

# Plot VIF for Numerical Predictors
plt.figure(figsize=(10, 5))
vif_plot_df = vif_num_table.sort_values(by="VIF", ascending=True)
bars = plt.barh(vif_plot_df["Feature"], vif_plot_df["VIF"], color="#34495e")
plt.axvline(
    5.0,
    color="#e74c3c",
    linestyle="--",
    linewidth=2.0,
    label="High Collinearity Threshold (VIF = 5.0)",
)
plt.axvline(
    2.5,
    color="#f39c12",
    linestyle=":",
    linewidth=2.0,
    label="Moderate Collinearity Threshold (VIF = 2.5)",
)
plt.title("Numerical & Ordinal Features Variance Inflation Factor (VIF)")
plt.xlabel("VIF Value")
plt.legend(frameon=True, facecolor="white")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "assumptions_multicollinearity_vif.png"), dpi=300)
plt.show()

# %% [markdown]
# Висновок щодо мультиколінеарності:
# Припущення про повну незалежність предикторів частково порушується:
# 1. Числові ознаки: найвищий VIF мають OverallQual (VIF = 3.10) та GrLivArea (VIF = 2.49), оскільки якість і житлова площа природно пов'язані з роком будівництва та кількістю кімнат.
# 2. Категоріальні змінні: одночасне кодування всіх категорій номінальних ознак створює пастку даммі-змінних (сума колонок дорівнює 1.0).
#
# Вплив на стабільність та інтерпретацію коефіцієнтів:
# Мультиколінеарність не знижує прогнозна точність моделі (Test R^2 = 0.859), проте збільшує дисперсію оцінок ваг Var(w_j). Це ускладнює фізичну інтерпретацію окремих вагових коефіцієнтів, які ділять спільний інформаційний сигнал.

# %% [markdown]
# ## 2.6 Етап 6. Завдання для глибокого розуміння: Експеримент В (Ridge L2-регуляризація)
#
# Оскільки на Етапі 5 виявлено помірну мультиколінеарність між структурними ознаками, реалізуємо Ridge регресію (L2-регуляризацію) на чистому NumPy для стабілізації оцінок коефіцієнтів.
#
# Математичний апарат Ridge регресії:
# 1. Функція втрат Ridge:
#    $$J_{\text{Ridge}}(\mathbf{w}, b) = \frac{1}{2n} \|\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}\|_2^2 + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
#    (Зміщення b не штрафується, щоб не зміщувати середній ціновий рівень).
# 2. Аналітичні градієнти:
#    $$\nabla_{\mathbf{w}} J_{\text{Ridge}} = \frac{1}{n} \mathbf{X}^T (\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y}) + \lambda \mathbf{w}$$
#    $$\nabla_b J_{\text{Ridge}} = \frac{1}{n} \mathbf{1}^T (\mathbf{X}\mathbf{w} + b\mathbf{1} - \mathbf{y})$$
# 3. Правило оновлення ваг (Weight Decay):
#    $$\mathbf{w}^{(t+1)} = (1 - \alpha \lambda) \mathbf{w}^{(t)} - \alpha \nabla_{\mathbf{w}} J_{\text{MSE}}$$
#
# У наступній комірці реалізовано клас `RidgeLRModel` на чистому NumPy.


# %%
class RidgeLRModel:
    """Pure NumPy Linear Regression with L2 Regularization (Ridge)."""

    lr: float
    alpha_reg: float
    weights: np.ndarray
    bias: float

    def __init__(
        lr_model,  # pyright: ignore
        lr: float = 0.02,
        alpha_reg: float = 0.1,
        weight_vector: np.ndarray | None = None,
        bias: float = 0.0,
    ):
        lr_model.lr = lr
        lr_model.alpha_reg = alpha_reg
        lr_model.weights = (
            weight_vector if weight_vector is not None else np.zeros(0, dtype=float)
        )
        lr_model.bias = bias

    def predict(lr_model, feature_matrix: np.ndarray) -> np.ndarray:  # pyright: ignore
        return (feature_matrix @ lr_model.weights) + lr_model.bias

    def compute_loss(
        lr_model,  # pyright: ignore
        feature_matrix: np.ndarray,
        target_vector: np.ndarray,
    ) -> float:
        n = len(target_vector)
        predictions = lr_model.predict(feature_matrix)
        mse_loss = float(np.sum((predictions - target_vector) ** 2) / (2 * n))
        l2_penalty = float(0.5 * lr_model.alpha_reg * np.sum(lr_model.weights**2))
        return mse_loss + l2_penalty

    def compute_gradients(
        lr_model,  # pyright: ignore
        feature_matrix: np.ndarray,
        target_vector: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        n = len(target_vector)
        errors = lr_model.predict(feature_matrix) - target_vector
        grad_w = (feature_matrix.T @ errors) / n + (
            lr_model.alpha_reg * lr_model.weights
        )
        grad_b = float(np.sum(errors) / n)
        return grad_w, grad_b

    def train(
        lr_model,  # pyright: ignore
        feature_matrix: np.ndarray,
        target_vector: np.ndarray,
        epochs: int,
        sampler: BatchSampler,
    ) -> list[float]:
        loss_history: list[float] = []

        for _ in range(epochs):
            for X_batch, y_batch in sampler.get_batches(feature_matrix, target_vector):
                grad_w, grad_b = lr_model.compute_gradients(X_batch, y_batch)
                lr_model.weights -= lr_model.lr * grad_w
                lr_model.bias -= lr_model.lr * grad_b

            epoch_loss = lr_model.compute_loss(feature_matrix, target_vector)
            loss_history.append(epoch_loss)

        return loss_history


# %% [markdown]
# ### 2.6.1 Дослідження впливу коефіцієнта регуляризації $\lambda$ на ваги ознак
#
# Навчаємо Ridge-модель при різних значеннях параметра штрафу lambda = 0.0, 0.01, 0.1, 1.0, 10.0, 50.0 та візуалізуємо стиснення вагових коефіцієнтів (Weight Shrinkage).

# %%
# Ridge Regularization Sweep across different lambda penalty values
lambda_values = [0.0, 0.01, 0.1, 1.0, 10.0, 50.0]
ridge_models: dict[float, RidgeLRModel] = {}
ridge_weights: dict[float, np.ndarray] = {}

for lmb in lambda_values:
    model_ridge = RidgeLRModel(
        lr=0.02,
        alpha_reg=lmb,
        weight_vector=np.zeros(num_features),
        bias=0.0,
    )
    model_ridge.train(
        X_train,
        y_train,
        epochs=100,
        sampler=MiniBatchGDSampler(batch_size=64, random_seed=42),
    )
    ridge_models[lmb] = model_ridge
    ridge_weights[lmb] = model_ridge.weights.copy()

# Compare weights for collinear pairs: GarageCars vs GarageArea, GrLivArea vs TotRmsAbvGrd
collinear_features_to_plot = [
    "OverallQual",
    "GrLivArea",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageArea",
    "1stFlrSF",
    "TotalBsmtSF",
    "YearBuilt",
]

collinear_indices = [
    feature_names.index(f) for f in collinear_features_to_plot if f in feature_names
]

plt.figure(figsize=(12, 6))
for idx, f_name in zip(collinear_indices, collinear_features_to_plot, strict=False):
    weights_path = [ridge_weights[lmb][idx] for lmb in lambda_values]
    plt.plot(
        range(len(lambda_values)),
        weights_path,
        marker="o",
        linewidth=2.2,
        label=f_name,
    )

plt.xticks(range(len(lambda_values)), [f"$\\lambda={lmb}$" for lmb in lambda_values])
plt.title("Impact of L2 Regularization (Ridge) on Feature Weights $\\mathbf{w}$")
plt.xlabel("Regularization Strength ($\\lambda$)")
plt.ylabel("Weight Magnitude ($w_j$)")
plt.legend(frameon=True, facecolor="white", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "deep_understanding_ridge_weights.png"), dpi=300)
plt.show()

# %% [markdown]
# Аналіз результатів Ridge-регуляризації:
#
# 1. Стиснення ваг та зменшення дисперсії:
#    Зі збільшенням сили штрафу lambda від 0.0 до 50.0 ваги скорельованих ознак (OverallQual, GrLivArea, GarageCars, YearBuilt) плавно зменшуються в бік нуля, знижуючи чутливість моделі до вибіркових шумів.
#
# 2. Стабілізація градієнтного спуску при поганій обумовленості:
#    У звичайній лінійній регресії (lambda = 0) матриця Гессе X^T X має близькі до нуля власні значення через мультиколінеарність, через що ландшафт втрат має вигляд витягнутого яру. Додавання штрафу lambda * I зміщує всі власні значення вгору:
#    $$\lambda_i(\mathbf{X}^T\mathbf{X} + \lambda \mathbf{I}) = \lambda_i(\mathbf{X}^T\mathbf{X}) + \lambda > 0$$
#    Це суттєво покращує число обумовленості гессіана kappa = (lambda_max + lambda) / (lambda_min + lambda), прискорює та стабілізує траєкторію градієнтного спуску.
