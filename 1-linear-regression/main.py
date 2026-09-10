# %% [markdown]
# # Лабораторна робота №1: Лінійна регресія, оптимізація та статистичний аналіз
#
# Мета роботи - розібрати внутрішню математику та архітектуру лінійної регресії, реалізувати з нуля градієнтні методи оптимізації засобами `numpy`, дослідити припущення класичної регресійної моделі та провести аналіз залишків.

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
# ### Target Variable Analysis (`SalePrice`)
# Inspecting the distribution and descriptive statistics (mean, median, skewness) of `SalePrice`.

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
# **Висновки з аналізу цільової змінної:**
# 1. Розподіл `SalePrice` має виражену правосторонню асиметрію (Skewness $\approx 1.88$) — більшість будинків зосереджена у діапазоні $100{,}000 - $250{,}000, але є дорогі маєтки понад $500{,}000.
# 2. Логарифмування суттєво наближає розподіл до нормального, що корисно пам'ятати при аналізі залишків лінійної регресії.

# %% [markdown]
# ### 2. Розбиття вибірки (Train / Validation / Test)
#
# > **Теоретичне питання:** Чому ми повинні розбити дані **до**, а не після масштабування, кодування або інших операцій? Поясніть термін **Data Leakage**.
# >
# > **Відповідь:**
# > - **Data Leakage (витік даних)** виникає, коли інформація з валідаційної або тестової вибірки ненавмисно потрапляє у процес навчання моделі або підготовки ознак.
# > - Якщо обчислити середнє ($\mu$) та дисперсію ($\sigma$) для масштабування або частоти категорій на всьому датасеті, тестові дані вплинуть на ці параметри. У реальному житті модель отримує нові дані, яких вона ніколи не бачила.
# > - Масштабування чи відбір ознак на всій вибірці створює **оптимістичне зміщення (optimistic bias)**: модель показує завищені метрики на тесті, які не відтворяться на практиці.
# > - Тому будь-яка трансформація повинна навчатися (**fit**) **виключно на Train**, а потім застосовуватися (**transform**) до Validation та Test.


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
# ### 3. Відбір ознак (Feature Selection на Train-вибірці)
#
# Дослідимо кореляцію числових ознак із `SalePrice` та між собою на **навчальній вибірці**, щоб обрати інформативні змінні та виявити мультиколінеарність.

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
# **Спостереження щодо мультиколінеарності:**
# - `GarageCars` та `GarageArea` мають кореляцію $> 0.88$ (дублюють інформацію про гараж).
# - `TotalBsmtSF` та `1stFlrSF` мають кореляцію $> 0.81$ (площа першого поверху майже рівна площі підвалу).
# - `GrLivArea` та `TotRmsAbvGrd` мають кореляцію $> 0.82$ (чим більша площа, тим більше кімнат).
#
# **Вибір фінальних ознак:**
# 1. **Числові ознаки:**
#    - `OverallQual` (загальна якість матеріалів та оздоблення, найвищий зв'язок $r \approx 0.79$).
#    - `GrLivArea` (житлова площа над рівнем землі, $r \approx 0.71$).
#    - `TotalBsmtSF` (загальна площа підвалу, $r \approx 0.61$).
#    - `GarageCars` (місткість гаража, $r \approx 0.64$).
#    - `YearBuilt` (рік побудови будинку, $r \approx 0.52$).
#    - `FullBath` (кількість повних ванних кімнат, $r \approx 0.56$).
# 2. **Категоріальні ознаки:**
#    - `KitchenQual` (якість кухні — важливий фактор ціноутворення).
#    - `Neighborhood` (район — ключовий локаційний фактор у нерухомості).
#    - `CentralAir` (наявність центрального кондиціонера: Y/N).

# %% [markdown]
# ### 4. Кодування категоріальних ознак (Categorical Encoding)
#
# > **Теоретичне питання:** Як обробляти категорії, які з'являться на валідації/тесті, але яких не було в Train?
# >
# > **Відповідь:**
# > 1. **One-Hot Encoding**: Для невідомих категорій під час трансформації валідаційного/тестового набору відповідні бінарні стовпчики заповнюються нулями (`all zeros`, параметр `handle_unknown='ignore'`).
# > 2. **Ordinal Encoding**: Невідомим категоріям присвоюється значення за замовчуванням (наприклад, медіанний або найчастіший ранг у Train, або спеціальне значення `0` / `-1`).
# > 3. **Рідкісні категорії**: Категорії з частотою $< 1\%$ у Train можна заздалегідь об'єднати у категорію `'Other'`.

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
# ### 5. Масштабування ознак (Standardization / Z-Score)
#
# Стандартизуємо числові дані за формулою:
# $$z = \frac{x - \mu}{\sigma}$$
# Параметри $\mu_{\text{train}}$ та $\sigma_{\text{train}}$ обчислюються **виключно на навчальній вибірці (Train)** і застосовуються до Validation та Test.


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
        y: np.ndarray,  # pyright: ignore
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
# ### 2.3 Stage 3. Optimization: Comparing Batch GD, SGD, and Mini-batch GD
#
# We train the Linear Regression model using the three sampling strategies:
# 1. **Batch GD**: updates using all samples per epoch ($\alpha = 0.05$)
# 2. **SGD**: updates per single sample ($\alpha = 0.001$)
# 3. **Mini-batch GD**: updates per batch of size $B=64$ ($\alpha = 0.01$)

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

print("Training finished for all 3 optimization methods.")
print("Final Train Loss (MSE/2):")
print(f"  Batch GD:      {history_bgd[-1]:,.2f}")
print(f"  SGD:           {history_sgd[-1]:,.2f}")
print(f"  Mini-batch GD: {history_mbgd[-1]:,.2f}")

# %%
# Comparative plot: Cost Function vs. Epochs
plt.figure(figsize=(12, 6))
plt.plot(
    range(1, epochs + 1),
    history_bgd,
    label="Batch GD (lr=0.05)",
    color="royalblue",
    linewidth=2.2,
)
plt.plot(
    range(1, epochs + 1),
    history_sgd,
    label="SGD (lr=0.001, B=1)",
    color="crimson",
    linewidth=1.8,
    linestyle="--",
)
plt.plot(
    range(1, epochs + 1),
    history_mbgd,
    label="Mini-batch GD (lr=0.01, B=64)",
    color="forestgreen",
    linewidth=2.0,
)

plt.title("Cost Function vs. Epochs for 3 Gradient Descent Variants")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE / 2)")
plt.yscale("log")
plt.legend(frameon=True, facecolor="white", edgecolor="none")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "optimization_comparison.png"), dpi=300)
plt.show()

# %% [markdown]
# ### Optimization Behavior Analysis
#
# - **Most Stable Method:** Batch GD displays the smoothest, strictly monotonic descent because each gradient step is computed over the entire training set ($N=1021$).
# - **Largest Fluctuations:** SGD exhibits the most stochastic noise along its path because each step is based on only a single sample ($B=1$).
# - **Impact of Batch Size:**
#   - Small batch size (SGD) executes $N$ updates per epoch, but cannot leverage NumPy vectorization efficiently.
#   - Large batch size (Batch GD) executes 1 update per epoch with maximal vectorization, but requires more epochs to make initial rapid progress.
#   - Mini-batch GD ($B=64$) achieves the optimal trade-off: efficient matrix multiplication in BLAS/NumPy with fast, smooth convergence.
# - **Impact of Learning Rate ($\alpha$):**
#   - SGD requires a smaller learning rate ($\alpha \approx 0.001$) to prevent violent oscillations from single-sample updates.
#   - Batch GD can comfortably handle a larger learning rate ($\alpha \approx 0.05$).
#   - An excessively large learning rate causes loss explosion (divergence), while an excessively small rate slows convergence.
