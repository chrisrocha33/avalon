import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

server = 'localhost'
port = '5432'
database = 'avalon'
username = 'admin'
password = 'password!'

conn_str = f'postgresql+psycopg2://{username}:{password}@{server}:{port}/{database}'
engine = create_engine(conn_str, future=True)   



model_data = pd.read_sql_query("SELECT * FROM ml_spx_data", engine)

# Set the index to the date column if it exists, otherwise use the default index
if 'index' in model_data.columns:
    model_data['index'] = pd.to_datetime(model_data['index'])
    model_data = model_data.set_index('index')
elif 'date' in model_data.columns:
    model_data['date'] = pd.to_datetime(model_data['date'])
    model_data = model_data.set_index('date')

spx = model_data["GSPC_log_return_next_period"]

# Regime via sliding window hysteresis: keep prior state inside a neutral band
window_size = 5
up_threshold = 0.003   # ~0.3% aggregated move
down_threshold = -0.003

regimes = []
window_sum = 0.0
prev_regime = 0
for i in range(len(spx)):
    window_sum += spx.iloc[i]
    if i >= window_size:
        window_sum -= spx.iloc[i - window_size]

    if i < window_size - 1:
        regimes.append(0)
        prev_regime = 0
        continue

    if window_sum > up_threshold:
        current_regime = 1
    elif window_sum < down_threshold:
        current_regime = -1
    else:
        current_regime = prev_regime

    regimes.append(current_regime)
    prev_regime = current_regime

model_data['regime'] = pd.Series(regimes, index=model_data.index).astype(int)

# Build feature matrix excluding non-features and datetime columns
# Remove any datetime columns and the target variables
columns_to_drop = ['GSPC_log_return_next_period', 'regime']
datetime_columns = model_data.select_dtypes(include=['datetime64[ns]']).columns.tolist()
columns_to_drop.extend(datetime_columns)

X = model_data.drop(columns=columns_to_drop)
y = model_data['regime']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {list(X.columns)[:10]}...")  # Show first 10 columns

# Print simple class distribution (loop-based)
counts = {-1: 0, 0: 0, 1: 0}
for r in regimes:
    if r in counts:
        counts[r] += 1
    else:
        counts[r] = 1
print(f"Regime counts: {counts}")

# Scale features only (not regime) right before TTS
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"X length : {len(X)}")
print(f"y length : {len(y)}")

# Check if we have enough samples of each class for stratified split
unique_classes = []
for val in y:
    if val not in unique_classes:
        unique_classes.append(val)

min_class_count = float('inf')
for cls in unique_classes:
    count = sum(1 for val in y if val == cls)
    if count < min_class_count:
        min_class_count = count

print(f"Minimum class count: {min_class_count}")
print(f"Unique classes: {unique_classes}")

# Use stratified split if we have enough samples, otherwise regular split
if min_class_count >= 2:  # Need at least 2 samples per class for stratified split
    print("Using stratified train-test split to ensure class representation...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X_scaled, y))
    x_train, x_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
else:
    print("Using regular train-test split (insufficient samples for stratified split)...")
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Use balanced class weights to handle imbalanced classes
lr_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_model.fit(x_train, y_train)
print("✓ Trained Logistic Regression with balanced class weights")

y_test_pred = lr_model.predict(x_test)

# Check class distribution in test set before showing classification report
print(f"\n{'='*60}")
print("TEST SET CLASS DISTRIBUTION")
print(f"{'='*60}")
test_counts = {-1: 0, 0: 0, 1: 0}
for val in y_test:
    if val in test_counts:
        test_counts[val] += 1
    else:
        test_counts[val] = 1

print(f"Test set regime distribution:")
print(f"  Down (-1): {test_counts.get(-1, 0)} samples")
print(f"  Neutral (0): {test_counts.get(0, 0)} samples") 
print(f"  Up (1): {test_counts.get(1, 0)} samples")
print(f"Total test samples: {len(y_test)}")

# Check predicted class distribution
pred_counts = {-1: 0, 0: 0, 1: 0}
for val in y_test_pred:
    if val in pred_counts:
        pred_counts[val] += 1
    else:
        pred_counts[val] = 1

print(f"\nPredicted regime distribution:")
print(f"  Down (-1): {pred_counts.get(-1, 0)} samples")
print(f"  Neutral (0): {pred_counts.get(0, 0)} samples")
print(f"  Up (1): {pred_counts.get(1, 0)} samples")

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT")
print(f"{'='*60}")
print(classification_report(y_test, y_test_pred, zero_division=0))

print(f"\nOverall Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# Comparison plot: Actual vs Predicted regimes on the test set
# Build lists using loops and sort by Date to make the plot readable
# Use the index (which should be datetime) for plotting
dates = []
actual = []
predicted = []

# Get test dates using the test indices
for i in range(len(y_test)):
    # Get the actual index value from y_test
    test_idx = y_test.index[i]
    # Find the position of this index in the original model_data
    pos_in_original = model_data.index.get_loc(test_idx)
    # Get the corresponding date
    test_date = model_data.index[pos_in_original]
    
    dates.append(test_date)
    actual.append(int(y_test.iloc[i]))
    predicted.append(int(y_test_pred[i]))

paired = []
for i in range(len(dates)):
    paired.append((dates[i], actual[i], predicted[i]))

paired.sort(key=lambda t: t[0])

plot_dates = []
actual_sorted = []
pred_sorted = []
for d, a, p in paired:
    plot_dates.append(d)
    actual_sorted.append(a)
    pred_sorted.append(p)

# Create a more effective visualization for regime classification
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

# 1. Scatter plot with different colors for each regime
colors_actual = []
colors_pred = []
for i in range(len(actual_sorted)):
    if actual_sorted[i] == -1:
        colors_actual.append('red')
    elif actual_sorted[i] == 0:
        colors_actual.append('gray')
    else:
        colors_actual.append('green')
    
    if pred_sorted[i] == -1:
        colors_pred.append('red')
    elif pred_sorted[i] == 0:
        colors_pred.append('gray')
    else:
        colors_pred.append('green')

ax1.scatter(plot_dates, actual_sorted, c=colors_actual, alpha=0.7, s=20, label='Actual')
ax1.scatter(plot_dates, pred_sorted, c=colors_pred, alpha=0.4, s=10, marker='x', label='Predicted')
ax1.set_title('Regime Classification: Actual vs Predicted (Scatter)')
ax1.set_ylabel('Regime')
ax1.set_yticks([-1, 0, 1])
ax1.set_yticklabels(['Down', 'Neutral', 'Up'])
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Bar plot showing regime distribution
regime_counts_actual = {-1: 0, 0: 0, 1: 0}
regime_counts_pred = {-1: 0, 0: 0, 1: 0}
for i in range(len(actual_sorted)):
    regime_counts_actual[actual_sorted[i]] += 1
    regime_counts_pred[pred_sorted[i]] += 1

regimes = ['Down', 'Neutral', 'Up']
actual_counts = [regime_counts_actual[-1], regime_counts_actual[0], regime_counts_actual[1]]
pred_counts = [regime_counts_pred[-1], regime_counts_pred[0], regime_counts_pred[1]]

x = np.arange(len(regimes))
width = 0.35

ax2.bar(x - width/2, actual_counts, width, label='Actual', color=['red', 'gray', 'green'], alpha=0.7)
ax2.bar(x + width/2, pred_counts, width, label='Predicted', color=['red', 'gray', 'green'], alpha=0.4)
ax2.set_title('Regime Distribution: Actual vs Predicted')
ax2.set_ylabel('Count')
ax2.set_xticks(x)
ax2.set_xticklabels(regimes)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Confusion matrix heatmap
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(actual_sorted, pred_sorted)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, 
            xticklabels=['Down', 'Neutral', 'Up'], 
            yticklabels=['Down', 'Neutral', 'Up'])
ax3.set_title('Confusion Matrix: Actual vs Predicted')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Print additional metrics
print(f"\n{'='*60}")
print("REGIME CLASSIFICATION ANALYSIS")
print(f"{'='*60}")
print(f"Total test samples: {len(actual_sorted)}")
print(f"Actual regime distribution:")
print(f"  Down (-1): {regime_counts_actual[-1]} ({regime_counts_actual[-1]/len(actual_sorted)*100:.1f}%)")
print(f"  Neutral (0): {regime_counts_actual[0]} ({regime_counts_actual[0]/len(actual_sorted)*100:.1f}%)")
print(f"  Up (1): {regime_counts_actual[1]} ({regime_counts_actual[1]/len(actual_sorted)*100:.1f}%)")
print(f"\nPredicted regime distribution:")
print(f"  Down (-1): {regime_counts_pred[-1]} ({regime_counts_pred[-1]/len(actual_sorted)*100:.1f}%)")
print(f"  Neutral (0): {regime_counts_pred[0]} ({regime_counts_pred[0]/len(actual_sorted)*100:.1f}%)")
print(f"  Up (1): {regime_counts_pred[1]} ({regime_counts_pred[1]/len(actual_sorted)*100:.1f}%)")

