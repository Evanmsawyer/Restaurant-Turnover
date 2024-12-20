import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_excel(r'D:\Final Deep\output_plots\all_model_results.xlsx')

# Verify data loading
print(df.head())

# Set a general style
sns.set_style("whitegrid")

# 1. Compare test F1 scores by model and target
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='model_name', y='test_f1', hue='target')
plt.title("Test F1 Scores by Model and Target")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("test_f1_by_model_target.png")
plt.show()

# 2. Grouped barplot to compare train, val, and test F1 for each model
f1_cols = ['train_f1', 'val_f1', 'test_f1']
df_f1_melted = df.melt(id_vars=['target', 'model_name'], value_vars=f1_cols, 
                       var_name='set_type', value_name='f1_score')

plt.figure(figsize=(12, 6))
sns.barplot(data=df_f1_melted, x='model_name', y='f1_score', hue='set_type')
plt.title("F1 Score Comparison by Model and Data Split (Train/Val/Test)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("f1_comparison_by_model.png")
plt.show()

# 3. Correlation plot focusing on F1 metrics along with others (optional)
# Adjust metric_cols as needed. Here we keep all metrics, but highlight F1.
metric_cols = ['train_f1', 'val_f1', 'test_f1',
               'train_mse', 'val_mse', 'test_mse', 
               'train_r2', 'val_r2', 'test_r2', 
               'train_acc', 'val_acc', 'test_acc',
               'cv_score_mean', 'cv_score_std']

corr = df[metric_cols].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Performance Metrics (Including F1)")
plt.tight_layout()
plt.savefig("correlation_metrics_f1.png")
plt.show()
