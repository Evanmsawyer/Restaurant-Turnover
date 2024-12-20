import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tpot import TPOTRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
import os

if not os.path.exists('output_plots'):
    os.makedirs('output_plots')

def load_data():
    df = pd.read_excel('employee_survey_data.xlsx')
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    print("\nDataset Overview:")
    print("-----------------")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nMissing values per column:")
    print(df.isnull().sum().sort_values(ascending=False).head())
    
    return df

def convert_to_months(tenure):
    """Convert tenure strings to months."""
    if pd.isna(tenure):
        return None

    tenure = str(tenure).lower().strip()

    match = re.search(r"(\d+\.?\d*)\s*(years?|yrs?|months?|mos?|weeks?|days?)", tenure)
    if match:
        value = float(match.group(1)) 
        unit = match.group(2)       

        # Convert to months
        if "year" in unit or "yr" in unit:
            return int(value * 12)
        elif "month" in unit or "mo" in unit:
            return int(value)
        elif "week" in unit:
            return int(value / 4.345) 
        elif "day" in unit:
            return int(value / 30.437) 
    elif re.search(r"^\d+$", tenure):  
        return int(float(tenure) * 12)  
    else:
        return None  


def map_certifications(cert_text):
    """Map certifications to predefined categories."""
    if pd.isna(cert_text):
        return None

    cert_text = str(cert_text).lower().strip()

    if "bartend" in cert_text:
        return "Bartending"
    elif "food handling" in cert_text or "food safety" in cert_text or "food handler" in cert_text:
        return "Food Handling"
    elif "cpr" in cert_text:
        return "CPR"
    elif "hospitality" in cert_text:
        return "Hospitality Degree"
    elif "alcohol" in cert_text or "safe serve" in cert_text:
        return "Alcohol Service"
    else:
        return "Other"

def map_position(position_text):
    """Map position to predefined categories."""
    if pd.isna(position_text):
        return None

    position_text = str(position_text).lower().strip()

    if "manager" in position_text or "supervisor" in position_text or "director" in position_text:
        return "Manager"

    elif "server" in position_text or "wait" in position_text or "front" in position_text or \
         "cashier" in position_text or "host" in position_text or "greeter" in position_text:
        return "Front-of-House (FOH)"

    elif "bartend" in position_text or "mixologist" in position_text or "barista" in position_text:
        return "Bartender"

    elif "cook" in position_text or "chef" in position_text or "bus" in position_text or \
         "line cook" in position_text or "prep cook" in position_text or "sous chef" in position_text:
        return "Culinary (BOH)"

    elif "dishwasher" in position_text or "kitchen staff" in position_text or "cleaner" in position_text:
        return "Dishwasher"

    elif "delivery" in position_text or "driver" in position_text:
        return "Delivery"

    elif "maintenance" in position_text or "repair" in position_text or "janitor" in position_text:
        return "Maintenance"

    elif "admin" in position_text or "reception" in position_text or "office" in position_text:
        return "Administrative"

    elif "trainer" in position_text or "training" in position_text:
        return "Trainer"

    elif "barback" in position_text or "runner" in position_text or "expediter" in position_text:
        return "Support Staff"
    elif "security" in position_text or "bouncer" in position_text:
        return "Security"
    elif "event" in position_text or "planner" in position_text or "coordinator" in position_text:
        return "Event Staff"
    elif "owner" in position_text or "proprietor" in position_text:
        return "Owner"
    elif "cashier" in position_text:
        return "Cashier"
    elif "hostess" in position_text:
        return "Host"
    else:
        return "Other"


def standardize_tip_amount(tip_text):
    """Standardize tip amount to dollars per hour."""
    if pd.isna(tip_text):
        return None

    tip_text = str(tip_text).lower().strip()
    match = re.search(r"(\d+\.?\d*)", tip_text)
    if match:
        value = float(match.group(1))

        if "hour" in tip_text or "/hr" in tip_text or "per hour" in tip_text:
            return value
        elif "%" in tip_text or "percent" in tip_text:
            print(f"Warning: Percentage detected in '{tip_text}' - handling may require additional logic")
            return value  
        elif "$" in tip_text:
            return value
        else:
            return value
    elif "hourly" in tip_text:
        return 10.0
    else:
        return None  

def convert_industry_experience(experience):
    """Convert industry experience strings to months."""
    if pd.isna(experience):
        return None

    experience = str(experience).lower().strip()

    match = re.search(r"(\d+\.?\d*)\s*(years?|yrs?|months?|mos?|weeks?)?", experience)
    if match:
        value = float(match.group(1))
        unit = match.group(2)     
        if unit is None:  
            return int(value * 12)
        elif "year" in unit or "yr" in unit:
            return int(value * 12)
        elif "month" in unit or "mo" in unit:
            return int(value)
        elif "week" in unit:
            return int(value / 4.345) 
    else:
        return None 

def standardize_earned_tip(tip_text):
    """Standardize earned_tip responses to a consistent numerical format."""
    if pd.isna(tip_text):
        return None

    tip_text = str(tip_text).lower().strip()
    match_percent = re.search(r"(\d+\.?\d*)\s*%", tip_text)
    if match_percent:
        return float(match_percent.group(1)) / 100  

    match_hourly = re.search(r"(\d+\.?\d*)\s*\$? per hour", tip_text)
    if match_hourly:
        return float(match_hourly.group(1))
    if tip_text in ["none", "na", "n/a", "not sure", "unsure"]:
        return 0

    if "varies" in tip_text or "probably" in tip_text:
        return None  

    match_numeric = re.search(r"(\d+\.?\d*)", tip_text)
    if match_numeric:
        return float(match_numeric.group(1))

    return None

def clean_data(df):
    df_clean = df.copy()
    
    print("\nBefore cleaning:")
    print(f"Shape: {df_clean.shape}")
    
    if 'res_tenure' in df_clean.columns:
        df_clean['res_tenure'] = df_clean['res_tenure'].astype(str)  
        df_clean['res_tenure_months'] = df_clean['res_tenure'].apply(convert_to_months)
        print("\nNormalized tenure column:")
        print(df_clean[['res_tenure', 'res_tenure_months']].head())

    if 'res_industry' in df_clean.columns:
        df_clean['res_industry_months'] = df_clean['res_industry'].apply(convert_industry_experience)
        print("\nNormalized industry experience column:")
        print(df_clean[['res_industry', 'res_industry_months']].head())

    if 'cert' in df_clean.columns:
        df_clean['cert_category'] = df_clean['cert'].apply(map_certifications)
        print("\nMapped certifications column:")
        print(df_clean[['cert', 'cert_category']].head())

    if 'position' in df_clean.columns:
        df_clean['position_category'] = df_clean['position'].apply(map_position)
        print("\nMapped positions column:")
        print(df_clean[['position', 'position_category']].head())

    if 'res_industry' in df_clean.columns:
        df_clean['res_industry_months'] = df_clean['res_industry'].apply(convert_industry_experience)
        print("\nNormalized industry experience column:")
        print(df_clean[['res_industry', 'res_industry_months']].head())

    if 'tip_amount_1_TEXT' in df_clean.columns:
        df_clean['tip_amount_standardized'] = df_clean['tip_amount_1_TEXT'].apply(standardize_tip_amount)
        print("\nStandardized tip amounts:")
        print(df_clean[['tip_amount_1_TEXT', 'tip_amount_standardized']].head())

    if 'earned_tip' in df_clean.columns:
        df_clean['earned_tip_normalized'] = df_clean['earned_tip'].apply(standardize_earned_tip)
        print("\nStandardized earned_tip column:")
        print(df_clean[['earned_tip', 'earned_tip_normalized']].head())

    current_year = 2024
    df_clean['age'] = current_year - pd.to_numeric(df_clean['YEAR'], errors='coerce')
    
    categorical_cols = ['SEX', 'MAR', 'ETHNIC', 'edu']
    label_encoders = {}
    
    print("\nCategorical variable distributions:")
    for col in categorical_cols:
        if col in df_clean.columns:
            print(f"\n{col} value counts:")
            print(df_clean[col].value_counts())
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le
    
    print("\nMissing values before imputation:")
    print(df_clean.isnull().sum().sum())
    
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    
    missing_threshold = len(df_clean) * 0.8
    cols_before = len(df_clean.columns)
    df_clean = df_clean.dropna(axis=1, thresh=missing_threshold)
    cols_after = len(df_clean.columns)
    
    print(f"\nDropped {cols_before - cols_after} columns due to missing values")
    print(f"Shape after cleaning: {df_clean.shape}")
    
    remaining_missing = df_clean.isnull().sum().sum()
    print(f"Remaining missing values: {remaining_missing}")
    
    return df_clean, label_encoders

def analyze_data(df):
    turnover_vars = ['INTN_1', 'INTN_2', 'INTN_3', 'INTN_4', 'INTN_5', 'INTN_6']
    corr_matrix = df[turnover_vars].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix - Turnover Intention Variables')
    plt.tight_layout()
    plt.savefig('output_plots/turnover_correlations.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    for var in turnover_vars:
        sns.kdeplot(data=df, x=var, label=var)
    plt.title('Distribution of Turnover Intention Variables')
    plt.legend()
    plt.savefig('output_plots/turnover_distributions.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', bins=30)
    plt.title('Age Distribution')
    plt.savefig('output_plots/age_distribution.png')
    plt.close()

def get_classification_metrics(y_true, y_pred, threshold=3.5):
    """Convert regression predictions to binary classification using threshold"""
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    f1 = f1_score(y_true_binary, y_pred_binary)
    
    return accuracy, f1

def engineer_features(df):
    """Create new features from existing data with NaN handling"""
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    new_features = {}
    
    if all(col in numeric_df.columns for col in ['job_1', 'safety_1']):
        new_features['job_safety_interaction'] = numeric_df['job_1'] * numeric_df['safety_1']
    if all(col in numeric_df.columns for col in ['HR_1', 'commu_2']):
        new_features['hr_comm_interaction'] = numeric_df['HR_1'] * numeric_df['commu_2']

    score_mappings = {
        'satisfaction_score': 'overall',
        'safety_score': 'safety',
        'hr_score': 'HR',
        'tech_score': 'tech',
        'place_score': 'place',
        'commu_score': 'commu'
    }
    
    for score_name, column_prefix in score_mappings.items():
        cols = [col for col in numeric_df.columns if column_prefix in col]
        if cols:
            new_features[score_name] = numeric_df[cols].mean(axis=1)
    
    zscore_features = {}
    for col in numeric_df.columns:
        try:
            zscore_features[f'{col}_zscore'] = stats.zscore(numeric_df[col], nan_policy='omit')
        except:
            print(f"Skipping z-score calculation for {col} due to invalid values")
    
    new_features.update(zscore_features)
    
    engineered_features = pd.DataFrame(new_features)
    
    engineered_features = engineered_features.fillna(0)
    
    result = pd.concat([numeric_df, engineered_features], axis=1)
    
    result = result.fillna(0)
    
    print(f"\nFeature Engineering Summary:")
    print(f"Original numeric features: {len(numeric_df.columns)}")
    print(f"New features created: {len(new_features)}")
    print(f"Total features: {len(result.columns)}")
    print(f"Remaining NaN values: {result.isna().sum().sum()}")
    
    return result

def analyze_feature_categories(feature_importance_df):
    """Analyze importance by feature category"""
    categories = {
        'overall': 'Overall satisfaction',
        'safety': 'Safety related',
        'hr': 'Human resources',
        'tech': 'Technology',
        'place': 'Workplace',
        'commu': 'Communication',
        'sense': 'Sense of belonging',
        'job': 'Job related',
        'affect': 'Emotional',
        'demographic': 'Demographic'
    }
    
    category_importance = {}
    for category, description in categories.items():
        category_cols = [col for col in feature_importance_df['feature'] if category in col.lower()]
        if category == 'demographic':
            category_cols = ['age', 'SEX', 'MAR', 'ETHNIC', 'edu']
        
        importance = feature_importance_df[
            feature_importance_df['feature'].isin(category_cols)
        ]['importance'].sum()
        
        category_importance[description] = importance
    
    category_df = pd.DataFrame({
        'Category': category_importance.keys(),
        'Total Importance': category_importance.values()
    }).sort_values('Total Importance', ascending=False)
    
    return category_df

def prepare_features(df):
    """Prepare features with proper NaN handling and target exclusion"""
    target_vars = ['INTN_1', 'INTN_2', 'INTN_3', 'INTN_4', 'INTN_5', 'INTN_6']
    target_related = []
    for var in target_vars:
        target_related.extend([col for col in df.columns if var in col])

    feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                   if col not in target_related]
    
    X = df[feature_cols].copy()
    
    X = X.fillna(X.mean())
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())
    
    print("Features prepared. Number of features:", len(feature_cols))
    print("NaN values remaining:", X.isna().sum().sum())
    print("\nFeature categories included:")
    categories = {
        'overall': len([col for col in feature_cols if 'overall' in col]),
        'safety': len([col for col in feature_cols if 'safety' in col]),
        'hr': len([col for col in feature_cols if 'HR' in col]),
        'tech': len([col for col in feature_cols if 'tech' in col]),
        'place': len([col for col in feature_cols if 'place' in col]),
        'commu': len([col for col in feature_cols if 'commu' in col]),
        'sense': len([col for col in feature_cols if 'sense' in col]),
        'job': len([col for col in feature_cols if 'job' in col]),
        'affect': len([col for col in feature_cols if 'affect' in col]),
        'demographic': len([col for col in feature_cols if col in ['age', 'SEX', 'MAR', 'ETHNIC', 'edu']])
    }
    for category, count in categories.items():
        print(f"{category}: {count} features")
    
    return X, feature_cols

def train_models(X, df, feature_cols):
    """Train models and store results for all models for each target."""
    target_vars = ['INTN_1', 'INTN_2', 'INTN_3', 'INTN_4', 'INTN_5', 'INTN_6']
    results = {}
    
    all_model_results = []

    if X.isna().sum().sum() > 0:
        print("Warning: Feature matrix contains NaN values. Filling with means...")
        X = X.fillna(X.mean())
    
    for target in target_vars:
        print(f"\nTraining model for {target}")
        y = df[target]
        
        if y.isna().sum() > 0:
            print(f"Warning: Target variable {target} contains NaN values. Filling with mean...")
            y = y.fillna(y.mean())
        
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)
        
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
        except Exception as e:
            print(f"Error in scaling: {str(e)}")
            print("Attempting to fix data...")
            X_train = X_train.fillna(X_train.mean())
            X_val = X_val.fillna(X_val.mean())
            X_test = X_test.fillna(X_test.mean())
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)

        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42, tree_method='gpu_hist', predictor='gpu_predictor'),
            'lgbm': LGBMRegressor(n_estimators=100, random_state=42, device='gpu'),
            'catboost': CatBoostRegressor(n_estimators=100, verbose=0, random_state=42, task_type='GPU', devices='0'),
            #'tpot': TPOTRegressor(generations=5, population_size=20, random_state=42, verbosity=2),
        }

        
        best_model = None
        best_score = float('-inf')
        best_metrics = None
        best_predictions = None
        best_model_name = None

        for name, model in models.items():
            model.fit(X_train_scaled, y_train)

            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            test_pred = model.predict(X_test_scaled)

            metrics = {
                'train_mse': mean_squared_error(y_train, train_pred),
                'val_mse': mean_squared_error(y_val, val_pred),
                'test_mse': mean_squared_error(y_test, test_pred),
                'train_r2': r2_score(y_train, train_pred),
                'val_r2': r2_score(y_val, val_pred),
                'test_r2': r2_score(y_test, test_pred),
            }
            
            train_acc, train_f1 = get_classification_metrics(y_train, train_pred)
            val_acc, val_f1 = get_classification_metrics(y_val, val_pred)
            test_acc, test_f1 = get_classification_metrics(y_test, test_pred)
            
            metrics.update({
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc,
                'train_f1': train_f1,
                'val_f1': val_f1,
                'test_f1': test_f1
            })

            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            metrics['cv_score_mean'] = cv_scores.mean()
            metrics['cv_score_std'] = cv_scores.std()
            
            metrics['cv_scores'] = cv_scores.tolist()

            model_result = {
                'target': target,
                'model_name': name
            }
            model_result.update(metrics)
            all_model_results.append(model_result)
            
            if metrics['test_r2'] > best_score:
                best_score = metrics['test_r2']
                best_model = model
                best_model_name = name
                best_metrics = metrics
                best_predictions = (test_pred, y_test)

        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
        else:
            importance = abs(best_model.coef_)
            
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(12, 8))
        plt.scatter(best_predictions[1], best_predictions[0], alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        
        z = np.polyfit(best_predictions[1], best_predictions[0], 1)
        p = np.poly1d(z)
        plt.plot(best_predictions[1], p(best_predictions[1]), "b--", alpha=0.5)
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted - {target}')
        plt.savefig(f'output_plots/actual_vs_predicted_{target}.png')
        plt.close()

        plt.figure(figsize=(14, 8))
        top_features = feature_importance.head(15)
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top 15 Feature Importance - {target}')
        plt.tight_layout()
        plt.savefig(f'output_plots/feature_importance_{target}.png')
        plt.close()

        print("\nFeature category importance:")
        category_importance = analyze_feature_categories(feature_importance)
        print(category_importance)

        plt.figure(figsize=(12, 6))
        sns.barplot(data=category_importance, x='Total Importance', y='Category')
        plt.title(f'Feature Category Importance - {target}')
        plt.tight_layout()
        plt.savefig(f'output_plots/category_importance_{target}.png')
        plt.close()
        
        results[target] = {
            'model': best_model,
            'metrics': best_metrics,
            'feature_importance': feature_importance
        }
        
        print(f"Best model for {target}: {best_model_name}")
        for metric, value in best_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nTop 5 important features for {target}:")
        print(feature_importance.head())

        print("\nFeature correlation with target:")
        correlations = pd.DataFrame({
            'feature': feature_cols,
            'correlation': [abs(np.corrcoef(X[col], y)[0,1]) for col in feature_cols]
        }).sort_values('correlation', ascending=False)
        print(correlations.head())

    all_model_results_df = pd.DataFrame(all_model_results)
    all_model_results_df.to_excel('output_plots/all_model_results.xlsx', index=False)
    print("\nAll model results saved to 'output_plots/all_model_results.xlsx'")

    return results

def main():
    print("Loading data...")
    df = load_data()

    print("\nVerifying data types:")
    print(df.dtypes.value_counts())
    
    print("\nCleaning data...")
    df_clean, label_encoders = clean_data(df)
    
    print("\nChecking for NaN values after cleaning:")
    nan_counts = df_clean.isna().sum()
    print(nan_counts[nan_counts > 0])
    
    print("\nEngineering features...")
    df_clean = engineer_features(df_clean)

    print("\nVerifying engineered features:")
    print("Shape:", df_clean.shape)
    print("NaN values:", df_clean.isna().sum().sum())
    
    print("\nPerforming data analysis...")
    analyze_data(df_clean)

    print("\nPreparing features...")
    X, feature_cols = prepare_features(df_clean)

    print("\nFinal data verification:")
    print("Feature matrix shape:", X.shape)
    print("NaN values in features:", X.isna().sum().sum())

    print("\nTraining models...")
    results = train_models(X, df_clean, feature_cols)

    print("\nSaving results...")
    with pd.ExcelWriter('output_plots/feature_importance_summary.xlsx') as writer:
        for target, result in results.items():
            result['feature_importance'].to_excel(writer, sheet_name=target)
            
            metrics_df = pd.DataFrame(result['metrics'].items(), columns=['Metric', 'Value'])
            metrics_df.to_excel(writer, sheet_name=f'{target}_metrics')
    
    print("\nAnalysis completed. Check 'output_plots' directory for visualizations and 'all_model_results.xlsx' for all model comparisons.")
    
    return df_clean, results

if __name__ == "__main__":
    df_clean, results = main()
