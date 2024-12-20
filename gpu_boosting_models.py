import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tpot import TPOTRegressor
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.feature_selection import SelectFromModel, RFE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import optuna
import warnings
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
import numpy as np
warnings.filterwarnings('ignore')

class EnhancedDataProcessor:
    def __init__(self, output_dir='output_plots_3'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.pca = None
        self.column_medians = {}
        
    def split_features_target(self, df, target_vars):
        """Split features and target before any processing"""
        X = df.drop(columns=target_vars)
        y = df[target_vars]
        return X, y
    
    def _preprocess_categorical(self, df, column, fit=True):
        """Handle categorical columns with appropriate encoding"""
        if df[column].dtype == 'object':
            if column in ['YEAR', 'earned_tip']:
                numeric_values = pd.to_numeric(df[column], errors='coerce')
                if fit:
                    self.column_medians[column] = numeric_values.median()
                return numeric_values.fillna(self.column_medians.get(column, numeric_values.median()))
            
            if fit:
                self.label_encoders[column] = LabelEncoder()
                return self.label_encoders[column].fit_transform(df[column].astype(str))
            return self.label_encoders[column].transform(df[column].astype(str))
        return df[column]
    
    def clean_data(self, df, target_vars=None, fit=True):
        """Clean data with proper train/test separation"""
        df_clean = df.copy()
        
        if target_vars:
            target_data = df_clean[target_vars].copy()
            df_clean = df_clean.drop(columns=target_vars)
        
        current_year = 2024
        df_clean['age'] = current_year - pd.to_numeric(df_clean['YEAR'], errors='coerce')
        
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df_clean[col] = self._preprocess_categorical(df_clean, col, fit)
        
        remaining_object_cols = df_clean.select_dtypes(include=['object']).columns
        if len(remaining_object_cols) > 0:
            df_clean = df_clean.drop(columns=remaining_object_cols)
        
        if fit:
            self.imputer = SimpleImputer(strategy='median')
            df_clean = pd.DataFrame(
                self.imputer.fit_transform(df_clean),
                columns=df_clean.columns,
                index=df_clean.index
            )
        else:
            df_clean = pd.DataFrame(
                self.imputer.transform(df_clean),
                columns=df_clean.columns,
                index=df_clean.index
            )
        
        if target_vars:
            df_clean[target_vars] = target_data
        
        return df_clean
    
    def load_data(self, file_path):
        """
        Load and perform initial verification of the data.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file (supports .xlsx, .csv, .parquet)
        
        Returns:
        --------
        pd.DataFrame
            Loaded and initially verified dataframe
        """
        file_extension = file_path.split('.')[-1].lower()
        
        try:
            if file_extension == 'xlsx':
                df = pd.read_excel(file_path)
            elif file_extension == 'csv':
                df = pd.read_csv(file_path)
            elif file_extension == 'parquet':
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            print(f"Data loaded successfully. Shape: {df.shape}")
            
            self._verify_data(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _verify_data(self, df):
        """Verify data quality and print summary statistics"""
        print("\nData Verification:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        missing_values = df.isnull().sum()
        if missing_values.any():
            print("\nMissing values per column:")
            print(missing_values[missing_values > 0].sort_values(ascending=False))
        
        print("\nData types:")
        print(df.dtypes.value_counts())
        
        print("\nCategorical columns:")
        print(df.select_dtypes(include=['object']).columns.tolist())
        print("\nNumeric columns:")
        print(df.select_dtypes(include=[np.number]).columns.tolist())

        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"\nWarning: Found {duplicates} duplicate rows")
        
        print("\nNumeric columns summary:")
        print(df.describe().round(2))
        
        return df

    def engineer_features(self, df, target_vars=None, fit=True):
        """Engineer features with proper target separation"""
        df_engineered = df.copy()
        
        if target_vars:
            target_data = df_engineered[target_vars].copy()
            df_engineered = df_engineered.drop(columns=target_vars)
        
        feature_groups = {
            'satisfaction': 'overall',
            'safety': 'safety',
            'hr': 'HR',
            'tech': 'tech',
            'place': 'place',
            'communication': 'commu',
            'job': 'job',
            'affect': 'affect'
        }
        
        for group_name, prefix in feature_groups.items():
            cols = [col for col in df_engineered.columns if prefix in col 
                   and df_engineered[col].dtype in ['int64', 'float64']]
            if cols:
                df_engineered[f'{group_name}_mean'] = df_engineered[cols].mean(axis=1)
                df_engineered[f'{group_name}_std'] = df_engineered[cols].std(axis=1)

        important_features = ['job_1', 'safety_1', 'HR_1', 'commu_2']
        available_features = [f for f in important_features if f in df_engineered.columns]
        
        for i in range(len(available_features)):
            for j in range(i+1, len(available_features)):
                new_col = f'interact_{available_features[i]}_{available_features[j]}'
                df_engineered[new_col] = df_engineered[available_features[i]] * df_engineered[available_features[j]]
        
        numeric_cols = df_engineered.select_dtypes(include=[np.number]).columns
        if fit:
            df_engineered[numeric_cols] = self.scaler.fit_transform(df_engineered[numeric_cols])
        else:
            df_engineered[numeric_cols] = self.scaler.transform(df_engineered[numeric_cols])
        
        if target_vars:
            df_engineered[target_vars] = target_data
        
        return df_engineered


class GPUOptimizedModels:
    def __init__(self):
        self.models = {
            'xgb': XGBRegressor(
                tree_method='gpu_hist',
                predictor='gpu_predictor',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8
            ),
            'lgbm': LGBMRegressor(
                device='gpu',
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                feature_fraction=0.8,
                force_row_wise=True,
                deterministic=True,
                min_child_samples=20,  
                min_data_in_leaf=20,   
                verbosity=-1,          
                silent=True,         
                min_gain_to_split=0,  
                gpu_platform_id=0,
                gpu_device_id=0
            ),
            'catboost': CatBoostRegressor(
                task_type='GPU',
                devices='0',
                n_estimators=100,
                learning_rate=0.1,
                depth=6,
                verbose=0
            )
        }
    
    def normalize_feature_importance(self, importance_dict):
        """Normalize feature importance scores to sum to 1"""
        total = sum(importance_dict.values())
        if total > 0:
            return {k: v/total for k, v in importance_dict.items()}
        return importance_dict
    
    def get_feature_importance(self, model, model_name, feature_names):
        """Extract and normalize feature importance"""
        importance_dict = {}
        
        try:
            if model_name == 'xgb':
                importance = model.feature_importances_
            elif model_name == 'lgbm':
                importance = model.feature_importances_
            elif model_name == 'catboost':
                importance = model.feature_importances_
            else:
                return None
            
            for feat, imp in zip(feature_names, importance):
                importance_dict[feat] = imp
            
            importance_dict = self.normalize_feature_importance(importance_dict)
            
            importance_dict = dict(sorted(importance_dict.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True))
            
            return importance_dict
            
        except Exception as e:
            print(f"Warning: Could not extract feature importance for {model_name}: {str(e)}")
            return None

def train_and_evaluate(X, y, gpu_models):
    """Training function with comprehensive metrics for both regression and classification"""
    cv_metrics = {model_name: {
        'r2': [], 'mse': [], 'f1': [], 'accuracy': [], 
        'precision': [], 'recall': []
    } for model_name in gpu_models.models.keys()}
    feature_importance = {}
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        for name, model in gpu_models.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            y_test_binary = (y_test > y_test.median()).astype(int)
            y_pred_binary = (y_pred > np.median(y_pred)).astype(int)
            
            f1 = f1_score(y_test_binary, y_pred_binary)
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            precision = precision_score(y_test_binary, y_pred_binary)
            recall = recall_score(y_test_binary, y_pred_binary)

            cv_metrics[name]['r2'].append(r2)
            cv_metrics[name]['mse'].append(mse)
            cv_metrics[name]['f1'].append(f1)
            cv_metrics[name]['accuracy'].append(accuracy)
            cv_metrics[name]['precision'].append(precision)
            cv_metrics[name]['recall'].append(recall)
            
            importance = gpu_models.get_feature_importance(model, name, X.columns)
            if importance:
                if name not in feature_importance:
                    feature_importance[name] = importance
                else:
                    for feat, imp in importance.items():
                        if feat in feature_importance[name]:
                            feature_importance[name][feat] = (feature_importance[name][feat] + imp) / 2
                        else:
                            feature_importance[name][feat] = imp
    
    final_metrics = {}
    for name in gpu_models.models.keys():
        metrics_list = ['r2', 'mse', 'f1', 'accuracy', 'precision', 'recall']
        for metric in metrics_list:
            final_metrics[f'{name}_{metric}_mean'] = np.mean(cv_metrics[name][metric])
            final_metrics[f'{name}_{metric}_std'] = np.std(cv_metrics[name][metric])
    
    return final_metrics, feature_importance

def main():
    try:
        processor = EnhancedDataProcessor()
        gpu_models = GPUOptimizedModels()
        
        print("Loading and processing data...")
        df = processor.load_data("employee_survey_data.xlsx")
        
        target_vars = ['INTN_1', 'INTN_2', 'INTN_3', 'INTN_4', 'INTN_5', 'INTN_6']
        
        print("\nCleaning data...")
        df_clean = processor.clean_data(df, target_vars=target_vars)
        
        print("\nEngineering features...")
        df_engineered = processor.engineer_features(df_clean, target_vars=target_vars)
        
        results = {}
        
        for target in target_vars:
            print(f"\nProcessing target: {target}")
            
            X = df_engineered.drop(columns=target_vars)
            y = df_engineered[target]
            
            metrics, feature_importance = train_and_evaluate(X, y, gpu_models)
            
            print(f"\nResults for {target}:")
            for metric_name, metric_value in metrics.items():
                if 'std' in metric_name:
                    print(f"{metric_name}: {metric_value:.4f}")
                else:
                    print(f"{metric_name}: {metric_value:.4f}")
            
            print("\nTop Categorical Features:")
            for model_name, importance in feature_importance.items():
                print(f"\n{model_name.upper()} Model:")
                cat_features = {k: v for k, v in importance.items() 
                              if any(cat in k.lower() for cat in 
                                   ['ethnic', 'sex', 'mar', 'edu', 'income', 'position', 'industry'])}
                for feat, imp in list(cat_features.items())[:10]:
                    print(f"{feat}: {imp:.4f}")
            
            results[target] = {
                'metrics': metrics,
                'feature_importance': feature_importance
            }
        
        return results
    
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()