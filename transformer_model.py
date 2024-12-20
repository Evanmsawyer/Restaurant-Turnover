import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import logging
import warnings
import os
import math
import json
import seaborn as sns
from collections import defaultdict
import torch.nn.functional as F
warnings.filterwarnings('ignore')

class EnhancedDataProcessor:
    def __init__(self, output_dir='output_plots_2'):
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

class TurnoverDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class EnhancedOutputAnalyzer:
    def __init__(self, output_dir='output_plots_2'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.metrics_dir = os.path.join(output_dir, 'metrics')
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.model_dir = os.path.join(output_dir, 'models')
        
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
    
    def calculate_metrics(self, y_true, y_pred, threshold=3.5):
        """Calculate comprehensive metrics with dimension checking"""
        metrics = {}
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if len(y_pred.shape) < 2:
            y_pred = y_pred.reshape(-1, y_true.shape[1])
        
        assert y_true.shape == y_pred.shape, f"Shape mismatch: y_true {y_true.shape} != y_pred {y_pred.shape}"
        
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        for i in range(y_true.shape[1]):
            target_metrics = {
                'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'mae': np.mean(np.abs(y_true[:, i] - y_pred[:, i])),
                'r2': r2_score(y_true[:, i], y_pred[:, i]),
                'f1': f1_score(y_true_binary[:, i], y_pred_binary[:, i]),
                'accuracy': accuracy_score(y_true_binary[:, i], y_pred_binary[:, i]),
                'precision': precision_score(y_true_binary[:, i], y_pred_binary[:, i]),
                'recall': recall_score(y_true_binary[:, i], y_pred_binary[:, i])
            }
            metrics[f'target_{i+1}'] = target_metrics
        
        avg_metrics = {
            'avg_mse': np.mean([m['mse'] for m in metrics.values()]),
            'avg_mae': np.mean([m['mae'] for m in metrics.values()]),
            'avg_r2': np.mean([m['r2'] for m in metrics.values()]),
            'avg_f1': np.mean([m['f1'] for m in metrics.values()]),
            'avg_accuracy': np.mean([m['accuracy'] for m in metrics.values()]),
            'avg_precision': np.mean([m['precision'] for m in metrics.values()]),
            'avg_recall': np.mean([m['recall'] for m in metrics.values()])
        }
        
        metrics['average'] = avg_metrics
        return metrics

    
    def plot_prediction_analysis(self, y_true, y_pred, target_vars):
        """Create comprehensive prediction analysis plots"""
        n_targets = len(target_vars)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i in range(n_targets):
            ax = axes[i]
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
            ax.plot([y_true[:, i].min(), y_true[:, i].max()], 
                   [y_true[:, i].min(), y_true[:, i].max()], 
                   'r--', lw=2)
            ax.set_title(f'Actual vs Predicted - {target_vars[i]}')
            ax.set_xlabel('Actual Values')
            ax.set_ylabel('Predicted Values')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'actual_vs_predicted.png'))
        plt.close()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i in range(n_targets):
            ax = axes[i]
            errors = y_pred[:, i] - y_true[:, i]
            sns.histplot(errors, kde=True, ax=ax)
            ax.set_title(f'Error Distribution - {target_vars[i]}')
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'error_distribution.png'))
        plt.close()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for i in range(n_targets):
            ax = axes[i]
            y_true_binary = (y_true[:, i] >= 3.5).astype(int)
            y_pred_proba = 1 / (1 + np.exp(-y_pred[:, i]))  
            
            fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_title(f'ROC Curve - {target_vars[i]}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'roc_curves.png'))
        plt.close()
    
    def save_metrics(self, metrics, filename='metrics.json'):
        """Save metrics to JSON file"""
        filepath = os.path.join(self.metrics_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {filepath}")
    
    def save_predictions(self, y_pred, target_vars, filename='predictions.csv'):
        """Save predictions to CSV file"""
        pred_df = pd.DataFrame(y_pred, columns=target_vars)
        filepath = os.path.join(self.metrics_dir, filename)
        pred_df.to_csv(filepath, index=False)
        self.logger.info(f"Predictions saved to {filepath}")
    
    def generate_report(self, metrics, history):
        """Generate comprehensive analysis report"""
        report = []
        report.append("# Turnover Prediction Model Analysis Report")
        report.append("\n## Model Performance Summary")
        
        report.append("\n### Average Performance Metrics")
        for metric, value in metrics['average'].items():
            report.append(f"- {metric}: {value:.4f}")
        
        report.append("\n### Individual Target Performance")
        for target, target_metrics in metrics.items():
            if target != 'average':
                report.append(f"\n#### {target}")
                for metric, value in target_metrics.items():
                    report.append(f"- {metric}: {value:.4f}")
        
        report.append("\n## Training History")
        report.append(f"- Final training loss: {history['train_loss'][-1]:.4f}")
        report.append(f"- Final validation loss: {history['val_loss'][-1]:.4f}")
        report.append(f"- Final training MAE: {history['train_mae'][-1]:.4f}")
        report.append(f"- Final validation MAE: {history['val_mae'][-1]:.4f}")
        
        report_path = os.path.join(self.metrics_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        self.logger.info(f"Analysis report saved to {report_path}")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ImprovedTurnoverTransformer(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        d_model: int = 512,
        nhead: int = 16,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.3,
        num_targets: int = 6
    ):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layers = []
        for _ in range(num_encoder_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            encoder_layers.append(layer)
            
        self.transformer_encoder = nn.ModuleList(encoder_layers)

        self.auxiliary_outputs = nn.ModuleList([
            nn.Linear(d_model, num_targets) 
            for _ in range(num_encoder_layers-1)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model // 4, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Linear(32, 1)
            ) for _ in range(num_targets)
        ])
        
        self.calibration = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ) for _ in range(num_targets)
        ])

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = src.size(0)
        
        x = self.input_projection(src) 
        
        pos_emb = self.pos_embedding.expand(batch_size, -1, -1)
        x = x + pos_emb
        
        auxiliary_outputs = []
        
        for i, encoder_layer in enumerate(self.transformer_encoder):
            x = encoder_layer(x)
            if i < len(self.transformer_encoder) - 1:
                aux_out = self.auxiliary_outputs[i](torch.mean(x, dim=1))
                auxiliary_outputs.append(aux_out)
        
        attention_weights = torch.softmax(
            torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(x.size(-1)), 
            dim=-1
        )
        x = torch.matmul(attention_weights, x)
        
        x_max, _ = torch.max(x, dim=1)
        x_avg = torch.mean(x, dim=1)
        x = self.output_projection(x_max + x_avg)
        
        outputs = []
        for i, (head, calibration) in enumerate(zip(self.output_heads, self.calibration)):
            raw_pred = head(x)
            
            calib_input = torch.cat([
                raw_pred,
                torch.mean(x, dim=1, keepdim=True)
            ], dim=1)
            
            calibrated_pred = calibration(calib_input) * 6 + 1
            outputs.append(calibrated_pred)
        
        final_output = torch.cat(outputs, dim=1)
        
        return final_output, auxiliary_outputs

class ImprovedTurnoverPredictor:
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 16,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.3,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        num_epochs: int = 500,
        weight_decay: float = 0.001,
        patience: int = 30,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.hyperparams = locals()
        del self.hyperparams['self']
        
        for key, value in self.hyperparams.items():
            setattr(self, key, value)
        
        self.scaler = StandardScaler()
        self.model = None
        self.history = {
            'train_loss': [], 'val_loss': [], 
            'train_mae': [], 'val_mae': [],
            'train_r2': [], 'val_r2': []
        }
        self.auxiliary_weight = 0.2
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_vars: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training by separating features and targets, scaling features,
        and converting to tensors.
        """

        X = df.drop(columns=target_vars)
        y = df[target_vars]
        
        X_scaled = self.scaler.fit_transform(X)

        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y.values)
        
        return X_tensor, y_tensor

    def custom_loss(self, pred, target, auxiliary_outputs=None):
        """
        Custom loss function combining MSE, L1, and Huber losses with auxiliary outputs
        """

        mse_loss = F.mse_loss(pred, target)
        l1_loss = F.l1_loss(pred, target)
        huber_loss = F.smooth_l1_loss(pred, target)
        
        main_loss = 0.4 * mse_loss + 0.3 * l1_loss + 0.3 * huber_loss
        
        if auxiliary_outputs is not None:
            aux_loss = 0
            for aux_out in auxiliary_outputs:
                aux_loss += (
                    0.4 * F.mse_loss(aux_out, target) +
                    0.3 * F.l1_loss(aux_out, target) +
                    0.3 * F.smooth_l1_loss(aux_out, target)
                )
            aux_loss /= len(auxiliary_outputs)
            return main_loss + self.auxiliary_weight * aux_loss
        
        return main_loss

    def _train_epoch(self, train_loader, optimizer, scheduler=None):
        """Train for one epoch with fixed batch dimension handling"""
        metrics = defaultdict(float)
        num_batches = len(train_loader)
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            outputs, auxiliary_outputs = self.model(batch_X)
            
            loss = self.custom_loss(outputs, batch_y, auxiliary_outputs)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            metrics['loss'] += loss.item()
            metrics['mae'] += torch.mean(torch.abs(outputs - batch_y)).item()
            
            y_true = batch_y.detach().cpu().numpy()
            y_pred = outputs.detach().cpu().numpy()
            batch_r2s = []
            
            for i in range(y_true.shape[1]):
                if len(np.unique(y_true[:, i])) > 1:  
                    try:
                        batch_r2 = r2_score(y_true[:, i], y_pred[:, i])
                        batch_r2s.append(batch_r2)
                    except:
                        continue
            
            if batch_r2s: 
                metrics['r2'] += np.mean(batch_r2s)
            
            if np.isnan(loss.item()):
                raise ValueError("Training loss is NaN. Stopping training.")

        for k in metrics:
            metrics[k] /= num_batches
            
        return metrics

    def _validate_epoch(self, val_loader):
        """Validate for one epoch with fixed batch dimension handling"""
        metrics = defaultdict(float)
        num_batches = len(val_loader)
        
        self.model.eval()
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs, _ = self.model(batch_X)
                loss = self.custom_loss(outputs, batch_y)
                
                metrics['loss'] += loss.item()
                metrics['mae'] += torch.mean(torch.abs(outputs - batch_y)).item()
                
                y_true = batch_y.cpu().numpy()
                y_pred = outputs.cpu().numpy()
                batch_r2s = []
                
                for i in range(y_true.shape[1]):
                    if len(np.unique(y_true[:, i])) > 1:
                        try:
                            batch_r2 = r2_score(y_true[:, i], y_pred[:, i])
                            batch_r2s.append(batch_r2)
                        except:
                            continue
                
                if batch_r2s:
                    metrics['r2'] += np.mean(batch_r2s)
        
        for k in metrics:
            metrics[k] /= num_batches
            
        return metrics

    def train(
        self, 
        df: pd.DataFrame, 
        target_vars: List[str],
        val_size: float = 0.2
    ) -> Dict:
        """
        Train the model with the given data
        """
        self.logger.info("Preparing data...")
        X_tensor, y_tensor = self.prepare_data(df, target_vars)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, 
            test_size=val_size, 
            random_state=42
        )
        
        train_dataset = TurnoverDataset(X_train, y_train)
        val_dataset = TurnoverDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size,
            num_workers=0,
            pin_memory=True
        )
        
        if self.model is None:
            self.model = ImprovedTurnoverTransformer(
                input_dim=X_train.shape[1],
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                num_targets=len(target_vars)
            ).to(self.device)
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            amsgrad=True
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            epochs=self.num_epochs,
            steps_per_epoch=len(train_loader)
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        self.logger.info("Starting training...")
        for epoch in range(self.num_epochs):
            self.model.train()
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler)
            
            self.model.eval()
            val_metrics = self._validate_epoch(val_loader)
            
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)
            
            if val_metrics['loss'] < best_val_loss * 0.9999:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                best_epoch = epoch
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
            
            if epoch >= 100 and patience_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1}")
                break
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{self.num_epochs} - "
                    f"Train Loss: {train_metrics['loss']:.4f} - "
                    f"Val Loss: {val_metrics['loss']:.4f} - "
                    f"Train MAE: {train_metrics['mae']:.4f} - "
                    f"Val MAE: {val_metrics['mae']:.4f} - "
                    f"Train R2: {train_metrics['r2']:.4f} - "
                    f"Val R2: {val_metrics['r2']:.4f}"
                )
        
        self.model.load_state_dict(torch.load('best_model.pt'))
        self.logger.info(f"Training completed. Best model was from epoch {best_epoch+1}")
        return self.history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data with correct batch processing"""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        print(f"Input tensor shape: {X_tensor.shape}")
        
        all_predictions = []
        with torch.no_grad():
            predict_dataset = TurnoverDataset(X_tensor, torch.zeros((len(X_tensor), 6)))
            predict_loader = DataLoader(
                predict_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0
            )
            
            total_batches = len(predict_loader)
            print(f"Total number of batches: {total_batches}")
            
            for batch_idx, (batch_X, _) in enumerate(predict_loader, 1):
                batch_X = batch_X.to(self.device)
                
                outputs, _ = self.model(batch_X)
                batch_predictions = outputs.cpu().numpy()
                all_predictions.append(batch_predictions)
                
                print(f"Batch {batch_idx}/{total_batches}: Input shape {batch_X.shape}, Output shape {batch_predictions.shape}")

        final_predictions = np.concatenate(all_predictions, axis=0)
        print(f"Final predictions shape: {final_predictions.shape}")

        assert len(final_predictions) == len(X), \
            f"Number of predictions ({len(final_predictions)}) doesn't match input size ({len(X)})"
        
        return final_predictions

    def save_model(self, path: str):
        """Save model to disk"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler_state': self.scaler,
                'hyperparams': self.hyperparams,
                'history': self.history
            }, path)
            self.logger.info(f"Model saved to {path}")
        else:
            self.logger.warning("No model to save")

    def load_model(self, path: str):
        """Load model from disk"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.hyperparams = checkpoint['hyperparams']
            for key, value in self.hyperparams.items():
                setattr(self, key, value)
            
            self.model = ImprovedTurnoverTransformer(
                input_dim=checkpoint['model_state_dict']['input_projection.0.weight'].shape[1],
                d_model=self.d_model,
                nhead=self.nhead,
                num_encoder_layers=self.num_encoder_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scaler = checkpoint['scaler_state']
            self.history = checkpoint['history']
            self.logger.info(f"Model loaded from {path}")
        else:
            raise FileNotFoundError(f"No model file found at {path}")

def main():
    print("Loading and processing data...")
    processor = EnhancedDataProcessor()
    output_analyzer = EnhancedOutputAnalyzer()
    
    df = processor.load_data("employee_survey_data.xlsx")
    target_vars = ['INTN_1', 'INTN_2', 'INTN_3', 'INTN_4', 'INTN_5', 'INTN_6']
    
    df_clean = processor.clean_data(df, target_vars=target_vars)
    df_engineered = processor.engineer_features(df_clean, target_vars=target_vars)
    
    print("\nSplitting data...")
    train_df, test_df = train_test_split(df_engineered, test_size=0.1, random_state=42)
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    predictor = ImprovedTurnoverPredictor(
        d_model=512,
        nhead=16,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.3,
        batch_size=16, 
        learning_rate=0.001,
        num_epochs=500,
        weight_decay=0.001,
        patience=30
    )
    
    print("\nTraining model...")
    history = predictor.train(train_df, target_vars)

    print("\nMaking predictions...")
    X_test = test_df.drop(columns=target_vars)
    y_test = test_df[target_vars].values
    predictions = predictor.predict(X_test)
    
    print("\nVerifying shapes:")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test targets shape: {y_test.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    assert predictions.shape == y_test.shape, \
        f"Shape mismatch: predictions {predictions.shape} != y_test {y_test.shape}"
    
    print("\nGenerating analysis...")
    metrics = output_analyzer.calculate_metrics(y_test, predictions)
    output_analyzer.plot_prediction_analysis(y_test, predictions, target_vars)
    output_analyzer.save_metrics(metrics)
    output_analyzer.save_predictions(predictions, target_vars)
    output_analyzer.generate_report(metrics, history)
    
    return predictor, predictions, metrics

if __name__ == "__main__":
    predictor, predictions, metrics = main()