import os
import pickle
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, make_scorer
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("  imbalanced-learn not installed. Install with: pip install imbalanced-learn")

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("  XGBoost not installed. Install with: pip install xgboost")

warnings.filterwarnings('ignore')


class ETHPricePredictor:
    """Train and save ETH price direction predictor"""
    
    def __init__(self, data_file='Whales_and_Price/whale_prices_ml_ready.csv', model_file='models/best_model.pkl'):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Leakage features
        self.leakage_features = [
            'next_day_return', 'next_day_price_movement', 
            'block_date', 'date', 'timestamp'
        ]
    
    def load_data(self):
        """Load preprocessed data"""
        print(" Loading data...")
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(
                f"Data file not found: {self.data_file}\n"
                "Please run the notebook first to generate the data."
            )
        
        df = pd.read_csv(self.data_file)
        print(f" Loaded {len(df)} rows × {len(df.columns)} columns")
        return df
    
    def prepare_features(self, df, target='next_day_price_direction'):
        """Prepare clean features"""
        print("\n Preparing features...")
        data = df.copy()
        
        # Remove leakage
        leakage_found = [col for col in self.leakage_features if col in data.columns]
        if leakage_found:
            data = data.drop(columns=leakage_found)
            print(f"   Removed {len(leakage_found)} leakage features")
        
        # Select numeric features
        feature_cols = [col for col in data.columns 
                       if col != target and pd.api.types.is_numeric_dtype(data[col])]
        
        # Handle inf/nan
        data[feature_cols] = data[feature_cols].replace([np.inf, -np.inf], np.nan)
        data[feature_cols] = data[feature_cols].fillna(method='ffill').fillna(
            data[feature_cols].median()
        )
        
        print(f" Prepared {len(feature_cols)} features")
        return data, feature_cols
    
    def time_split(self, X, y, train_ratio=0.6, val_ratio=0.2):
        """Time-based split"""
        N = len(X)
        train_size = int(N * train_ratio)
        val_size = int(N * val_ratio)
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        print(f"\n Data split:")
        print(f"   Train: {len(X_train):4d} ({len(X_train)/N*100:.1f}%)")
        print(f"   Val:   {len(X_val):4d} ({len(X_val)/N*100:.1f}%)")
        print(f"   Test:  {len(X_test):4d} ({len(X_test)/N*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def tune_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Hyperparameter tuning"""
        print("\n" + "="*70)
        print(" HYPERPARAMETER TUNING ".center(70))
        print("="*70)
        
        # Custom scoring (balance accuracy and recall)
        def custom_score(y_true, y_pred):
            acc = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, zero_division=0)
            return 0.4 * acc + 0.6 * recall
        
        scorer = make_scorer(custom_score)
        
        # Define parameter grids
        param_grids = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10, 20],
                    'min_samples_leaf': [1, 2, 4, 8],
                    'class_weight': ['balanced', 'balanced_subsample', None],
                    'max_features': ['sqrt', 'log2', None]
                }
            }
        }
        
        if HAS_XGBOOST:
            param_grids['XGBoost'] = {
                'model': XGBClassifier(random_state=42, eval_metric='logloss', 
                                      use_label_encoder=False),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 10],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'scale_pos_weight': [0.8, 1.0, 1.2, 1.5]
                }
            }
        
        # Tune each model
        results = {}
        
        for name, config in param_grids.items():
            print(f"\n Tuning {name}...")
            
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=20,
                scoring=scorer,
                cv=3,
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
            
            search.fit(X_train, y_train)
            
            # Validate
            val_preds = search.best_estimator_.predict(X_val)
            val_acc = accuracy_score(y_val, val_preds)
            val_recall = recall_score(y_val, val_preds, zero_division=0)
            val_auc = roc_auc_score(y_val, search.best_estimator_.predict_proba(X_val)[:, 1])
            
            results[name] = {
                'model': search.best_estimator_,
                'params': search.best_params_,
                'val_accuracy': val_acc,
                'val_recall': val_recall,
                'val_auc': val_auc,
                'custom_score': custom_score(y_val, val_preds)
            }
            
            print(f"   Val Accuracy: {val_acc:.4f}")
            print(f"   Val Recall:   {val_recall:.4f}")
            print(f"   Val AUC:      {val_auc:.4f}")
        
        # Select best
        best_name = max(results.items(), key=lambda x: x[1]['custom_score'])[0]
        print(f"\n Best Model: {best_name}")
        
        return results[best_name]['model'], best_name, results[best_name]['params']
    
    def train(self):
        """Complete training pipeline"""
        print("\n" + "="*70)
        print(" ETH PRICE PREDICTOR TRAINING ".center(70))
        print("="*70)
        
        # Load data
        df = self.load_data()
        
        # Prepare features
        data, feature_cols = self.prepare_features(df)
        self.feature_names = feature_cols
        
        X = data[feature_cols].values
        y = data['next_day_price_direction'].values
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.time_split(X, y)
        
        # Scale
        print("\n Scaling features...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle imbalance
        if HAS_IMBLEARN:
            print("  Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            print(f"   After SMOTE: {np.bincount(y_train_final.astype(int))}")
        else:
            X_train_final, y_train_final = X_train_scaled, y_train
        
        # Tune hyperparameters
        best_model, best_name, best_params = self.tune_hyperparameters(
            X_train_final, y_train_final, X_val_scaled, y_val
        )
        
        # Retrain on train+val
        print("\n" + "="*70)
        print(" FINAL TRAINING ON TRAIN + VAL ".center(70))
        print("="*70)
        
        X_full = np.vstack([X_train, X_val])
        y_full = np.concatenate([y_train, y_val])
        
        self.scaler.fit(X_full)
        X_full_scaled = self.scaler.transform(X_full)
        X_test_scaled = self.scaler.transform(X_test)
        
        if HAS_IMBLEARN:
            X_full_final, y_full_final = SMOTE(random_state=42).fit_resample(
                X_full_scaled, y_full
            )
        else:
            X_full_final, y_full_final = X_full_scaled, y_full
        
        best_model.fit(X_full_final, y_full_final)
        self.model = best_model
        print(f" Retrained {best_model.__class__.__name__}")
        
        # Evaluate on test set
        self.evaluate(X_test_scaled, y_test, best_name, best_params)
    
    def evaluate(self, X_test, y_test, model_name, best_params):
        """Evaluate model"""
        print("\n" + "="*70)
        print(" TEST SET EVALUATION ".center(70))
        print("="*70)
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n Test Performance:")
        print(f"   Accuracy:  {acc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   AUC:       {auc:.4f}")
        
        print("\n Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\n Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Down    Up")
        print(f"Actual Down   {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Up     {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n" + "="*70)
            print(" TOP 15 FEATURE IMPORTANCES ".center(70))
            print("="*70)
            
            importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
            
            for i, (feat, imp) in enumerate(importance.head(15).items(), 1):
                print(f"{i:2d}. {feat:45s} {imp:.6f}")
        
        # Save results
        self.save_model(model_name, best_params, acc, precision, recall, f1, auc)
    
    def save_model(self, model_name, best_params, acc, precision, recall, f1, auc):
        """Save model to disk"""
        os.makedirs('models', exist_ok=True)
        
        model_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target': 'next_day_price_direction',
            'model_name': model_name,
            'best_params': best_params,
            'trained_date': datetime.now().isoformat(),
            'test_accuracy': acc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_auc': auc
        }

        MODEL_FILE = 'models/best_model.pkl'

        with open(self.model_file, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"\n Model saved: {self.model_file}")
        print("\n" + "="*70)
        print(" TRAINING COMPLETE ".center(70))
        print("="*70)


def main():
    """Main training function"""
    trainer = ETHPricePredictor(
        data_file='Whales_and_Price/whale_prices_ml_ready.csv',
        model_file='models/best_model.pkl'
    )
    trainer.train()


if __name__ == "__main__":
    main()