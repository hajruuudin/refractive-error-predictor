"""
Machine Learning Algorithm - Training on SMOTE-Balanced Data

This script trains multiple ML models on the SMOTE-balanced survey data to predict
3 composite refractive error outcomes:
1. Myopia: combination of myopia_level + refractive_worsening
2. CVS: combination of cvs_headache_strain + cvs_dry_eyes
3. Astigmatism: combination of refractive_worsening + astigmatism_symptoms

Models trained:
- Random Forest
- XGBoost

Evaluation metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Feature Importance Analysis
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class MLModelTrainer:
    """Train and evaluate ML models for refractive error prediction"""
    
    def __init__(self, data_path):
        """
        Initialize trainer with SMOTE-balanced data and create composite targets
        
        Args:
            data_path: Path to training_data_smote.csv
        """
        self.data = pd.read_csv(data_path)
        self.input_cols = [
            'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
            'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
            'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
            'genetics', 'age_first_rx'
        ]
        
        # Create composite targets
        self.data['myopia'] = (self.data['myopia_level'] + self.data['refractive_worsening']) / 2
        self.data['cvs'] = (self.data['cvs_headache_strain'] + self.data['cvs_dry_eyes']) / 2
        self.data['astigmatism'] = (self.data['refractive_worsening'] + self.data['astigmatism_symptoms']) / 2
        
        self.target_cols = ['myopia', 'cvs', 'astigmatism']
        self.results = {}
        
    def train_target(self, target, model_type='random_forest', test_size=0.2):
        """
        Train model for a single target variable
        
        Args:
            target: Target column name
            model_type: Type of model to train
            test_size: Fraction of data for testing
        
        Returns:
            Dictionary with training results
        """
        X = self.data[self.input_cols]
        y = self.data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train model
        if model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        
        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Evaluate
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.input_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'cv_mae': cv_mae,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_test': y_pred_test
        }
    
    def train_all_targets(self, model_type='random_forest'):
        """Train models for all 5 target variables"""
        print(f"\n{'='*70}")
        print(f"TRAINING {model_type.upper()} MODELS")
        print(f"{'='*70}")
        
        for target in self.target_cols:
            print(f"\nTraining {target}...")
            results = self.train_target(target, model_type=model_type)
            self.results[target] = results
            
            print(f"  Train MAE: {results['mae_train']:.4f}")
            print(f"  Test MAE:  {results['mae_test']:.4f}")
            print(f"  Train RMSE: {results['rmse_train']:.4f}")
            print(f"  Test RMSE:  {results['rmse_test']:.4f}")
            print(f"  Train R²: {results['r2_train']:.4f}")
            print(f"  Test R²:  {results['r2_test']:.4f}")
            print(f"  CV MAE (5-fold): {results['cv_mae']:.4f}")
            
            print(f"\n  Top 5 Important Features:")
            for idx, row in results['feature_importance'].head(5).iterrows():
                print(f"    {row['feature']:25s}: {row['importance']:.4f}")
    
    def print_summary(self):
        """Print comprehensive summary of all models"""
        print(f"\n{'='*70}")
        print(f"MODEL PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        
        summary_data = []
        for target, results in self.results.items():
            summary_data.append({
                'Target': target,
                'Test MAE': f"{results['mae_test']:.4f}",
                'Test RMSE': f"{results['rmse_test']:.4f}",
                'Test R²': f"{results['r2_test']:.4f}",
                'CV MAE': f"{results['cv_mae']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    def save_results(self, output_dir):
        """Save detailed results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for target, results in self.results.items():
            # Save feature importance
            results['feature_importance'].to_csv(
                os.path.join(output_dir, f'{target}_feature_importance.csv'),
                index=False
            )
            
            # Save predictions vs actual
            pred_df = pd.DataFrame({
                'actual': results['y_test'],
                'predicted': results['y_pred_test'],
                'error': results['y_test'] - results['y_pred_test']
            })
            pred_df.to_csv(
                os.path.join(output_dir, f'{target}_predictions.csv'),
                index=False
            )

def main():
    """Main pipeline"""
    # Paths
    script_dir = os.path.dirname(__file__)
    data_path = os.path.join(script_dir, 'training_data_smote.csv')
    output_dir = os.path.join(script_dir, 'model_results')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return
    
    print("=" * 70)
    print("MACHINE LEARNING PIPELINE - MODEL TRAINING")
    print("=" * 70)
    
    # Initialize trainer
    trainer = MLModelTrainer(data_path)
    print(f"\nLoaded {len(trainer.data)} SMOTE-balanced training samples")
    print(f"Features: {len(trainer.input_cols)} input variables")
    print(f"Targets: {len(trainer.target_cols)} composite outcome variables")
    print(f"  - Myopia: (myopia_level + refractive_worsening) / 2")
    print(f"  - CVS: (cvs_headache_strain + cvs_dry_eyes) / 2")
    print(f"  - Astigmatism: (refractive_worsening + astigmatism_symptoms) / 2")
    
    # Train Random Forest models
    trainer.train_all_targets(model_type='random_forest')
    
    # Train XGBoost if available
    if XGBOOST_AVAILABLE:
        trainer.train_all_targets(model_type='xgboost')
    
    # Print summary
    trainer.print_summary()
    
    # Save results
    print(f"\nSaving results to {output_dir}...")
    trainer.save_results(output_dir)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
