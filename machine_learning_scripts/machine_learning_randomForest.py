"""
PART 02A: RANDOM FOREST MODEL TRAINING

This script trains Random Forest regression models for each of the three composite targets (Myopia Score, 
CVS Score, Astigmatism Score) using the SMOTE-balanced datasets created in the previous script. 
The script performs the following steps for each target:
1. Loads the corresponding SMOTE-balanced CSV file
2. Splits the data into training and testing sets
3. Trains a Random Forest Regressor on the training data
4. Evaluates the model using three metrics: MAE, R², and 5-fold cross-validated MAE
5. Extracts and saves feature importance for each model
6. Saves predictions vs actuals for the test set
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


class RandomForestTrainer:
    def __init__(self, myopia_path, cvs_path, astigmatism_path):
        self.datasets = {
            'myopia_score':       pd.read_csv(myopia_path),
            'cvs_score':          pd.read_csv(cvs_path),
            'astigmatism_score':  pd.read_csv(astigmatism_path)
        }

        self.input_cols = [
            'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
            'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
            'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
            'genetics', 'age_first_rx'
        ]

        self.results = {}

    def train_target(self, target, test_size=0.2):
        df = self.datasets[target]

        X = df[self.input_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred_test = model.predict(X_test)

        # --- Three evaluation metrics ---

        # 1. MAE — average prediction error in target units
        mae = mean_absolute_error(y_test, y_pred_test)

        # 2. R² — proportion of variance explained by the model
        r2 = r2_score(y_test, y_pred_test)

        # 3. CV MAE — 5-fold cross-validated MAE to check generalisation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5, scoring='neg_mean_absolute_error'
        )
        cv_mae = -cv_scores.mean()

        feature_importance = pd.DataFrame({
            'feature':    self.input_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        return {
            'model':              model,
            'mae':                mae,
            'r2':                 r2,
            'cv_mae':             cv_mae,
            'feature_importance': feature_importance,
            'X_test':             X_test,
            'y_test':             y_test,
            'y_pred_test':        y_pred_test
        }

    def train_all(self):
        """Train Random Forest models for all three composite targets"""
        print(f"\n{'='*70}")
        print(f"TRAINING RANDOM FOREST MODELS")
        print(f"{'='*70}")

        for target in self.datasets.keys():
            print(f"\nTraining: {target}...")
            print(f"  Dataset size: {len(self.datasets[target])} rows")

            results = self.train_target(target)
            self.results[target] = results

            print(f"  MAE:          {results['mae']:.4f}")
            print(f"  R²:           {results['r2']:.4f}")
            print(f"  CV MAE:       {results['cv_mae']:.4f}")

            print(f"\n  Top 5 Important Features:")
            for _, row in results['feature_importance'].head(5).iterrows():
                print(f"    {row['feature']:25s}: {row['importance']:.4f}")

    def print_summary(self):
        """Print a clean summary table of all three models"""
        print(f"\n{'='*70}")
        print(f"MODEL PERFORMANCE SUMMARY")
        print(f"{'='*70}")

        summary_data = []
        for target, results in self.results.items():
            summary_data.append({
                'Target':  target,
                'MAE':     f"{results['mae']:.4f}",
                'R²':      f"{results['r2']:.4f}",
                'CV MAE':  f"{results['cv_mae']:.4f}"
            })

        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))

    def save_results(self, output_dir):
        """Save feature importance and predictions to CSV files"""
        os.makedirs(output_dir, exist_ok=True)

        for target, results in self.results.items():
            # Feature importance
            results['feature_importance'].to_csv(
                os.path.join(output_dir, f'{target}_feature_importance.csv'),
                index=False
            )

            # Predictions vs actuals
            pred_df = pd.DataFrame({
                'actual':    results['y_test'],
                'predicted': results['y_pred_test'],
                'error':     results['y_test'] - results['y_pred_test']
            })
            pred_df.to_csv(
                os.path.join(output_dir, f'{target}_predictions.csv'),
                index=False
            )

        print(f"  Results saved to {output_dir}")


def main():
    script_dir = os.path.dirname(__file__)

    myopia_path      = os.path.join(script_dir, 'training_data_smote_myopia.csv')
    cvs_path         = os.path.join(script_dir, 'training_data_smote_computervs.csv')
    astigmatism_path = os.path.join(script_dir, 'training_data_smote_astigmatism.csv')

    for path in [myopia_path, cvs_path, astigmatism_path]:
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run the SMOTE script first.")
            return

    output_dir = os.path.join(script_dir, 'model_results_rf')

    print("=" * 70)
    print("RANDOM FOREST PIPELINE - MODEL TRAINING")
    print("=" * 70)

    trainer = RandomForestTrainer(myopia_path, cvs_path, astigmatism_path)
    trainer.train_all()
    trainer.print_summary()

    print(f"\nSaving results...")
    trainer.save_results(output_dir)

    print("\n" + "=" * 70)
    print("RANDOM FOREST TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()