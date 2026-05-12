"""
Linear Model - Equation Analysis and Validation

This script validates the linear model approach:
1. Load 500 fake survey responses
2. Calculate 4 aggregated variables (strain_index, digital_habits, lifestyle, biologic)
3. Create 3 composite targets (myopia, cvs, astigmatism)
4. Use weighted equations to predict risk scores from the 4 aggregated variables
5. Compare predictions vs actual targets to validate weight accuracy

Key concept: Some variables are PROACTIVE (reduce risk):
- outdoor_activity: reduces myopia/cvs risk
- These are inverted so higher values = lower risk contribution
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LinearEquationAnalyzer:
    """Analyze linear model equations combining 4 aggregated variables"""
    
    def __init__(self, csv_path):
        """
        Initialize with fake survey responses
        
        Args:
            csv_path: Path to fake_survey_responses.csv
        """
        self.df = pd.read_csv(csv_path)
        self.input_cols = [
            'age', 'gender', 'daily_screen_time', 'continuous_usage', 'intensity',
            'lighting', 'multi_device', 'phone_distance', 'monitor_distance',
            'blue_light_filter', 'before_bed_usage', 'profession', 'outdoor_activity',
            'genetics', 'age_first_rx'
        ]
        self.target_cols = [
            'myopia_level', 'refractive_worsening', 'cvs_headache_strain',
            'cvs_dry_eyes', 'astigmatism_symptoms'
        ]
    
    def normalize_column(self, col):
        """Normalize a column to 0-1 range"""
        min_val = col.min()
        max_val = col.max()
        if max_val == min_val:
            return pd.Series([0.5] * len(col))
        return (col - min_val) / (max_val - min_val)
    
    def calculate_aggregated_variables(self):
        """
        Calculate 4 aggregated variables from 15 inputs using literature-based weights
        
        Returns:
            DataFrame with strain_index, digital_habits, lifestyle, biologic
        """
        # Normalize all inputs
        normalized_df = pd.DataFrame()
        for col in self.input_cols:
            normalized_df[col] = self.normalize_column(self.df[col])
        
        # Define groupings
        strain_index_cols = ['daily_screen_time', 'continuous_usage']
        digital_habits_cols = ['intensity', 'lighting', 'multi_device', 'blue_light_filter', 
                              'before_bed_usage', 'phone_distance', 'monitor_distance']
        lifestyle_cols = ['profession', 'outdoor_activity']
        biologic_cols = ['age', 'age_first_rx', 'gender', 'genetics']
        
        # Define weights for each outcome
        weights = {
            'myopia': {
                'daily_screen_time': 0.70, 'continuous_usage': 0.65,
                'intensity': 0.60, 'lighting': 0.55, 'multi_device': 0.50, 
                'blue_light_filter': 0.35, 'before_bed_usage': 0.60,
                'phone_distance': 0.75, 'monitor_distance': 0.75,
                'profession': 0.55, 'outdoor_activity': 0.80,  # outdoor = proactive (reduces risk)
                'genetics': 0.90, 'age_first_rx': 0.75, 'age': 0.50, 'gender': 0.40
            },
            'cvs': {
                'daily_screen_time': 0.90, 'continuous_usage': 0.85,
                'intensity': 0.80, 'lighting': 0.75, 'multi_device': 0.80, 
                'blue_light_filter': 0.50, 'before_bed_usage': 0.55,
                'phone_distance': 0.70, 'monitor_distance': 0.70,
                'profession': 0.75, 'outdoor_activity': 0.45,  # outdoor = proactive
                'genetics': 0.25, 'age_first_rx': 0.60, 'age': 0.60, 'gender': 0.50
            },
            'astigmatism': {
                'daily_screen_time': 0.30, 'continuous_usage': 0.25,
                'intensity': 0.20, 'lighting': 0.20, 'multi_device': 0.15, 
                'blue_light_filter': 0.10, 'before_bed_usage': 0.15,
                'phone_distance': 0.25, 'monitor_distance': 0.25,
                'profession': 0.25, 'outdoor_activity': 0.20,
                'genetics': 0.80, 'age_first_rx': 0.50, 'age': 0.30, 'gender': 0.20
            }
        }
        
        # Use myopia weights as primary for aggregation
        primary_weights = weights['myopia']
        
        # Calculate weighted aggregated variables
        strain_index = sum(normalized_df[col] * primary_weights.get(col, 0) 
                          for col in strain_index_cols) / len(strain_index_cols)
        
        digital_habits = sum(normalized_df[col] * primary_weights.get(col, 0) 
                            for col in digital_habits_cols) / len(digital_habits_cols)
        
        # Lifestyle: outdoor_activity is PROACTIVE (reduces risk), so invert it
        lifestyle_values = []
        for col in lifestyle_cols:
            if col == 'outdoor_activity':
                # Invert: high outdoor_activity = low risk contribution
                lifestyle_values.append((1 - normalized_df[col]) * primary_weights.get(col, 0))
            else:
                lifestyle_values.append(normalized_df[col] * primary_weights.get(col, 0))
        lifestyle = sum(lifestyle_values) / len(lifestyle_cols)
        
        biologic = sum(normalized_df[col] * primary_weights.get(col, 0) 
                      for col in biologic_cols) / len(biologic_cols)
        
        return pd.DataFrame({
            'strain_index': strain_index,
            'digital_habits': digital_habits,
            'lifestyle': lifestyle,
            'biologic': biologic
        })
    
    def calculate_composite_targets(self):
        """
        Create 3 composite targets from 5 original targets
        
        Returns:
            DataFrame with myopia, cvs, astigmatism
        """
        return pd.DataFrame({
            'myopia': (self.df['myopia_level'] + self.df['refractive_worsening']) / 2,
            'cvs': (self.df['cvs_headache_strain'] + self.df['cvs_dry_eyes']) / 2,
            'astigmatism': (self.df['refractive_worsening'] + self.df['astigmatism_symptoms']) / 2
        })
    
    def predict_risk_scores(self, aggregated, outcome='myopia'):
        """
        Use weighted equation to predict risk score from 4 aggregated variables
        
        Equation: risk_score = w1*strain_index + w2*digital_habits + w3*lifestyle + w4*biologic
        
        Args:
            aggregated: DataFrame with 4 aggregated variables
            outcome: 'myopia', 'cvs', or 'astigmatism'
        
        Returns:
            Array of predicted risk scores (0-1 range approximately)
        """
        # Define weights for combining the 4 aggregated variables per outcome
        equation_weights = {
            'myopia': {
                'strain_index': 0.35,      # Screen time effect on myopia
                'digital_habits': 0.25,    # Digital environment effect
                'lifestyle': 0.20,         # Outdoor activity reduces risk
                'biologic': 0.20           # Genetic predisposition
            },
            'cvs': {
                'strain_index': 0.40,      # Screen time heavily affects CVS
                'digital_habits': 0.30,    # Environment matters for eyes
                'lifestyle': 0.15,         # Outdoor time helps
                'biologic': 0.15           # Some genetic component
            },
            'astigmatism': {
                'strain_index': 0.20,      # Moderate screen effect
                'digital_habits': 0.15,    # Less than myopia/cvs
                'lifestyle': 0.15,         # Genetics dominant
                'biologic': 0.50           # Strongly genetic
            }
        }
        
        weights = equation_weights[outcome]
        
        # Calculate weighted sum
        risk_score = (
            aggregated['strain_index'] * weights['strain_index'] +
            aggregated['digital_habits'] * weights['digital_habits'] +
            aggregated['lifestyle'] * weights['lifestyle'] +
            aggregated['biologic'] * weights['biologic']
        )
        
        return risk_score
    
    def bin_risk_score(self, score, max_score=None):
        """
        Bin continuous risk score into 5 categories (0-4)
        
        Args:
            score: Risk score (continuous)
            max_score: Maximum possible score (for normalization)
        
        Returns:
            Risk category (0-4)
        """
        if max_score is None:
            max_score = score.max()
        
        # Normalize to 0-1
        normalized = score / max_score if max_score > 0 else score
        
        # Bin into 5 categories
        return pd.cut(normalized, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                     labels=[0, 1, 2, 3, 4], include_lowest=True)
    
    def evaluate_predictions(self, predicted, actual, outcome):
        """
        Evaluate how well predictions align with actual targets
        
        Args:
            predicted: Predicted risk scores
            actual: Actual target values
            outcome: 'myopia', 'cvs', or 'astigmatism'
        """
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    def run_analysis(self):
        """Execute full linear model analysis"""
        print("=" * 70)
        print("LINEAR MODEL - EQUATION ANALYSIS AND VALIDATION")
        print("=" * 70)
        
        print(f"\nLoaded {len(self.df)} survey responses")
        
        # Step 1: Calculate aggregated variables
        print(f"\nStep 1: Calculating 4 aggregated variables...")
        aggregated = self.calculate_aggregated_variables()
        print(f"  Aggregated variables created")
        print(f"  - strain_index: [0, 1]")
        print(f"  - digital_habits: [0, 1]")
        print(f"  - lifestyle: [0, 1] (outdoor_activity inverted as proactive)")
        print(f"  - biologic: [0, 1]")
        
        # Step 2: Calculate composite targets
        print(f"\nStep 2: Calculating 3 composite targets...")
        targets = self.calculate_composite_targets()
        print(f"  Composite targets created")
        print(f"  - myopia: (myopia_level + refractive_worsening) / 2")
        print(f"  - cvs: (cvs_headache_strain + cvs_dry_eyes) / 2")
        print(f"  - astigmatism: (refractive_worsening + astigmatism_symptoms) / 2")
        
        # Step 3: Make predictions and compare
        print(f"\nStep 3: Predicting risk scores using linear equations...")
        print(f"{'='*70}\n")
        
        results = {}
        for outcome in ['myopia', 'cvs', 'astigmatism']:
            print(f"Predicting {outcome}...")
            
            # Predict risk score
            predicted_score = self.predict_risk_scores(aggregated, outcome)
            
            # Get actual target
            actual = targets[outcome]
            
            # Evaluate
            metrics = self.evaluate_predictions(predicted_score, actual, outcome)
            results[outcome] = metrics
            
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")
            print()
        
        # Print summary
        print(f"{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        summary_data = []
        for outcome, metrics in results.items():
            summary_data.append({
                'Outcome': outcome,
                'MAE': f"{metrics['mae']:.4f}",
                'RMSE': f"{metrics['rmse']:.4f}",
                'R²': f"{metrics['r2']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print("VALIDATION COMPLETE")
        print("Lower MAE/RMSE and higher R² indicate better equation weights")
        print(f"{'='*70}")

def main():
    """Main pipeline"""
    project_root = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(project_root, 'fake_survey_responses.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return
    
    analyzer = LinearEquationAnalyzer(csv_path)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
