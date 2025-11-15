"""
predict.py - ETH Whale Activity Price Predictor Prediction Script

Loads the trained model and makes predictions on new data.

Usage:
    # Predict on new data
    python predict.py --data whale_prices_ml_ready.csv
    
    # Predict on latest day
    python predict.py --latest
    
    # Interactive mode
    python predict.py --interactive
"""

import argparse
import pickle
import warnings
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')


class ETHPricePredictor:
    """Load model and make predictions"""
    
    def __init__(self, model_file='models/best_model.pkl'):
        self.model_file = model_file
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_info = None
        
        self.load_model()
    
    def load_model(self):
        """Load trained model from disk"""
        print("📦 Loading model...")
        
        try:
            with open(self.model_file, 'rb') as f:
                model_dict = pickle.load(f)
            
            self.model = model_dict['model']
            self.scaler = model_dict['scaler']
            self.feature_names = model_dict['feature_names']
            self.model_info = {
                'model_name': model_dict.get('model_name', 'Unknown'),
                'trained_date': model_dict.get('trained_date', 'Unknown'),
                'test_accuracy': model_dict.get('test_accuracy', None),
                'test_recall': model_dict.get('test_recall', None),
                'test_auc': model_dict.get('test_auc', None)
            }
            
            print(f" Model loaded: {self.model_info['model_name']}")
            print(f"   Trained: {self.model_info['trained_date']}")
            print(f"   Test Accuracy: {self.model_info['test_accuracy']:.4f}")
            print(f"   Test Recall: {self.model_info['test_recall']:.4f}")
            print(f"   Test AUC: {self.model_info['test_auc']:.4f}")
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found: {self.model_file}\n"
                "Please run train.py first to train the model."
            )
    
    def prepare_features(self, df):
        """Prepare features for prediction"""
        # Ensure we have all required features
        missing_features = set(self.feature_names) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}\n"
                "Please ensure data has all necessary features."
            )
        
        # Select and order features
        X = df[self.feature_names].values
        
        # Handle inf/nan
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, df, return_proba=True):
        """Make predictions on new data"""
        print(f"\n🔮 Making predictions on {len(df)} samples...")
        
        X_scaled = self.prepare_features(df)
        
        # Predictions
        predictions = self.model.predict(X_scaled)
        
        if return_proba:
            probabilities = self.model.predict_proba(X_scaled)
            return predictions, probabilities
        
        return predictions
    
    def predict_latest(self, data_file='Whales_and_Price/whale_prices_ml_ready.csv'):
        """Predict on the latest day in the dataset"""
        print("\n" + "="*70)
        print(" PREDICTING LATEST DAY ".center(70))
        print("="*70)
        
        df = pd.read_csv(data_file)
        
        # Get latest row
        if 'block_date' in df.columns:
            df['block_date'] = pd.to_datetime(df['block_date'])
            df = df.sort_values('block_date')
            latest_date = df['block_date'].iloc[-1]
        else:
            latest_date = "Unknown"
        
        latest_row = df.iloc[[-1]]
        
        # Predict
        predictions, probabilities = self.predict(latest_row, return_proba=True)
        
        # Display results
        print(f"\n📅 Date: {latest_date}")
        print(f"\n📊 Prediction:")
        
        pred = predictions[0]
        prob_down = probabilities[0][0]
        prob_up = probabilities[0][1]
        
        if pred == 1:
            print(f"   🟢 UP (Price will likely increase tomorrow)")
        else:
            print(f"   🔴 DOWN (Price will likely decrease tomorrow)")
        
        print(f"\n📈 Confidence:")
        print(f"   Down: {prob_down*100:.1f}%")
        print(f"   Up:   {prob_up*100:.1f}%")
        
        # Show key metrics if available
        if 'whale_net_exchange_flow_weth' in latest_row.columns:
            print(f"\n🐋 Key Whale Metrics:")
            print(f"   Net Exchange Flow: {latest_row['whale_net_exchange_flow_weth'].iloc[0]:,.1f} WETH")
        if 'total_whale_volume_weth' in latest_row.columns:
            print(f"   Total Whale Volume: {latest_row['total_whale_volume_weth'].iloc[0]:,.1f} WETH")
        if 'eth_price' in df.columns:
            print(f"   Current ETH Price: ${df['eth_price'].iloc[-1]:,.2f}")
    
    def predict_batch(self, data_file):
        """Predict on a batch of data"""
        print("\n" + "="*70)
        print(" BATCH PREDICTIONS ".center(70))
        print("="*70)
        
        df = pd.read_csv(data_file)
        print(f"\n📁 Loaded {len(df)} rows from {data_file}")
        
        # Predict
        predictions, probabilities = self.predict(df, return_proba=True)
        
        # Add results to dataframe
        results = df.copy()
        results['predicted_direction'] = predictions
        results['probability_down'] = probabilities[:, 0]
        results['probability_up'] = probabilities[:, 1]
        results['prediction_label'] = results['predicted_direction'].map({0: 'Down', 1: 'Up'})
        
        # Summary
        print(f"\n📊 Prediction Summary:")
        print(f"   Total predictions: {len(predictions)}")
        print(f"   Predicted UP:   {(predictions == 1).sum()} ({(predictions == 1).mean()*100:.1f}%)")
        print(f"   Predicted DOWN: {(predictions == 0).sum()} ({(predictions == 0).mean()*100:.1f}%)")
        
        # Save results
        output_file = data_file.replace('.csv', '_predictions.csv')
        
        # Select relevant columns for output
        output_cols = []
        if 'block_date' in results.columns:
            output_cols.append('block_date')
        if 'eth_price' in results.columns:
            output_cols.append('eth_price')
        
        output_cols.extend([
            'predicted_direction', 'prediction_label', 
            'probability_down', 'probability_up'
        ])
        
        # Add actual if available
        if 'next_day_price_direction' in results.columns:
            output_cols.append('next_day_price_direction')
            
            # Calculate accuracy
            actual = results['next_day_price_direction'].dropna()
            pred_subset = predictions[:len(actual)]
            accuracy = (actual == pred_subset).mean()
            print(f"\n✅ Accuracy on this data: {accuracy:.4f}")
        
        results[output_cols].to_csv(output_file, index=False)
        print(f"\n💾 Saved predictions: {output_file}")
        
        return results
    
    def interactive_predict(self):
        """Interactive prediction mode"""
        print("\n" + "="*70)
        print(" INTERACTIVE PREDICTION MODE ".center(70))
        print("="*70)
        
        print("\nEnter whale activity metrics to predict next day price direction.")
        print("Press Ctrl+C to exit.\n")
        
        # Create a sample input template
        sample_input = {feat: 0.0 for feat in self.feature_names}
        
        # Key features to prompt for
        key_features = [
            'whale_net_exchange_flow_weth',
            'total_whale_volume_weth',
            'eth_daily_return',
            'btc_daily_return',
            'volume_vs_ma7'
        ]
        
        available_key_features = [f for f in key_features if f in self.feature_names]
        
        if not available_key_features:
            print("⚠️  No key features available for interactive input.")
            print("   Using all features with default values.")
            available_key_features = self.feature_names[:5]
        
        try:
            while True:
                print("\n" + "-"*70)
                print("Enter values for key features (press Enter to use default 0):")
                
                input_data = sample_input.copy()
                
                for feature in available_key_features:
                    value = input(f"  {feature}: ").strip()
                    if value:
                        try:
                            input_data[feature] = float(value)
                        except ValueError:
                            print(f"    ⚠️  Invalid value, using 0")
                            input_data[feature] = 0.0
                
                # Make prediction
                df_input = pd.DataFrame([input_data])
                predictions, probabilities = self.predict(df_input, return_proba=True)
                
                pred = predictions[0]
                prob_down = probabilities[0][0]
                prob_up = probabilities[0][1]
                
                print("\n📊 Prediction:")
                if pred == 1:
                    print("   🟢 UP (Price will likely increase tomorrow)")
                else:
                    print("   🔴 DOWN (Price will likely decrease tomorrow)")
                
                print(f"\n📈 Confidence:")
                print(f"   Down: {prob_down*100:.1f}%")
                print(f"   Up:   {prob_up*100:.1f}%")
                
        except KeyboardInterrupt:
            print("\n\n✅ Exiting interactive mode.")


def main():
    """Main prediction function"""
    parser = argparse.ArgumentParser(
        description='ETH Whale Activity Price Predictor'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to CSV file with whale data'
    )
    parser.add_argument(
        '--latest',
        action='store_true',
        help='Predict on the latest day only'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive prediction mode'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/best_model.pkl',
        help='Path to trained model file'
    )
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = ETHPricePredictor(model_file=args.model)
    
    # Run appropriate mode
    if args.interactive:
        predictor.interactive_predict()
    elif args.latest:
        data_file = args.data if args.data else 'whale_prices_ml_ready.csv'
        predictor.predict_latest(data_file)
    elif args.data:
        predictor.predict_batch(args.data)
    else:
        # Default: predict on latest
        print("No mode specified. Using --latest mode.")
        predictor.predict_latest()


if __name__ == "__main__":
    main()