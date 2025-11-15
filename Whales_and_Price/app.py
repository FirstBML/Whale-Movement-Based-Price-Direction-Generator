"""
app.py - ETH Whale Activity Price Predictor API

FastAPI web service for making predictions on whale data.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 9696

The service will start on http://localhost:9696
API docs will be available at http://localhost:9696/docs
"""

import pickle
import warnings
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="ETH Whale Activity Price Predictor API",
    description="Predict ETH price movements based on whale activity",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
SCALER = None
FEATURE_NAMES = None
MODEL_INFO = None
DATA_DF = None
MODEL_FILE = 'models/best_model.pkl'
DATA_FILE = 'whale_prices_ml_ready.csv'


# Pydantic models
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: Optional[str] = None
    trained_date: Optional[str] = None
    features: Optional[int] = None
    data_loaded: bool
    latest_data_date: Optional[str] = None
    total_records: Optional[int] = None


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_name: str
    trained_date: str
    test_accuracy: Optional[float]
    test_recall: Optional[float]
    test_auc: Optional[float]
    n_features: int


class KeyMetrics(BaseModel):
    """Key whale metrics"""
    net_exchange_flow: Optional[float] = None
    total_whale_volume: Optional[float] = None
    eth_price: Optional[float] = None


class PredictionResponse(BaseModel):
    """Prediction response"""
    success: bool
    date: str  # Date of whale data used
    next_day_prediction: str  # "Up" or "Down" - prediction for NEXT day
    confidence: float
    probability_up: float
    probability_down: float
    key_metrics: Optional[KeyMetrics] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "date": "2025-11-14",
                "next_day_prediction": "Up",
                "confidence": 0.78,
                "probability_up": 0.78,
                "probability_down": 0.22,
                "key_metrics": {
                    "net_exchange_flow": -1250.5,
                    "total_whale_volume": 35465343.56,
                    "eth_price": 3235.73
                }
            }
        }


class HistoricalPrediction(BaseModel):
    """Historical prediction with date"""
    date: str  # Date of whale data
    next_day_prediction: str  # Prediction for the NEXT day
    confidence: float
    probability_up: float
    probability_down: float
    eth_price: Optional[float] = None


class HistoricalResponse(BaseModel):
    """Historical predictions response"""
    success: bool
    predictions: List[HistoricalPrediction]
    count: int
    date_range: str


def load_model():
    """Load model at startup"""
    global MODEL, SCALER, FEATURE_NAMES, MODEL_INFO
    
    print("🔄 Loading model...")
    
    try:
        with open(MODEL_FILE, 'rb') as f:
            model_dict = pickle.load(f)
        
        MODEL = model_dict['model']
        SCALER = model_dict['scaler']
        FEATURE_NAMES = model_dict['feature_names']
        MODEL_INFO = {
            'model_name': model_dict.get('model_name', 'Unknown'),
            'trained_date': model_dict.get('trained_date', 'Unknown'),
            'test_accuracy': model_dict.get('test_accuracy'),
            'test_recall': model_dict.get('test_recall'),
            'test_auc': model_dict.get('test_auc'),
            'n_features': len(model_dict['feature_names'])
        }
        
        print(f"✅ Model loaded: {MODEL_INFO['model_name']}")
        print(f"   Trained: {MODEL_INFO['trained_date']}")
        print(f"   Test Accuracy: {MODEL_INFO['test_accuracy']:.4f}")
        print(f"   Test Recall: {MODEL_INFO['test_recall']:.4f}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODEL_FILE}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def load_data():
    """Load whale data at startup"""
    global DATA_DF
    
    print("🔄 Loading whale data...")
    
    try:
        DATA_DF = pd.read_csv(DATA_FILE)
        
        if 'block_date' in DATA_DF.columns:
            DATA_DF['block_date'] = pd.to_datetime(DATA_DF['block_date'])
            DATA_DF = DATA_DF.sort_values('block_date')
            latest_date = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d')
        else:
            latest_date = "Unknown"
        
        print(f"✅ Data loaded: {len(DATA_DF)} records")
        print(f"   Latest date: {latest_date}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Data file not found: {DATA_FILE}")
        print("   API will run but predictions won't be available")
        return False
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def prepare_features(df):
    """Prepare features for prediction"""
    missing_features = set(FEATURE_NAMES) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features: {list(missing_features)[:5]}")
    
    X = df[FEATURE_NAMES].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = SCALER.transform(X)
    
    return X_scaled


def get_key_metrics(row):
    """Extract key metrics from data row"""
    metrics = KeyMetrics()
    
    if 'whale_net_exchange_flow_weth' in row:
        metrics.net_exchange_flow = float(row['whale_net_exchange_flow_weth'])
    if 'total_whale_volume_weth' in row:
        metrics.total_whale_volume = float(row['total_whale_volume_weth'])
    if 'eth_price' in row:
        metrics.eth_price = float(row['eth_price'])
    
    return metrics


@app.on_event("startup")
async def startup_event():
    """Load model and data when API starts"""
    model_loaded = load_model()
    data_loaded = load_data()
    
    if not model_loaded:
        print("\n❌ Failed to load model. API will not function properly.")
    if not data_loaded:
        print("\n⚠️  Failed to load data. Predictions won't be available.")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with API documentation"""
    if MODEL is None:
        model_status = "❌ Not loaded"
    else:
        model_status = f"✅ {MODEL_INFO['model_name']}"
    
    if DATA_DF is None:
        data_status = "❌ Not loaded"
        latest_date = "N/A"
    else:
        data_status = f"✅ {len(DATA_DF)} records"
        latest_date = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d') if 'block_date' in DATA_DF.columns else "Unknown"
    
    docs = f"""
    <html>
    <head>
        <title>ETH Price Predictor API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 8px; max-width: 900px; margin: 0 auto; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            code {{ background: #ecf0f1; padding: 2px 6px; border-radius: 3px; font-family: monospace; }}
            pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: monospace; }}
            .endpoint {{ background: #e8f5e9; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #4caf50; }}
            .method {{ display: inline-block; padding: 4px 8px; border-radius: 3px; font-weight: bold; margin-right: 10px; }}
            .get {{ background: #61affe; color: white; }}
            .status {{ padding: 10px; background: #fff3cd; border-radius: 5px; margin: 20px 0; }}
            .docs-link {{ background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 5px; }}
            .docs-link:hover {{ background: #0056b3; }}
            .highlight {{ background: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐋 ETH Whale Activity Price Predictor</h1>
            
            <div class="status">
                <strong>Model:</strong> {model_status}<br>
                <strong>Data:</strong> {data_status}<br>
                <strong>Latest Data:</strong> {latest_date}<br>
                <strong>Server:</strong> Running on port 9696
            </div>
            
            <div class="highlight">
                <strong>🎯 Main Endpoint:</strong> Predict TOMORROW's ETH price direction based on TODAY's whale activity<br>
                <code>GET /predict/latest</code>
            </div>
            
            <div style="margin: 20px 0;">
                <a href="/docs" class="docs-link">📖 Interactive API Docs</a>
                <a href="/redoc" class="docs-link">📘 ReDoc</a>
            </div>
            
            <h2>📡 Endpoints</h2>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/health</strong><br>
                Health check - verify API and data status
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/model-info</strong><br>
                Get model performance metrics
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/predict/latest</strong><br>
                🎯 <strong>Main endpoint:</strong> Predict TOMORROW's price direction using TODAY's latest whale data
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/predict/date/{{date}}</strong><br>
                Predict for a specific date (format: YYYY-MM-DD)
            </div>
            
            <div class="endpoint">
                <span class="method get">GET</span>
                <strong>/predict/historical</strong><br>
                Get predictions for the last N days (default: 7)
            </div>
            
            <h2>📝 Example Usage</h2>
            
            <h3>Get Latest Prediction (Most Common)</h3>
            <pre>curl http://localhost:9696/predict/latest</pre>
            
            <h3>Response Example</h3>
            <pre>{{
  "success": true,
  "date": "2024-03-15",
  "next_day_prediction": "Up",
  "confidence": 0.78,
  "probability_up": 0.78,
  "probability_down": 0.22,
  "key_metrics": {{
    "net_exchange_flow": -1250.5,
    "total_whale_volume": 8500.2,
    "eth_price": 3245.67
  }}
}}</pre>
            
            <p><strong>Note:</strong> If today is March 15, this predicts March 16's price direction!</p>
            
            <h3>Predict for Specific Date</h3>
            <pre>curl http://localhost:9696/predict/date/2024-03-10</pre>
            
            <h3>Get Last 30 Days of Predictions</h3>
            <pre>curl "http://localhost:9696/predict/historical?days=30"</pre>
            
            <h2>💡 How It Works</h2>
            <ul>
                <li>📊 The API automatically loads whale activity data from the CSV file</li>
                <li>🤖 The trained ML model analyzes 62 features from TODAY's whale behavior</li>
                <li>🎯 It predicts whether ETH price will go UP or DOWN TOMORROW</li>
                <li>📈 Confidence scores show how certain the model is about the prediction</li>
                <li>⏰ The prediction is for the NEXT DAY based on CURRENT whale activity</li>
            </ul>
            
            <h2>🔍 Understanding the Results</h2>
            <ul>
                <li><strong>Date:</strong> The date of whale data used (e.g., March 15)</li>
                <li><strong>Next Day Prediction:</strong> "Up" or "Down" - predicted price direction for TOMORROW (e.g., March 16)</li>
                <li><strong>Confidence:</strong> 0-1 scale (higher = more certain)</li>
                <li><strong>Net Exchange Flow:</strong> Negative = whales withdrawing (bullish), Positive = whales depositing (bearish)</li>
                <li><strong>Whale Volume:</strong> Total WETH moved by large holders</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=docs)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if MODEL is None or DATA_DF is None:
        status_str = "degraded" if MODEL is not None or DATA_DF is not None else "unhealthy"
    else:
        status_str = "healthy"
    
    latest_date = None
    if DATA_DF is not None and 'block_date' in DATA_DF.columns:
        latest_date = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d')
    
    return HealthResponse(
        status=status_str,
        model=MODEL_INFO['model_name'] if MODEL else None,
        trained_date=MODEL_INFO['trained_date'] if MODEL else None,
        features=MODEL_INFO['n_features'] if MODEL else None,
        data_loaded=DATA_DF is not None,
        latest_data_date=latest_date,
        total_records=len(DATA_DF) if DATA_DF is not None else None
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfoResponse(
        model_name=MODEL_INFO['model_name'],
        trained_date=MODEL_INFO['trained_date'],
        test_accuracy=MODEL_INFO['test_accuracy'],
        test_recall=MODEL_INFO['test_recall'],
        test_auc=MODEL_INFO['test_auc'],
        n_features=MODEL_INFO['n_features']
    )


@app.get("/predict/latest", response_model=PredictionResponse)
async def predict_latest():
    """
    Predict TOMORROW's price direction based on TODAY's latest whale data
    
    Uses the most recent whale activity data to predict next-day price movement.
    Example: If latest data is from March 15, this predicts March 16's direction.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if DATA_DF is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Data not loaded"
        )
    
    try:
        # Get latest row
        latest_row = DATA_DF.iloc[[-1]]
        
        if 'block_date' in DATA_DF.columns:
            date_str = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d')
        else:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Prepare and predict
        X_scaled = prepare_features(latest_row)
        pred = MODEL.predict(X_scaled)[0]
        proba = MODEL.predict_proba(X_scaled)[0]
        
        # Extract key metrics
        key_metrics = get_key_metrics(latest_row.iloc[0])
        
        return PredictionResponse(
            success=True,
            date=date_str,
            next_day_prediction="Up" if pred == 1 else "Down",
            confidence=float(max(proba)),
            probability_up=float(proba[1]),
            probability_down=float(proba[0]),
            key_metrics=key_metrics
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/predict/date/{date}", response_model=PredictionResponse)
async def predict_date(date: str):
    """
    Predict for a specific date's whale data
    
    Date format: YYYY-MM-DD (e.g., 2024-03-15)
    This predicts the NEXT day's direction based on the given date's whale activity.
    Example: date=2024-03-15 predicts March 16's price direction.
    """
    if MODEL is None or DATA_DF is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or data not loaded"
        )
    
    try:
        target_date = pd.to_datetime(date)
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid date format. Use YYYY-MM-DD"
        )
    
    # Find matching row
    if 'block_date' not in DATA_DF.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Date column not available in data"
        )
    
    matching_rows = DATA_DF[DATA_DF['block_date'] == target_date]
    
    if len(matching_rows) == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No data found for date {date}"
        )
    
    try:
        row = matching_rows.iloc[[-1]]
        
        # Prepare and predict
        X_scaled = prepare_features(row)
        pred = MODEL.predict(X_scaled)[0]
        proba = MODEL.predict_proba(X_scaled)[0]
        
        # Extract key metrics
        key_metrics = get_key_metrics(row.iloc[0])
        
        return PredictionResponse(
            success=True,
            date=date,
            next_day_prediction="Up" if pred == 1 else "Down",
            confidence=float(max(proba)),
            probability_up=float(proba[1]),
            probability_down=float(proba[0]),
            key_metrics=key_metrics
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/predict/historical", response_model=HistoricalResponse)
async def predict_historical(
    days: int = Query(default=7, ge=1, le=365, description="Number of days to predict")
):
    """
    Get next-day predictions for the last N days
    
    Returns predictions for the most recent days in the dataset.
    Each prediction shows what was predicted for the NEXT day based on that day's whale data.
    """
    if MODEL is None or DATA_DF is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or data not loaded"
        )
    
    try:
        # Get last N rows
        last_rows = DATA_DF.iloc[-days:]
        
        # Prepare and predict
        X_scaled = prepare_features(last_rows)
        predictions = MODEL.predict(X_scaled)
        probabilities = MODEL.predict_proba(X_scaled)
        
        # Build response
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            row = last_rows.iloc[i]
            
            if 'block_date' in last_rows.columns:
                date_str = pd.to_datetime(row['block_date']).strftime('%Y-%m-%d')
            else:
                date_str = f"Row {i}"
            
            eth_price = float(row['eth_price']) if 'eth_price' in row else None
            
            results.append(
                HistoricalPrediction(
                    date=date_str,
                    next_day_prediction="Up" if pred == 1 else "Down",
                    confidence=float(max(proba)),
                    probability_up=float(proba[1]),
                    probability_down=float(proba[0]),
                    eth_price=eth_price
                )
            )
        
        # Date range
        if 'block_date' in last_rows.columns:
            start_date = last_rows['block_date'].iloc[0].strftime('%Y-%m-%d')
            end_date = last_rows['block_date'].iloc[-1].strftime('%Y-%m-%d')
            date_range = f"{start_date} to {end_date}"
        else:
            date_range = f"Last {len(results)} rows"
        
        return HistoricalResponse(
            success=True,
            predictions=results,
            count=len(results),
            date_range=date_range
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == '__main__':
    import uvicorn
    
    print("\n" + "="*70)
    print(" ETH PRICE PREDICTOR API ".center(70))
    print("="*70)
    print(f"\n🚀 Server: http://localhost:9696")
    print(f"📖 API Docs: http://localhost:9696/docs")
    print(f"🎯 Main Endpoint: http://localhost:9696/predict/latest")
    print(f"\n💡 Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=9696)