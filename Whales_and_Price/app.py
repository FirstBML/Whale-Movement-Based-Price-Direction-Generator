"""
app.py - ETH Whale Activity Price Predictor API with Auto-Refresh

FastAPI web service with automated daily data fetching and prediction caching.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 9696
"""

import os
import pickle
import warnings
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler

# Import our data pipeline
from data_pipeline import fetch_and_prepare_data

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="ETH Whale Activity Price Predictor API",
    description="Predict ETH price movements with automated daily data refresh",
    version="3.0.0",
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
CACHED_PREDICTION = None
LAST_UPDATE = None

# File paths
MODEL_FILE = 'models/best_model.pkl'
DATA_FILE = 'whale_prices_ml_ready.csv'
CACHE_FILE = 'prediction_cache.json'

# Admin authentication
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")


# ==================== AUTHENTICATION ====================

def verify_admin_key(x_admin_key: str = Header(..., alias="x-admin-key")):
    """
    Verify admin API key with secure comparison
    
    Args:
        x_admin_key: Admin API key from request header
        
    Raises:
        HTTPException: If key is invalid or not configured
    """
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin authentication not configured on server"
        )
    
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key"
        )
    return True


# ==================== PYDANTIC MODELS ====================

class HealthResponse(BaseModel):
    status: str
    model: Optional[str] = None
    trained_date: Optional[str] = None
    features: Optional[int] = None
    data_loaded: bool
    latest_data_date: Optional[str] = None
    total_records: Optional[int] = None
    last_update: Optional[str] = None
    next_update: Optional[str] = None
    cache_available: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    trained_date: str
    test_accuracy: Optional[float]
    test_recall: Optional[float]
    test_auc: Optional[float]
    n_features: int


class KeyMetrics(BaseModel):
    net_exchange_flow: Optional[float] = None
    total_whale_volume: Optional[float] = None
    eth_price: Optional[float] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "net_exchange_flow": -1250.5,
                "total_whale_volume": 35465343.56,
                "eth_price": 3235.73
            }
        }


class PredictionResponse(BaseModel):
    success: bool
    date: str
    next_day_prediction: str
    confidence: float
    probability_up: float
    probability_down: float
    key_metrics: Optional[KeyMetrics] = None
    cached: bool = False
    generated_at: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "date": "2025-11-17",
                "next_day_prediction": "Up",
                "confidence": 0.78,
                "probability_up": 0.78,
                "probability_down": 0.22,
                "key_metrics": {
                    "net_exchange_flow": -1250.5,
                    "total_whale_volume": 35465343.56,
                    "eth_price": 3235.73
                },
                "cached": True,
                "generated_at": "2025-11-17T10:30:00"
            }
        }


class RefreshResponse(BaseModel):
    success: bool
    message: str
    latest_date: Optional[str] = None
    timestamp: str


class AdminRefreshRequest(BaseModel):
    force: bool = Field(False, description="Force refresh even if data is recent")

class PredictionHistoryItem(BaseModel):
    date: str
    prediction: str
    confidence: float
    probability_up: float
    probability_down: float
    eth_price: Optional[float] = None
    
class PredictionHistoryResponse(BaseModel):
    success: bool
    total_predictions: int
    predictions: List[PredictionHistoryItem]
    date_range: dict



# ==================== CORE FUNCTIONS ====================

def load_model():
    """Load trained model"""
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
    """Load whale data from CSV"""
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
        return False
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


def load_cache():
    """Load cached prediction if available"""
    global CACHED_PREDICTION, LAST_UPDATE
    
    try:
        if Path(CACHE_FILE).exists():
            with open(CACHE_FILE, 'r') as f:
                CACHED_PREDICTION = json.load(f)
            
            LAST_UPDATE = datetime.fromisoformat(CACHED_PREDICTION['generated_at'])
            print(f"✅ Loaded cached prediction from {CACHED_PREDICTION['date']}")
            return True
    except Exception as e:
        print(f"⚠️  Could not load cache: {e}")
    
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


def refresh_data_and_predict():
    """
    Main refresh function: Fetch new data and generate prediction
    
    This is called by the scheduler and can be triggered manually.
    """
    global DATA_DF, CACHED_PREDICTION, LAST_UPDATE
    
    print("\n" + "="*70)
    print(" AUTOMATED DATA REFRESH ".center(70))
    print("="*70)
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Fetch and prepare fresh data
        df_new = fetch_and_prepare_data(save_file=DATA_FILE)
        
        # 2. Update global DataFrame
        DATA_DF = df_new
        
        # 3. Make prediction on latest data
        latest_row = DATA_DF.iloc[[-1]]
        date_str = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d')
        
        X_scaled = prepare_features(latest_row)
        pred = MODEL.predict(X_scaled)[0]
        proba = MODEL.predict_proba(X_scaled)[0]
        
        key_metrics = get_key_metrics(latest_row.iloc[0])
        
        # 4. Cache the prediction
        prediction_data = {
            "success": True,
            "date": date_str,
            "next_day_prediction": "Up" if pred == 1 else "Down",
            "confidence": float(max(proba)),
            "probability_up": float(proba[1]),
            "probability_down": float(proba[0]),
            "key_metrics": {
                "net_exchange_flow": key_metrics.net_exchange_flow,
                "total_whale_volume": key_metrics.total_whale_volume,
                "eth_price": key_metrics.eth_price
            },
            "cached": True,
            "generated_at": datetime.now().isoformat()
        }
        
        CACHED_PREDICTION = prediction_data
        LAST_UPDATE = datetime.now()
        
        # Save cache to file
        with open(CACHE_FILE, 'w') as f:
            json.dump(prediction_data, f, indent=2)
        
        print(f"\n✅ Prediction cached for {date_str}")
        print(f"   Prediction: {prediction_data['next_day_prediction']}")
        print(f"   Confidence: {prediction_data['confidence']:.2%}")
        print(f"   Next update: Tomorrow at 00:30 UTC")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Data refresh failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def schedule_daily_refresh():
    """Schedule daily data refresh at 00:30 UTC"""
    scheduler = BackgroundScheduler()
    
    # Run every day at 00:30 UTC (after market closes and data is available)
    scheduler.add_job(
        refresh_data_and_predict,
        trigger='cron',
        hour=0,
        minute=30,
        id='daily_refresh'
    )
    
    scheduler.start()
    print("\n⏰ Scheduled daily refresh at 00:30 UTC")


# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("\n" + "="*70)
    print(" STARTING API SERVER ".center(70))
    print("="*70)
    
    # Check if admin key is configured
    if not ADMIN_API_KEY:
        print("⚠️  ADMIN_API_KEY not set - refresh endpoint will be disabled")
    else:
        print("✅ Admin authentication configured")
    
    # Load model
    model_loaded = load_model()
    if not model_loaded:
        print("\n❌ Failed to load model. API will not function.")
        return
    
    # Load existing data
    data_loaded = load_data()
    
    # Load cached prediction
    cache_loaded = load_cache()
    
    # Check if we need to refresh
    if not cache_loaded or not data_loaded:
        print("\n🔄 No cache found. Running initial data refresh...")
        refresh_data_and_predict()
    else:
        # Check if cache is from today
        cache_date = datetime.fromisoformat(CACHED_PREDICTION['generated_at']).date()
        if cache_date < datetime.now().date():
            print("\n🔄 Cache is stale. Refreshing data...")
            refresh_data_and_predict()
        else:
            print(f"\n✅ Using cached prediction from {cache_date}")
    
    # Start scheduler
    schedule_daily_refresh()
    
    print("\n✅ API ready!")


# ==================== API ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page"""
    if MODEL is None:
        model_status = "❌ Not loaded"
    else:
        model_status = f"✅ {MODEL_INFO['model_name']}"
    
    if DATA_DF is None:
        data_status = "❌ Not loaded"
        latest_date = "N/A"
    else:
        data_status = f"✅ {len(DATA_DF)} records"
        latest_date = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d')
    
    cache_status = "✅ Available" if CACHED_PREDICTION else "❌ Not cached"
    last_update_str = LAST_UPDATE.strftime('%Y-%m-%d %H:%M UTC') if LAST_UPDATE else "N/A"
    admin_status = "✅ Configured" if ADMIN_API_KEY else "❌ Disabled"
       
    docs = f"""
    <html>
    <head>
        <title>ETH Price Predictor API</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 8px; max-width: 900px; margin: 0 auto; }}
            h1 {{ color: #2c3e50; }}
            .status {{ padding: 15px; background: #d4edda; border-radius: 5px; margin: 20px 0; }}
            .highlight {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
            .docs-link {{ background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 10px 5px; }}
            .docs-link:hover {{ background: #0056b3; }}
            .refresh-btn {{ background: #28a745; }}
            .refresh-btn:hover {{ background: #218838; }}
            .admin-btn {{ background: #dc3545; }}
            .admin-btn:hover {{ background: #c82333; }}
            .admin-info {{ background: #e2e3e5; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            
            /* Modal styles */
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.4);
            }}
            .modal-content {{
                background-color: #fefefe;
                margin: 15% auto;
                padding: 30px;
                border-radius: 8px;
                width: 400px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .modal-header {{
                margin-bottom: 20px;
            }}
            .modal-header h2 {{
                margin: 0;
                color: #2c3e50;
            }}
            .form-group {{
                margin-bottom: 20px;
            }}
            .form-group label {{
                display: block;
                margin-bottom: 8px;
                font-weight: bold;
                color: #555;
            }}
            .form-group input {{
                width: 100%;
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
            }}
            .form-group input:focus {{
                outline: none;
                border-color: #007bff;
            }}
            .modal-buttons {{
                display: flex;
                gap: 10px;
                justify-content: flex-end;
            }}
            .btn {{
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
            }}
            .btn-primary {{
                background-color: #dc3545;
                color: white;
            }}
            .btn-primary:hover {{
                background-color: #c82333;
            }}
            .btn-secondary {{
                background-color: #6c757d;
                color: white;
            }}
            .btn-secondary:hover {{
                background-color: #5a6268;
            }}
            .close {{
                color: #aaa;
                float: right;
                font-size: 28px;
                font-weight: bold;
                cursor: pointer;
            }}
            .close:hover {{
                color: #000;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🐋 ETH Whale Activity Price Predictor</h1>
            <h3>🔄 Auto-Refresh Enabled | 🔐 Admin Protected</h3>
            
            <div class="status">
                <strong>Model:</strong> {model_status}<br>
                <strong>Data:</strong> {data_status}<br>
                <strong>Latest Data:</strong> {latest_date}<br>
                <strong>Cached Prediction:</strong> {cache_status}<br>
                <strong>Last Update:</strong> {last_update_str}<br>
                <strong>Admin Auth:</strong> {admin_status}<br>
                <strong>Next Auto-Update:</strong> Tomorrow at 00:30 UTC
            </div>
            
            <div class="highlight">
                <strong>🎯 Get Latest Prediction:</strong><br>
                <code>GET /predict/latest</code> - Instant cached result
            </div>
            
            <div class="admin-info">
                <strong>🔐 Admin Access Required:</strong><br>
                <code>/refresh</code> - Force data refresh (Requires Admin Key)
            </div>
            
            <div>
                <a href="/docs" class="docs-link">📖 API Docs</a>
                <a href="/predict/latest" class="docs-link">🔮 Get Prediction</a>
                <a href="/predict/history" class="docs-link">📊 History</a>
                <a href="/health" class="docs-link">❤️ Health Check</a>
                <button onclick="showAdminModal()" class="docs-link admin-btn">🔄 Force Refresh (Admin)</button>
            </div>
            
            <h2>🚀 Features</h2>
            <ul>
                <li>✅ Auto-refresh: Fetches latest data daily at 00:30 UTC</li>
                <li>✅ Cached predictions: Instant results</li>
                <li>✅ Prediction history: View past predictions</li>
                <li>🔐 Admin refresh: Secure manual updates</li>
            </ul>
        </div>

        <!-- Admin Modal -->
        <div id="adminModal" class="modal">
            <div class="modal-content">
                <div class="modal-header">
                    <span class="close" onclick="closeAdminModal()">&times;</span>
                    <h2>🔐 Admin Authentication</h2>
                </div>
                <form onsubmit="submitRefresh(event)">
                    <div class="form-group">
                        <label for="adminKey">Admin API Key:</label>
                        <input 
                            type="password" 
                            id="adminKey" 
                            name="adminKey" 
                            placeholder="Enter your admin key"
                            required
                            autocomplete="off"
                        >
                    </div>
                    <div class="modal-buttons">
                        <button type="button" class="btn btn-secondary" onclick="closeAdminModal()">Cancel</button>
                        <button type="submit" class="btn btn-primary">🔄 Refresh Data</button>
                    </div>
                </form>
            </div>
        </div>
        
        <script>
            function showAdminModal() {{
                document.getElementById('adminModal').style.display = 'block';
                document.getElementById('adminKey').focus();
            }}
            
            function closeAdminModal() {{
                document.getElementById('adminModal').style.display = 'none';
                document.getElementById('adminKey').value = '';
            }}
            
            // Close modal when clicking outside
            window.onclick = function(event) {{
                const modal = document.getElementById('adminModal');
                if (event.target == modal) {{
                    closeAdminModal();
                }}
            }}
            
            // Close modal on Escape key
            document.addEventListener('keydown', function(event) {{
                if (event.key === 'Escape') {{
                    closeAdminModal();
                }}
            }});
            
            function submitRefresh(event) {{
                event.preventDefault();
                
                const adminKey = document.getElementById('adminKey').value;
                const submitBtn = event.target.querySelector('.btn-primary');
                
                // Disable button and show loading
                submitBtn.disabled = true;
                submitBtn.textContent = '⏳ Refreshing...';
                
                fetch('/refresh', {{
                    method: 'POST',
                    headers: {{
                        'X-Admin-Key': adminKey,
                        'Content-Type': 'application/json'
                    }}
                }})
                .then(response => {{
                    if (!response.ok) {{
                        return response.json().then(err => {{
                            throw new Error(err.detail || 'Authentication failed');
                        }});
                    }}
                    return response.json();
                }})
                .then(data => {{
                    alert('✅ ' + data.message);
                    closeAdminModal();
                    setTimeout(() => location.reload(), 2000);
                }})
                .catch(error => {{
                    alert('❌ Refresh failed: ' + error.message);
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🔄 Refresh Data';
                }});
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=docs)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    # FIX: Use 'is not None' instead of truthy check for DataFrame
    status_str = "healthy" if (MODEL is not None and DATA_DF is not None) else "degraded"
    
    latest_date = None
    if DATA_DF is not None and 'block_date' in DATA_DF.columns:
        latest_date = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d')
    
    last_update_str = LAST_UPDATE.isoformat() if LAST_UPDATE else None
    
    # Next update at 00:30 UTC tomorrow
    now = datetime.utcnow()
    next_update = datetime(now.year, now.month, now.day, 0, 30) + timedelta(days=1)
    
    return HealthResponse(
        status=status_str,
        model=MODEL_INFO['model_name'] if MODEL else None,
        trained_date=MODEL_INFO['trained_date'] if MODEL else None,
        features=MODEL_INFO['n_features'] if MODEL else None,
        data_loaded=DATA_DF is not None,
        latest_data_date=latest_date,
        total_records=len(DATA_DF) if DATA_DF is not None else None,
        last_update=last_update_str,
        next_update=next_update.isoformat(),
        cache_available=CACHED_PREDICTION is not None
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
    Get cached prediction (recommended)
    
    Returns the cached prediction from the last data refresh.
    Fast and resource-efficient.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if CACHED_PREDICTION is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No prediction available. Try POST /refresh"
        )
    
    return PredictionResponse(**CACHED_PREDICTION)


@app.post("/refresh", response_model=RefreshResponse)
async def manual_refresh(
    background_tasks: BackgroundTasks,
    admin_authenticated: bool = Depends(verify_admin_key)
):
    """
    🔐 ADMIN: Manually trigger data refresh
    
    Fetches fresh data from Dune and CoinGecko, then generates new prediction.
    Requires admin authentication via X-Admin-Key header.
    
    Headers:
        X-Admin-Key: Your admin API key
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    print(f"\n🔐 ADMIN: Manual refresh triggered at {datetime.now().isoformat()}")
    
    # Run refresh in background
    background_tasks.add_task(refresh_data_and_predict)
    
    return RefreshResponse(
        success=True,
        message="Admin data refresh triggered. Check /predict/latest in 1-2 minutes.",
        latest_date=DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d') if DATA_DF is not None else None,
        timestamp=datetime.now().isoformat()
    )

@app.get("/predict/history", response_model=PredictionHistoryResponse)
async def prediction_history(limit: int = 30, days: int = 30):
    """
    Get historical predictions
    
    Args:
        limit: Maximum number of predictions to return (default: 30)
        days: Number of days to look back (default: 30)
    
    Returns:
        Historical predictions with dates and confidence scores
    """
    if MODEL is None or DATA_DF is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or data not loaded"
        )
    
    try:
        # Get last N days of data
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter data
        if 'block_date' in DATA_DF.columns:
            recent_df = DATA_DF[DATA_DF['block_date'] >= cutoff_date].copy()
        else:
            recent_df = DATA_DF.tail(limit).copy()
        
        # Limit results
        recent_df = recent_df.tail(limit)
        
        # Make predictions for historical data
        X_scaled = prepare_features(recent_df)
        predictions = MODEL.predict(X_scaled)
        probabilities = MODEL.predict_proba(X_scaled)
        
        # Build history list
        history = []
        for idx, (_, row) in enumerate(recent_df.iterrows()):
            pred = predictions[idx]
            proba = probabilities[idx]
            
            date_str = row['block_date'].strftime('%Y-%m-%d') if 'block_date' in row else f"Record {idx}"
            eth_price = float(row['eth_price']) if 'eth_price' in row else None
            
            history.append(PredictionHistoryItem(
                date=date_str,
                prediction="Up" if pred == 1 else "Down",
                confidence=float(max(proba)),
                probability_up=float(proba[1]),
                probability_down=float(proba[0]),
                eth_price=eth_price
            ))
        
        # Date range info
        date_range = {
            "start": history[0].date if history else None,
            "end": history[-1].date if history else None
        }
        
        return PredictionHistoryResponse(
            success=True,
            total_predictions=len(history),
            predictions=history,
            date_range=date_range
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate history: {str(e)}"
        )

if __name__ == '__main__':
    import uvicorn
    
    print("\n" + "="*70)
    print(" ETH PRICE PREDICTOR API ".center(70))
    print("="*70)
    print(f"\n🚀 Server: http://localhost:9696")
    print(f"📖 API Docs: http://localhost:9696/docs")
    print(f"🎯 Main Endpoint: http://localhost:9696/predict/latest")
    print(f"🔐 Admin Refresh: POST http://localhost:9696/refresh (with X-Admin-Key header)")
    
    if not ADMIN_API_KEY:
        print("⚠️  WARNING: ADMIN_API_KEY not set - refresh endpoint disabled")
    
    print(f"\n💡 Press Ctrl+C to stop\n")
    
    uvicorn.run(app, host="0.0.0.0", port=9696)