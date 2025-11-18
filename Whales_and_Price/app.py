"""
app.py - ETH Whale Activity Price Predictor API with Auto-Refresh

Updated to:
 - Stream refresh progress via Server-Sent Events (SSE)
 - Use APIKeyHeader for OpenAPI/Swagger admin input (shows lock)
 - Home page is a dashboard-style UI served as a static string (no f-strings)
 - Admin modal calls the SSE stream and displays live progress; password input remains masked
 - `refresh_data_and_predict` accepts an optional progress callback for fine-grained updates

Usage:
    uvicorn app:app --host 0.0.0.0 --port 9696
"""

import os
import pickle
import warnings
import json
import secrets
from datetime import datetime, timedelta
from typing import Optional, Callable
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status, BackgroundTasks, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.security import APIKeyHeader

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

# Admin authentication from env
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

# APIKeyHeader for OpenAPI/Swagger integration (this makes the UI show a lock)
api_key_header = APIKeyHeader(name="x-admin-key", auto_error=False)


# ==================== AUTHENTICATION ====================

def verify_admin_key(x_admin_key: str = Header(..., alias="x-admin-key")):
    """
    Verify admin API key with secure comparison (header-based, production use)
    """
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin authentication not configured on server"
        )

    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin API key"
        )

    if not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key"
        )
    return True


async def verify_admin_key_openapi(x_admin_key: str = Depends(api_key_header)):
    """
    This dependency is used so the OpenAPI/Swagger UI shows a security control.
    It is similar to verify_admin_key but works with FastAPI's security dependency.
    It will raise a 401 if the key is incorrect.
    """
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin authentication not configured on server"
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin API key")
    return True


# For SSE (EventSource) we can't set headers easily from the browser. Provide a query-param verifier
def verify_admin_key_query(admin_key: Optional[str] = Query(None, alias='admin_key')):
    if not ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin authentication not configured on server"
        )
    if not admin_key or not secrets.compare_digest(admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin API key")
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


def refresh_data_and_predict(progress_callback: Optional[Callable[[str], None]] = None, force: bool = False):
    """
    Main refresh function: Fetch new data and generate prediction

    Accepts an optional progress_callback(message: str) that will be called at key steps.
    Returns True/False.
    """
    global DATA_DF, CACHED_PREDICTION, LAST_UPDATE

    def _send(msg: str):
        ts = datetime.utcnow().isoformat()
        entry = f"[{ts}] {msg}"
        print(entry)
        if progress_callback:
            try:
                progress_callback(entry)
            except Exception:
                pass

    _send("Starting data refresh...")

    try:
        _send("Fetching and preparing data...")
        df_new = fetch_and_prepare_data(save_file=DATA_FILE)
        if df_new is None or df_new.empty:
            raise RuntimeError("fetch_and_prepare_data returned no data")

        DATA_DF = df_new
        _send(f"Data fetched: {len(DATA_DF)} records")

        _send("Preparing features for latest row...")
        latest_row = DATA_DF.iloc[[-1]]
        date_str = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d')

        X_scaled = prepare_features(latest_row)
        _send("Running model prediction...")
        pred = MODEL.predict(X_scaled)[0]
        proba = MODEL.predict_proba(X_scaled)[0]

        key_metrics = get_key_metrics(latest_row.iloc[0])

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
            "generated_at": datetime.utcnow().isoformat()
        }

        CACHED_PREDICTION = prediction_data
        LAST_UPDATE = datetime.utcnow()

        with open(CACHE_FILE, 'w') as f:
            json.dump(prediction_data, f, indent=2)

        _send(f"Prediction cached for {date_str} (confidence={prediction_data['confidence']:.4f})")
        _send("Refresh complete")

        return True

    except Exception as e:
        _send(f"Data refresh failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def schedule_daily_refresh():
    """Schedule daily data refresh at 00:30 UTC"""
    scheduler = BackgroundScheduler()

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
    print("\n" + "="*70)
    print(" STARTING API SERVER ".center(70))
    print("="*70)

    if not ADMIN_API_KEY:
        print("⚠️  ADMIN_API_KEY not set - refresh endpoint will be disabled")
    else:
        print("✅ Admin authentication configured")

    model_loaded = load_model()
    if not model_loaded:
        print("\n❌ Failed to load model. API will not function.")
        return

    data_loaded = load_data()
    cache_loaded = load_cache()

    if not cache_loaded or not data_loaded:
        print("\n🔄 No cache found. Running initial data refresh...")
        # Run inline to ensure cache exists on startup
        refresh_data_and_predict()
    else:
        cache_date = datetime.fromisoformat(CACHED_PREDICTION['generated_at']).date()
        if cache_date < datetime.utcnow().date():
            print("\n🔄 Cache is stale. Refreshing data...")
            refresh_data_and_predict()
        else:
            print(f"\n✅ Using cached prediction from {cache_date}")

    schedule_daily_refresh()
    print("\n✅ API ready!")


# ==================== API ENDPOINTS ====================
@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page with in-page fetches and SSE-driven refresh UI (static HTML)
    This function returns a static HTML string (no f-strings) to avoid syntax errors
    caused by `{` and `}` inside JavaScript.
    """
    docs = """
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width,initial-scale=1" />
      <title>ETH Whale Activity Price Predictor</title>
      <style>
        body { font-family: Inter, Arial, sans-serif; background:#f4f6f8; margin:0; padding:24px; }
        .wrap { max-width:1100px; margin:0 auto; }
        header { display:flex; align-items:center; justify-content:space-between; margin-bottom:18px; }
        h1 { margin:0; font-size:20px; }
        .grid { display:grid; grid-template-columns: 1fr 360px; gap:18px; }
        .card { background:white; border-radius:10px; padding:18px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
        .row { display:flex; gap:12px; }
        .muted { color:#6b7280; font-size:13px; }
        button { cursor:pointer; border:0; padding:10px 12px; border-radius:8px; font-weight:600; }
        .btn-primary { background:#0b74ff; color:white; }
        .btn-ghost { background:#eef2ff; color:#0b74ff; }
        .btn-danger { background:#dc3545; color:white; }
        #progressBox { background:#0b1220; color:#7ef9a3; padding:12px; height:220px; overflow:auto; font-family:monospace; border-radius:8px; }
        pre { white-space:pre-wrap; word-break:break-word; background:#f8fafc; padding:12px; border-radius:6px; }
        .cards-col { display:grid; gap:12px; }
      </style>
    </head>
    <body>
      <div class="wrap">
        <header>
          <div>
            <h1>🐋 ETH Whale Activity Price Predictor — Dashboard</h1>
            <div class="muted">Auto-refresh daily • Admin-controlled manual refresh</div>
          </div>
          <div class="row">
            <button class="btn-ghost" onclick="loadHealth()">Health</button>
            <button class="btn-primary" onclick="loadPrediction()">Latest Prediction</button>
            <button class="btn-danger" onclick="showAdminModal()">Admin Refresh</button>
          </div>
        </header>

        <div class="grid">
          <div>
            <div class="card">
              <h3>Prediction</h3>
              <div id="predictionCard">
                <div class="muted">No prediction loaded.</div>
              </div>
            </div>

            <div style="height:12px"></div>

            <div class="card">
              <h3>History (Last 30)</h3>
              <div id="historyCard"><div class="muted">Load to view recent predictions.</div></div>
            </div>

          </div>

          <div class="cards-col">
            <div class="card">
              <h3>System Health</h3>
              <div id="healthCard"><div class="muted">No health check yet.</div></div>
            </div>

            <div class="card">
              <h3>Live Refresh Progress</h3>
              <div id="progressBox">No activity</div>
            </div>

            <div class="card">
              <h3>Actions</h3>
              <div style="display:flex; gap:8px; margin-top:8px;">
                <button class="btn-primary" onclick="loadPrediction()">Refresh Prediction</button>
                <button class="btn-ghost" onclick="loadHistory()">Load History</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Admin Modal -->
        <div id="adminModal" style="display:none; position:fixed; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,0.4); z-index:1000;">
            <div style="background:white; padding:20px; width:420px; margin:10% auto; border-radius:8px; position:relative;">
                <span style="position:absolute; right:12px; top:6px; cursor:pointer; font-size:20px;" onclick="closeAdminModal()">&times;</span>
                <h2>🔐 Admin Authentication</h2>
                <form onsubmit="startStreamRefresh(event)">
                    <div style="margin-bottom:12px;">
                        <label for="adminKey"><strong>Admin API Key</strong></label>
                        <input id="adminKey" name="adminKey" type="password" placeholder="Enter admin key" style="width:100%; padding:8px; margin-top:6px;" required autocomplete="off" />
                    </div>
                    <div style="text-align:right;">
                        <button type="button" onclick="closeAdminModal()" style="margin-right:8px; padding:8px 12px;">Cancel</button>
                        <button type="submit" id="refreshBtn" style="background:#dc3545; color:white; padding:8px 12px;">🔄 Start Refresh (Stream)</button>
                    </div>
                </form>
                <p style="font-size:12px; color:#666; margin-top:12px;">Note: For the stream we pass the key as a query param (EventSource cannot set headers). This is intended for local/admin UIs only.</p>
            </div>
        </div>

      </div>

      <script>
        function showAdminModal(){
            document.getElementById('adminModal').style.display='block';
            document.getElementById('adminKey').focus();
        }
        function closeAdminModal(){
            document.getElementById('adminModal').style.display='none';
            document.getElementById('adminKey').value='';
        }

        async function loadPrediction(){
            const card = document.getElementById('predictionCard');
            card.innerHTML = '<div class="muted">Loading...</div>';
            try{
                const res = await fetch('/predict/latest');
                if(!res.ok){ const e = await res.json(); throw new Error(e.detail || 'Failed'); }
                const j = await res.json();
                card.innerHTML = '<pre>' + JSON.stringify(j, null, 2) + '</pre>';
            }catch(err){ card.innerHTML = '<pre style="color:red">' + err.message + '</pre>'; }
        }

        async function loadHealth(){
            const card = document.getElementById('healthCard');
            card.innerHTML = '<div class="muted">Loading...</div>';
            try{
                const res = await fetch('/health');
                if(!res.ok){ const e = await res.json(); throw new Error(e.detail || 'Failed'); }
                const j = await res.json();
                card.innerHTML = '<pre>' + JSON.stringify(j, null, 2) + '</pre>';
            }catch(err){ card.innerHTML = '<pre style="color:red">' + err.message + '</pre>'; }
        }

        async function loadHistory(){
            const card = document.getElementById('historyCard');
            card.innerHTML = '<div class="muted">Loading...</div>';
            try{
                const res = await fetch('/predict/history?limit=30&days=30');
                if(!res.ok){ const e = await res.json(); throw new Error(e.detail || 'Failed'); }
                const j = await res.json();
                card.innerHTML = '<pre>' + JSON.stringify(j, null, 2) + '</pre>';
            }catch(err){ card.innerHTML = '<pre style="color:red">' + err.message + '</pre>'; }
        }

        // Start SSE refresh stream
        function startStreamRefresh(e){
            e.preventDefault();
            const adminKey = document.getElementById('adminKey').value;
            if(!adminKey) return alert('Admin key required');

            const btn = document.getElementById('refreshBtn');
            btn.disabled = true;
            btn.textContent = '⏳ Starting...';

            const box = document.getElementById('progressBox');
            box.innerHTML = '';

            // Use EventSource with admin_key query param
            const src = new EventSource('/refresh/stream?admin_key=' + encodeURIComponent(adminKey));

            src.onmessage = function(ev){
                const line = document.createElement('div');
                line.textContent = ev.data;
                box.appendChild(line);
                box.scrollTop = box.scrollHeight;
                btn.textContent = '⏳ Refreshing...';
                if(ev.data === '__STREAM_DONE__'){
                    btn.disabled = false;
                    btn.textContent = '🔄 Start Refresh (Stream)';
                    try{ src.close(); }catch(e){}
                }
            }

            src.onerror = function(ev){
                btn.disabled = false;
                btn.textContent = '🔄 Start Refresh (Stream)';
                try{ src.close(); }catch(e){}
            }

            closeAdminModal();
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=docs)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    status_str = "healthy" if (MODEL is not None and DATA_DF is not None) else "degraded"

    latest_date = None
    if DATA_DF is not None and 'block_date' in DATA_DF.columns:
        latest_date = DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d')

    last_update_str = LAST_UPDATE.isoformat() if LAST_UPDATE else None

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


@app.post("/refresh", response_model=RefreshResponse, dependencies=[Depends(verify_admin_key)])
async def manual_refresh(background_tasks: BackgroundTasks):
    """
    ADMIN: Manually trigger data refresh (header-based auth)
    Runs refresh in background and returns immediately.
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    print(f"\n🔐 ADMIN: Manual refresh triggered at {datetime.utcnow().isoformat()}")
    background_tasks.add_task(refresh_data_and_predict)

    return RefreshResponse(
        success=True,
        message="Admin data refresh triggered in background. Check /predict/latest after a minute.",
        latest_date=DATA_DF['block_date'].iloc[-1].strftime('%Y-%m-%d') if DATA_DF is not None else None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/refresh/stream")
async def refresh_stream(admin_ok: bool = Depends(verify_admin_key_query)):
    """
    Stream data refresh progress using Server-Sent Events (SSE).
    Accepts `admin_key` query param for authentication (EventSource-friendly).
    """
    # Generator that yields SSE 'data:' lines
    def event_generator():
        # progress callback that yields messages
        messages = []
        def cb(msg: str):
            messages.append(msg)

        # immediate starting message
        yield "data: Starting stream...\n\n"
        # Run the refresh and stream messages
        try:
            success = refresh_data_and_predict(progress_callback=cb)
            for m in messages:
                yield "data: " + str(m) + "\n\n"
            if success:
                yield "data: Refresh completed successfully.\n\n"
            else:
                yield "data: Refresh failed (see server logs).\n\n"
        except Exception as e:
            yield "data: Exception during refresh: " + str(e) + "\n\n"
        # finally close the stream
        yield "data: __STREAM_DONE__\n\n"

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get("/predict/history", response_model=PredictionHistoryResponse)
async def prediction_history(limit: int = 30, days: int = 30):
    if MODEL is None or DATA_DF is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or data not loaded"
        )

    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        if 'block_date' in DATA_DF.columns:
            recent_df = DATA_DF[DATA_DF['block_date'] >= cutoff_date].copy()
        else:
            recent_df = DATA_DF.tail(limit).copy()

        recent_df = recent_df.tail(limit)

        X_scaled = prepare_features(recent_df)
        predictions = MODEL.predict(X_scaled)
        probabilities = MODEL.predict_proba(X_scaled)

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

        date_range = {"start": history[0].date if history else None, "end": history[-1].date if history else None}

        return PredictionHistoryResponse(success=True, total_predictions=len(history), predictions=history, date_range=date_range)

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to generate history: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    print("\n" + "="*70)
    print(" ETH PRICE PREDICTOR API ".center(70))
    print("="*70)
    print(f"\n🚀 Server: http://localhost:9696")
    print(f"📖 API Docs: http://localhost:9696/docs")
    print(f"🎯 Main Endpoint: http://localhost:9696/predict/latest")
    print(f"🔐 Admin Refresh (header): POST http://localhost:9696/refresh (with X-Admin-Key header)")
    print(f"🔐 Admin Refresh (stream): GET http://localhost:9696/refresh/stream?admin_key=YOUR_KEY")

    if not ADMIN_API_KEY:
        print("⚠️  WARNING: ADMIN_API_KEY not set - refresh endpoint disabled")

    print(f"\n💡 Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=9696)
