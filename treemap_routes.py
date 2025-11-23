"""
Treemap and Global Indices Module
Provides routes and background jobs for Nifty50 treemap and global indices data
"""

from datetime import datetime, date, timedelta, time
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from contextlib import contextmanager
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
import math
import pandas as pd
from fastapi import HTTPException

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Date, DateTime, ForeignKey, desc
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# =============================== DB Setup ===============================
# Use the same PostgreSQL database as app.py
DATABASE_URL = "postgresql://postgres:ZpfLDFFOLJemAIEkOTBpEjCuBWYyIwSm@switchback.proxy.rlwy.net:19114/railway"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# =============================== Models ===============================
class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    ticker = Column(String, unique=True, index=True)
    name = Column(String)
    sector = Column(String)
    market_cap = Column(Float, nullable=True)
    is_nifty = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    ohlc = relationship("OHLC", back_populates="company", cascade="all, delete-orphan")

class OHLC(Base):
    __tablename__ = "ohlc"
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, ForeignKey("companies.id", ondelete="CASCADE"))
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float, nullable=True)
    company = relationship("Company", back_populates="ohlc")

class Index(Base):
    __tablename__ = "indices"
    id = Column(Integer, primary_key=True)
    symbol = Column(String, unique=True, index=True)
    name = Column(String)
    market_cap = Column(Float, nullable=True)
    current_price = Column(Float, nullable=True)
    previous_close = Column(Float, nullable=True)
    pct_change = Column(Float, nullable=True)
    last_updated = Column(DateTime, default=datetime.utcnow)

class IndexHistory(Base):
    __tablename__ = "index_history"
    id = Column(Integer, primary_key=True)
    index_id = Column(Integer, ForeignKey("indices.id", ondelete="CASCADE"))
    date = Column(Date, index=True)
    close = Column(Float)
    volume = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

# =============================== Pydantic Schemas ===============================
class TreemapNode(BaseModel):
    ticker: str
    name: str
    sector: Optional[str]
    market_cap: Optional[float]
    weight: Optional[float]
    last_close: Optional[float]
    pct_change_1d: Optional[float]

class IndexPieNode(BaseModel):
    symbol: str
    name: str
    market_cap: Optional[float]
    current_price: Optional[float]
    pct_change: Optional[float]
    weight: Optional[float]

class IndexInfo(BaseModel):
    continent: str
    country: str
    index_name: str
    symbol: str

class OHLCRow(BaseModel):
    date: str
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: Optional[float]
    volume: Optional[float]

# =============================== Constants ===============================
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS",
    "KOTAKBANK.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "HCLTECH.NS", "WIPRO.NS",
    "POWERGRID.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS", "NTPC.NS",
    "TECHM.NS", "ONGC.NS", "NESTLEIND.NS", "JSWSTEEL.NS", "COALINDIA.NS",
    "BAJAJ-AUTO.NS", "GRASIM.NS", "HDFCLIFE.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "BRITANNIA.NS", "SHREECEM.NS", "CIPLA.NS", "EICHERMOT.NS", "HEROMOTOCO.NS",
    "HINDALCO.NS", "INDUSINDBK.NS", "TATACONSUM.NS", "TATAPOWER.NS", "M&M.NS",
    "APOLLOHOSP.NS", "BAJAJFINSV.NS", "BPCL.NS", "SBILIFE.NS", "UPL.NS"
]

MAJOR_INDICES = [
    {"symbol": "^NSEI", "name": "Nifty 50", "yahoo_symbol": "^NSEI"},
    {"symbol": "^BSESN", "name": "BSE Sensex", "yahoo_symbol": "^BSESN"},
    {"symbol": "NIFTY_BANK.NS", "name": "Nifty Bank", "yahoo_symbol": "^NSEBANK"},
    {"symbol": "^NDX", "name": "NASDAQ 100", "yahoo_symbol": "^NDX"},
    {"symbol": "^GSPC", "name": "S&P 500", "yahoo_symbol": "^GSPC"},
    {"symbol": "^DJI", "name": "Dow Jones Industrial Average", "yahoo_symbol": "^DJI"},
    {"symbol": "^FTSE", "name": "FTSE 100", "yahoo_symbol": "^FTSE"},
    {"symbol": "^N225", "name": "Nikkei 225", "yahoo_symbol": "^N225"},
    {"symbol": "^HSI", "name": "Hang Seng Index", "yahoo_symbol": "^HSI"},
    {"symbol": "NIFTY_MIDCAP.NS", "name": "Nifty Midcap 100", "yahoo_symbol": "^CNXMDCP"}
]

continent_indices = {
    "North America": {
        "USA": {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ Composite": "^IXIC",
            "Russell 2000": "^RUT",
            "NYSE Composite": "^NYA"
        },
        "Canada": {"S&P/TSX Composite": "^GSPTSE"},
        "Mexico": {"IPC": "^MXX"}
    },
    "South America": {
        "Brazil": {"Ibovespa": "^BVSP"},
        "Argentina": {"MERVAL": "^MERV"},
    },
    "Europe": {
        "UK": {"FTSE 100": "^FTSE"},
        "Germany": {"DAX": "^GDAXI"},
        "France": {"CAC 40": "^FCHI"},
        "Italy": {"FTSE MIB": "FTSEMIB.MI"},
        "Spain": {"IBEX 35": "^IBEX"},
        "Netherlands": {"AEX": "^AEX"},
        "Switzerland": {"SMI": "^SSMI"},
        "Eurozone": {"Euro Stoxx 50": "^STOXX50E"}
    },
    "Asia": {
        "Japan": {"Nikkei 225": "^N225"},
        "China": {"Shanghai Composite": "000001.SS"},
        "India": {"Nifty 50": "^NSEI", "Sensex": "^BSESN"},
        "Hong Kong": {"Hang Seng": "^HSI"},
        "South Korea": {"KOSPI": "^KS11"},
        "Taiwan": {"TAIEX": "^TWII"},
        "Singapore": {"STI": "^STI"},
        "Malaysia": {"KLCI": "^KLSE"},
        "Indonesia": {"IDX Composite": "^JKSE"},
    },
    "Oceania": {
        "Australia": {"ASX 200": "^AXJO"},
        "New Zealand": {"NZX 50": "^NZ50"}
    }
}

PERIOD_INTERVAL_MAP = {
    "1d": {"period": "1d", "interval": "5m"},
    "5d": {"period": "5d", "interval": "1h"},
    "1w": {"period": "7d", "interval": "1h"},
    "1m": {"period": "1mo", "interval": "1d"},
    "6m": {"period": "6mo", "interval": "1d"},
    "ytd": {"period": "ytd", "interval": "1d"},
    "1y": {"period": "1y", "interval": "1d"},
    "5y": {"period": "5y", "interval": "1wk"},
    "max": {"period": "max", "interval": "1mo"}
}

# =============================== Helper ===============================
@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================== Market Hours Check ===============================
def is_market_hours():
    now = datetime.now()
    current_time = now.time()
    market_start = time(9, 15)
    market_end = time(15, 30)
    return market_start <= current_time <= market_end

def is_weekday():
    return datetime.now().weekday() < 5

def should_update():
    return is_weekday() and is_market_hours()

def flatten_index_dict():
    items = []
    for cont, countries in continent_indices.items():
        for country, indices in countries.items():
            for idx_name, symbol in indices.items():
                items.append(IndexInfo(
                    continent=cont,
                    country=country,
                    index_name=idx_name,
                    symbol=symbol
                ))
    return items

# =============================== Yahoo Finance Updater ===============================
def update_ticker(db, yf_symbol: str):
    yf_tk = yf.Ticker(yf_symbol)
    info = yf_tk.info
    base_ticker = yf_symbol.replace(".NS", "")
    company = db.query(Company).filter(Company.ticker == base_ticker).first()

    if not company:
        company = Company(
            ticker=base_ticker,
            name=info.get("longName", base_ticker),
            sector=info.get("sector", "Unknown"),
            is_nifty=True
        )
        db.add(company)
        db.commit()
        db.refresh(company)

    mc = info.get("marketCap")
    if mc:
        company.market_cap = float(mc)

    hist = yf_tk.history(period="5d")
    for idx, row in hist.iterrows():
        d = idx.date()
        exists = db.query(OHLC).filter(OHLC.company_id == company.id, OHLC.date == d).first()
        if exists:
            continue
        
        # Convert numpy types to Python native types
        o = OHLC(
            company_id=company.id,
            date=d,
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]) if not math.isnan(row["Volume"]) else None
        )
        db.add(o)
    db.commit()

def update_all_tickers():
    if not should_update():
        print(f"[Updater] Skipping update - outside market hours or weekend. Current time: {datetime.now()}")
        return
    
    with db_session() as db:
        print(f"[Updater] Running at {datetime.now()} (Market Hours)")
        for sym in NIFTY50_TICKERS:
            try:
                update_ticker(db, sym)
                print("Updated", sym)
            except Exception as e:
                print("Error updating", sym, e)
        print("==== Update complete ====")

def update_index_data(db, index_config: dict):
    try:
        yf_tk = yf.Ticker(index_config["yahoo_symbol"])
        info = yf_tk.info
        hist = yf_tk.history(period="2d")
        
        if hist.empty:
            print(f"No data for {index_config['symbol']}")
            return
        
        # Convert numpy types to Python native types
        current_close = float(hist['Close'].iloc[-1])
        previous_close = float(hist['Close'].iloc[0]) if len(hist) > 1 else current_close
        pct_change = ((current_close - previous_close) / previous_close) * 100 if previous_close != 0 else 0
        market_cap = info.get('marketCap')
        
        if market_cap is None:
            market_cap = current_close * 1000
        
        index = db.query(Index).filter(Index.symbol == index_config["symbol"]).first()
        if not index:
            index = Index(symbol=index_config["symbol"], name=index_config["name"])
            db.add(index)
            db.commit()
            db.refresh(index)
        
        index.market_cap = float(market_cap)
        index.current_price = float(current_close)
        index.previous_close = float(previous_close)
        index.pct_change = float(pct_change)
        index.last_updated = datetime.utcnow()
        
        today = date.today()
        existing_history = db.query(IndexHistory).filter(
            IndexHistory.index_id == index.id,
            IndexHistory.date == today
        ).first()
        
        if not existing_history:
            # Convert volume to Python int, handling NaN
            volume_val = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else None
            if volume_val is not None and not math.isnan(volume_val):
                volume_val = int(volume_val)
            else:
                volume_val = None
            
            history = IndexHistory(
                index_id=index.id,
                date=today,
                close=float(current_close),
                volume=volume_val
            )
            db.add(history)
        
        db.commit()
        print(f"Updated index: {index_config['name']}")
        
    except Exception as e:
        db.rollback()  # Rollback on error
        print(f"Error updating index {index_config['symbol']}: {str(e)}")

def update_all_indices():
    if not should_update():
        print(f"[Index Updater] Skipping update - outside market hours or weekend. Current time: {datetime.now()}")
        return
    
    with db_session() as db:
        print(f"[Index Updater] Running at {datetime.now()} (Market Hours)")
        for index_config in MAJOR_INDICES:
            try:
                update_index_data(db, index_config)
            except Exception as e:
                print(f"Failed to update {index_config['name']}: {str(e)}")
        print("==== Index update complete ====")

# =============================== Scheduler ===============================
scheduler = BackgroundScheduler()

def start_scheduler():
    scheduler.add_job(update_all_tickers, "interval", minutes=5)
    scheduler.add_job(update_all_indices, "interval", minutes=5)
    scheduler.start()
    print("Scheduler started. Will update companies and indices every 5 minutes during market hours.")

    # Initial load
    with db_session() as db:
        print("Running initial company data load...")
        for sym in NIFTY50_TICKERS:
            try:
                update_ticker(db, sym)
                print("Initial load:", sym)
            except Exception as e:
                print("Error in initial load", sym, e)
        
        print("Running initial indices data load...")
        for index_config in MAJOR_INDICES:
            try:
                update_index_data(db, index_config)
                print("Initial index load:", index_config["name"])
            except Exception as e:
                print("Error in initial index load", index_config["name"], e)

# =============================== Route Handlers ===============================
def treemap_nifty50():
    with db_session() as db:
        companies = db.query(Company).filter(Company.is_nifty == True).all()
        if not companies:
            return []

        total_market_cap = sum([c.market_cap for c in companies if c.market_cap is not None])
        nodes = []
        for c in companies:
            ohlc = db.query(OHLC).filter(OHLC.company_id == c.id).order_by(OHLC.date.desc()).limit(2).all()
            last_close = ohlc[0].close if len(ohlc) > 0 else None
            pct_change = None
            if len(ohlc) == 2 and ohlc[1].close > 0:
                pct_change = ((ohlc[0].close - ohlc[1].close) / ohlc[1].close) * 100

            node = TreemapNode(
                ticker=c.ticker,
                name=c.name,
                sector=c.sector,
                market_cap=c.market_cap,
                weight=(c.market_cap / total_market_cap) if c.market_cap else None,
                last_close=last_close,
                pct_change_1d=pct_change
            )
            nodes.append(node)

        return sorted(nodes, key=lambda x: (x.market_cap or 0), reverse=True)

def piechart_indices():
    with db_session() as db:
        indices = db.query(Index).all()
        if not indices:
            return []
        
        total_market_cap = sum([idx.market_cap for idx in indices if idx.market_cap is not None])
        nodes = []
        for idx in indices:
            node = IndexPieNode(
                symbol=idx.symbol,
                name=idx.name,
                market_cap=idx.market_cap,
                current_price=idx.current_price,
                pct_change=idx.pct_change,
                weight=(idx.market_cap / total_market_cap) if idx.market_cap and total_market_cap > 0 else None
            )
            nodes.append(node)
        
        return sorted(nodes, key=lambda x: (x.market_cap or 0), reverse=True)

def get_indices_detailed():
    with db_session() as db:
        indices = db.query(Index).all()
        result = []
        for idx in indices:
            result.append({
                "symbol": idx.symbol,
                "name": idx.name,
                "market_cap": idx.market_cap,
                "current_price": idx.current_price,
                "previous_close": idx.previous_close,
                "pct_change": idx.pct_change,
                "last_updated": idx.last_updated
            })
        return result

def api_index_list():
    return flatten_index_dict()

def api_all_indices_history(period: str = "1m"):
    if period not in PERIOD_INTERVAL_MAP:
        raise HTTPException(status_code=400, detail="Invalid period")

    config = PERIOD_INTERVAL_MAP[period]
    yf_period = config["period"]
    yf_interval = config["interval"]

    all_symbols = []
    symbol_to_info = {}
    
    for continent, countries in continent_indices.items():
        for country, indices in countries.items():
            for index_name, symbol in indices.items():
                all_symbols.append(symbol)
                symbol_to_info[symbol] = {
                    "continent": continent,
                    "country": country,
                    "index_name": index_name
                }

    def safe_float(val):
        try:
            return float(val) if pd.notna(val) else None
        except (ValueError, TypeError):
            return None

    try:
        hist = yf.download(
            all_symbols,
            period=yf_period,
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
            group_by='ticker'
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {str(e)}")

    if hist.empty:
        raise HTTPException(status_code=404, detail="No data available")

    result = {}
    
    for symbol in all_symbols:
        try:
            if len(all_symbols) == 1:
                symbol_data = hist
            else:
                symbol_data = hist[symbol]
            
            if symbol_data.empty:
                result[symbol] = {
                    "info": symbol_to_info[symbol],
                    "data": [],
                    "error": "No data available"
                }
                continue
            
            symbol_data = symbol_data.reset_index()
            date_col = 'Date' if 'Date' in symbol_data.columns else 'Datetime'
            symbol_data = symbol_data.dropna(subset=[date_col])
            
            ohlc_data = []
            for _, r in symbol_data.iterrows():
                if yf_interval in ["5m", "15m", "30m", "1h"]:
                    date_str = r[date_col].strftime("%Y-%m-%d %H:%M:%S")
                else:
                    date_str = r[date_col].strftime("%Y-%m-%d")
                
                ohlc_data.append({
                    "date": date_str,
                    "open": safe_float(r["Open"]),
                    "high": safe_float(r["High"]),
                    "low": safe_float(r["Low"]),
                    "close": safe_float(r["Close"]),
                    "volume": safe_float(r["Volume"])
                })
            
            result[symbol] = {
                "info": symbol_to_info[symbol],
                "data": ohlc_data
            }
            
        except Exception as e:
            result[symbol] = {
                "info": symbol_to_info[symbol],
                "data": [],
                "error": str(e)
            }
    
    return result

def get_index_current_price(symbol: str):
    try:
        yf_tk = yf.Ticker(symbol)
        info = yf_tk.info
        hist = yf_tk.history(period="2d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data available for this symbol")
        
        current_price = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[0] if len(hist) > 1 else current_price
        pct_change = ((current_price - previous_close) / previous_close) * 100 if previous_close != 0 else 0
        
        return {
            "symbol": symbol,
            "current_price": float(current_price),
            "previous_close": float(previous_close),
            "pct_change": float(pct_change),
            "name": info.get('shortName', symbol),
            "currency": info.get('currency', 'USD'),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to fetch data: {str(e)}")

def list_companies():
    with db_session() as db:
        return [
            {
                "ticker": c.ticker,
                "name": c.name,
                "sector": c.sector,
                "market_cap": c.market_cap
            }
            for c in db.query(Company).all()
        ]

def get_market_status():
    return {
        "current_time": datetime.now().isoformat(),
        "is_weekday": is_weekday(),
        "is_market_hours": is_market_hours(),
        "should_update": should_update()
    }
