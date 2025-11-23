# Full FastAPI app with: DB models, sample loader, Yahoo Finance updater, 5‑minute scheduler, treemap API, pie chart API, global indices API

# ---- INSTALL FIRST ----
# pip install fastapi uvicorn sqlalchemy yfinance apscheduler pydantic pandas

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, date, timedelta, time
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
import math
import pandas as pd

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, Date, DateTime, ForeignKey, desc
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# =============================== DB Setup ===============================
DB_FILE = "nifty_realtime.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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

# =============================== New Index Models ===============================
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

# =============================== Global Indices Models ===============================
class Continent(Base):
    __tablename__ = "continents"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    countries = relationship("Country", back_populates="continent", cascade="all, delete-orphan")

class Country(Base):
    __tablename__ = "countries"
    id = Column(Integer, primary_key=True)
    name = Column(String, index=True)
    continent_id = Column(Integer, ForeignKey("continents.id", ondelete="CASCADE"))
    continent = relationship("Continent", back_populates="countries")
    indices = relationship("IndexMeta", back_populates="country", cascade="all, delete-orphan")

class IndexMeta(Base):
    __tablename__ = "global_indices"
    id = Column(Integer, primary_key=True)
    country_id = Column(Integer, ForeignKey("countries.id", ondelete="CASCADE"))
    name = Column(String, index=True)
    symbol = Column(String, index=True)          # primary yfinance symbol
    fallback_symbol = Column(String, nullable=True)  # ETF fallback if available
    last_error = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    country = relationship("Country", back_populates="indices")
    ohlc = relationship("IndexOHLC", back_populates="index_meta", cascade="all, delete-orphan")

class IndexOHLC(Base):
    __tablename__ = "global_index_ohlc"
    id = Column(Integer, primary_key=True)
    index_id = Column(Integer, ForeignKey("global_indices.id", ondelete="CASCADE"))
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float, nullable=True)

    index_meta = relationship("IndexMeta", back_populates="ohlc")

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

# =============================== Global Indices Schemas ===============================
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

class GlobalIndexNode(BaseModel):
    continent: str
    country: str
    index_name: str
    symbol: str
    last_open: Optional[float]
    last_high: Optional[float]
    last_low: Optional[float]
    last_close: Optional[float]
    last_volume: Optional[float]
    pct_change_1d: Optional[float]
    fallback_used: Optional[bool]
    last_error: Optional[str]

# =============================== FastAPI Init ===============================
app = FastAPI(title="Nifty50 Real‑Time Treemap & Global Indices API")

# Allow your React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================== NIFTY50 Ticker List ===============================
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

# =============================== Index Configuration ===============================
MAJOR_INDICES = [
    {
        "symbol": "^NSEI",  # Nifty 50
        "name": "Nifty 50",
        "yahoo_symbol": "^NSEI"
    },
    {
        "symbol": "^BSESN",  # Sensex
        "name": "BSE Sensex", 
        "yahoo_symbol": "^BSESN"
    },
    {
        "symbol": "NIFTY_BANK.NS",  # Bank Nifty
        "name": "Nifty Bank",
        "yahoo_symbol": "^NSEBANK"
    },
    {
        "symbol": "^NDX",  # NASDAQ
        "name": "NASDAQ 100",
        "yahoo_symbol": "^NDX"
    },
    {
        "symbol": "^GSPC",  # S&P 500
        "name": "S&P 500",
        "yahoo_symbol": "^GSPC"
    },
    {
        "symbol": "^DJI",  # Dow Jones
        "name": "Dow Jones Industrial Average",
        "yahoo_symbol": "^DJI"
    },
    {
        "symbol": "^FTSE",  # FTSE
        "name": "FTSE 100",
        "yahoo_symbol": "^FTSE"
    },
    {
        "symbol": "^N225",  # Nikkei
        "name": "Nikkei 225",
        "yahoo_symbol": "^N225"
    },
    {
        "symbol": "^HSI",  # Hang Seng
        "name": "Hang Seng Index",
        "yahoo_symbol": "^HSI"
    },
    {
        "symbol": "NIFTY_MIDCAP.NS",  # Nifty Midcap
        "name": "Nifty Midcap 100",
        "yahoo_symbol": "^CNXMDCP"
    }
]

# =============================== Global Indices Configuration ===============================
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
        "Mexico": {"S&P/BMV IPC": "^MXX"}
    },
    "South America": {
        "Brazil": {"Ibovespa": "^BVSP"},
        "Argentina": {"MERVAL": "^MERV"},
        "Chile": {"S&P/CLX IPSA": "^IPSA"},
        "Peru": {"S&P/BVL Peru General": "^IGRA"},
        "Colombia": {"COLCAP": "^COLCAP"}
    },
    "Europe": {
        "United Kingdom": {"FTSE 100": "^FTSE"},
        "Germany": {"DAX": "^GDAXI"},
        "France": {"CAC 40": "^FCHI"},
        "Italy": {"FTSE MIB": "FTSEMIB.MI"},
        "Spain": {"IBEX 35": "^IBEX"},
        "Switzerland": {"SMI": "^SSMI"},
        "Netherlands": {"AEX": "^AEX"},
        "Eurozone": {"Euro Stoxx 50": "^STOXX50E"},
        "Sweden": {"OMX Stockholm 30": "^OMXS30"}
    },
    "Asia": {
        "Japan": {"Nikkei 225": "^N225", "TOPIX": "^TOPX"},
        "China": {"Shanghai Composite": "000001.SS", "CSI 300": "000300.SS", "Shenzhen Component": "399001.SZ"},
        "India": {"Nifty 50": "^NSEI", "Sensex": "^BSESN"},
        "Hong Kong": {"Hang Seng": "^HSI"},
        "South Korea": {"KOSPI": "^KS11"},
        "Taiwan": {"TAIEX": "^TWII"},
        "Singapore": {"STI": "^STI"},
        "Thailand": {"SET": "^SETI"},
        "Malaysia": {"KLCI": "^KLSE"},
        "Indonesia": {"IDX Composite": "^JKSE"},
        "Saudi Arabia": {"Tadawul All Share": "TASI.SR"},
        "Israel": {"TA-125": "^TA125.TA"}
    },
    "Africa": {
        "South Africa": {"FTSE/JSE All Share": "JALSH.JO"},
        "Egypt": {"EGX 30": "^EGX30"},
        "Nigeria": {"All-Share": "NGSEINDX"},
        "Kenya": {"NSE 20": "^NSE20"}
    },
    "Oceania": {
        "Australia": {"ASX 200": "^AXJO", "All Ordinaries": "^AORD"},
        "New Zealand": {"NZX 50": "^NZ50"}
    }
}

# =============================== Fallback ETF Configuration ===============================
fallback_etfs = {
    "^IPSA": "ECH",        # Chile ETF (example)
    "^IGRA": "EPU",        # Peru ETF (example)
    "^COLCAP": "ICOL",     # Colombia ETF (example)
    "^EGX30": "EGPT",      # Egypt ETF (example)
    "NGSEINDX": None,      # Nigeria - no common ETF
    "^NSE20": None,
    "TASI.SR": None,
    "JALSH.JO": None
}

PERIOD_INTERVAL_MAP = {
    "1d": {"period": "1d", "interval": "5m"},      # 1 day: 5-minute intervals
    "5d": {"period": "5d", "interval": "1h"},      # 5 days: 1-hour intervals
    "1w": {"period": "7d", "interval": "1h"},      # 1 week: 1-hour intervals
    "1m": {"period": "1mo", "interval": "1d"},     # 1 month: daily
    "6m": {"period": "6mo", "interval": "1d"},     # 6 months: daily
    "ytd": {"period": "ytd", "interval": "1d"},    # YTD: daily
    "1y": {"period": "1y", "interval": "1d"},      # 1 year: daily
    "5y": {"period": "5y", "interval": "1wk"},     # 5 years: weekly
    "max": {"period": "max", "interval": "1mo"}    # Max: monthly
}

# =============================== Helper ===============================
from contextlib import contextmanager
@contextmanager
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================== Market Hours Check ===============================
def is_market_hours():
    """Check if current time is within Indian stock market hours (9:15 AM to 3:30 PM IST)"""
    now = datetime.now()
    current_time = now.time()
    
    # Market hours: 9:15 AM to 3:30 PM
    market_start = time(9, 15)   # 9:15 AM
    market_end = time(15, 30)    # 3:30 PM
    
    # Check if current time is within market hours
    return market_start <= current_time <= market_end

def is_weekday():
    """Check if today is a weekday (Monday to Friday)"""
    return datetime.now().weekday() < 5  # 0=Monday, 4=Friday

def should_update():
    """Check if we should update data based on market hours and weekday"""
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

# =============================== Global Indices DB Loader ===============================
def load_indices_into_db():
    """Populate DB from continent_indices if DB is empty."""
    with db_session() as db:
        any_continents = db.query(Continent).first()
        if any_continents:
            return  # already loaded

        for cont_name, countries in continent_indices.items():
            cont = Continent(name=cont_name)
            db.add(cont)
            db.flush()  # get cont.id

            for country_name, indices in countries.items():
                c = Country(name=country_name, continent=cont)
                db.add(c)
                db.flush()
                for idx_name, sym in indices.items():
                    fallback = fallback_etfs.get(sym)  # may be None
                    im = IndexMeta(country=c, name=idx_name, symbol=sym, fallback_symbol=fallback)
                    db.add(im)
        db.commit()
        print("[Loader] Continent/country/index metadata loaded into DB.")

# =============================== Yahoo Finance Safe Updater ===============================
def safe_fetch_history(symbol: str, days: int = 2) -> pd.DataFrame:
    """
    Fetch up to `days` daily bars for the given yfinance symbol.
    Returns a DataFrame (could be empty).
    """
    try:
        tk = yf.Ticker(symbol)
        # Use history with period '2d' to get last + prev if available
        df = tk.history(period=f"{days}d", interval="1d", auto_adjust=False)
        # normalize index to date
        if not df.empty:
            df = df.reset_index()
            df['date'] = df['Date'].dt.date
            df = df.set_index('date')[['Open','High','Low','Close','Volume']]
        return df
    except Exception as e:
        # return empty DataFrame on error
        return pd.DataFrame()

def upsert_index_ohlc(db, index_row: IndexMeta, df: pd.DataFrame, fallback_used: bool):
    """
    Insert OHLC rows from df into DB for given index_row.
    """
    for dt, row in df.iterrows():
        # dt is a datetime.date
        exists = db.query(IndexOHLC).filter(IndexOHLC.index_id == index_row.id, IndexOHLC.date == dt).first()
        if exists:
            continue
        rec = IndexOHLC(
            index_id=index_row.id,
            date=dt,
            open=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=float(row.get('Volume') or 0)
        )
        db.add(rec)
    # update last_error to None if we succeeded
    index_row.last_error = None
    db.commit()

def update_single_global_index(db, index_row: IndexMeta):
    """
    Fetch last 2 days for index_row.symbol. If missing, try fallback_symbol.
    """
    sym = index_row.symbol
    # Try primary symbol
    df = safe_fetch_history(sym, days=2)

    fallback_used = False
    if df.empty and index_row.fallback_symbol:
        # try fallback ETF if available
        df = safe_fetch_history(index_row.fallback_symbol, days=2)
        if not df.empty:
            fallback_used = True

    # No data at all
    if df.empty:
        index_row.last_error = f"No data for {sym} (fallback used: {fallback_used})"
        db.commit()
        return {"ok": False, "error": index_row.last_error}

    # Insert available rows
    upsert_index_ohlc(db, index_row, df, fallback_used)
    index_row.last_error = None
    db.commit()
    return {"ok": True, "fetched_rows": len(df), "fallback_used": fallback_used}

def update_all_global_indices():
    """Main updater run over all global indices in DB."""
    with db_session() as db:
        print(f"[Global Updater] Running update at {datetime.utcnow().isoformat()}")
        indices = db.query(IndexMeta).all()
        for im in indices:
            try:
                res = update_single_global_index(db, im)
                if res.get("ok"):
                    print(f"Updated {im.name} ({im.symbol}) - rows={res.get('fetched_rows')} fallback={res.get('fallback_used')}")
                else:
                    print(f"Failed {im.name} ({im.symbol}): {res.get('error')}")
            except Exception as e:
                im.last_error = str(e)
                db.commit()
                print(f"Exception updating {im.symbol}: {e}")
        print("[Global Updater] Done.")

# =============================== Yahoo Finance Updater ===============================
def update_ticker(db, yf_symbol: str):
    yf_tk = yf.Ticker(yf_symbol)
    info = yf_tk.info

    base_ticker = yf_symbol.replace(".NS", "")
    company = db.query(Company).filter(Company.ticker == base_ticker).first()

    if not company:
        # create entry if missing
        company = Company(
            ticker=base_ticker,
            name=info.get("longName", base_ticker),
            sector=info.get("sector", "Unknown"),
            is_nifty=True
        )
        db.add(company)
        db.commit(); db.refresh(company)

    # Update market cap
    mc = info.get("marketCap")
    if mc:
        company.market_cap = float(mc)

    # Fetch 5 days OHLC
    hist = yf_tk.history(period="5d")
    for idx, row in hist.iterrows():
        d = idx.date()
        exists = db.query(OHLC).filter(OHLC.company_id == company.id, OHLC.date == d).first()
        if exists:
            continue
        o = OHLC(
            company_id=company.id,
            date=d,
            open=row["Open"], high=row["High"], low=row["Low"], close=row["Close"], volume=row["Volume"]
        )
        db.add(o)
    db.commit()

def update_all_tickers():
    """Update all tickers only during market hours on weekdays"""
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

# =============================== Index Updater ===============================
def update_index_data(db, index_config: dict):
    """Update individual index data from Yahoo Finance"""
    try:
        yf_tk = yf.Ticker(index_config["yahoo_symbol"])
        info = yf_tk.info
        
        # Get current price data
        hist = yf_tk.history(period="2d")
        
        if hist.empty:
            print(f"No data for {index_config['symbol']}")
            return
        
        # Get latest and previous close
        current_close = hist['Close'].iloc[-1]
        previous_close = hist['Close'].iloc[0] if len(hist) > 1 else current_close
        
        # Calculate percentage change
        pct_change = ((current_close - previous_close) / previous_close) * 100 if previous_close != 0 else 0
        
        # For indices, we'll use the index value as a proxy for "market cap" in the pie chart
        market_cap = info.get('marketCap')
        
        # If market cap not available, use index value with a multiplier for visualization
        if market_cap is None:
            # Use a scaled value for better visualization in pie chart
            market_cap = current_close * 1000  # Arbitrary multiplier for visualization
        
        # Find or create index
        index = db.query(Index).filter(Index.symbol == index_config["symbol"]).first()
        if not index:
            index = Index(
                symbol=index_config["symbol"],
                name=index_config["name"]
            )
            db.add(index)
            db.commit()
            db.refresh(index)
        
        # Update index data
        index.market_cap = float(market_cap)
        index.current_price = float(current_close)
        index.previous_close = float(previous_close)
        index.pct_change = float(pct_change)
        index.last_updated = datetime.utcnow()
        
        # Save historical data
        today = date.today()
        existing_history = db.query(IndexHistory).filter(
            IndexHistory.index_id == index.id,
            IndexHistory.date == today
        ).first()
        
        if not existing_history:
            history = IndexHistory(
                index_id=index.id,
                date=today,
                close=current_close,
                volume=hist['Volume'].iloc[-1] if 'Volume' in hist.columns and not math.isnan(hist['Volume'].iloc[-1]) else None
            )
            db.add(history)
        
        db.commit()
        print(f"Updated index: {index_config['name']}")
        
    except Exception as e:
        print(f"Error updating index {index_config['symbol']}: {str(e)}")

def update_all_indices():
    """Update all major indices data"""
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

@app.on_event("startup")
def start_background_jobs():
    # Load global indices metadata if missing
    load_indices_into_db()

    # Schedule company updates every 5 minutes
    scheduler.add_job(update_all_tickers, "interval", minutes=5)
    
    # Schedule indices updates every 5 minutes
    scheduler.add_job(update_all_indices, "interval", minutes=5)
    
    # Schedule global indices updates every 5 minutes
    scheduler.add_job(update_all_global_indices, "interval", minutes=5)
    
    scheduler.start()
    print("Scheduler started. Will update companies and indices every 5 minutes during market hours (9:15 AM - 3:30 PM IST, Mon-Fri).")

    # Run once at startup regardless of market hours to ensure initial data
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
        
        print("Running initial global indices data load...")
        update_all_global_indices()

# =============================== Treemap API ===============================
@app.get("/api/treemap/nifty50", response_model=List[TreemapNode])
def treemap_nifty50():
    with db_session() as db:
        companies = db.query(Company).filter(Company.is_nifty == True).all()
        if not companies:
            return []

        total_market_cap = sum([
            c.market_cap for c in companies if c.market_cap is not None
        ])

        nodes = []
        for c in companies:
            # fetch latest 2 OHLC rows
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

        # sort largest first
        return sorted(nodes, key=lambda x: (x.market_cap or 0), reverse=True)

# =============================== Pie Chart API ===============================
@app.get("/api/piechart/indices", response_model=List[IndexPieNode])
def piechart_indices():
    """Get major indices data for pie chart visualization"""
    with db_session() as db:
        indices = db.query(Index).all()
        
        if not indices:
            # If no data, return empty list
            return []
        
        # Calculate total market cap for weight calculation
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
        
        # Sort by market cap (largest first)
        return sorted(nodes, key=lambda x: (x.market_cap or 0), reverse=True)

@app.get("/api/indices/details")
def get_indices_detailed():
    """Get detailed information for all indices"""
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

# =============================== Global Indices API ===============================
@app.get("/api/global/index-list", response_model=List[IndexInfo])
def api_index_list():
    """Get list of all global indices organized by continent and country"""
    return flatten_index_dict()

@app.get("/api/global/indices", response_model=List[GlobalIndexNode])
def list_global_indices(include_errors: bool = False):
    """
    Returns all global indices with latest OHLC and 1d % change.
    include_errors: include indices that have last_error set (default False -> drop failed)
    """
    with db_session() as db:
        nodes = []
        # join continents -> countries -> indices
        continents = db.query(Continent).all()
        for cont in continents:
            for country in cont.countries:
                for im in country.indices:
                    # skip errors unless requested
                    if (im.last_error is not None) and (not include_errors):
                        continue

                    # get latest available rows (limit 2)
                    ohlcs = db.query(IndexOHLC).filter(IndexOHLC.index_id == im.id).order_by(IndexOHLC.date.desc()).limit(2).all()
                    last = ohlcs[0] if len(ohlcs) >= 1 else None
                    prev = ohlcs[1] if len(ohlcs) >= 2 else None

                    pct = None
                    if last and prev and prev.close and prev.close != 0:
                        pct = ((last.close - prev.close) / prev.close) * 100

                    node = GlobalIndexNode(
                        continent=cont.name,
                        country=country.name,
                        index_name=im.name,
                        symbol=im.symbol if not (im.fallback_symbol and (im.last_error and im.last_error.startswith("No data"))) else (im.fallback_symbol or im.symbol),
                        last_open=last.open if last else None,
                        last_high=last.high if last else None,
                        last_low=last.low if last else None,
                        last_close=last.close if last else None,
                        last_volume=last.volume if last else None,
                        pct_change_1d=pct,
                        fallback_used=False if not im.fallback_symbol else None,  # unknown here
                        last_error=im.last_error
                    )
                    nodes.append(node)
        # sort by continent/country/name
        nodes_sorted = sorted(nodes, key=lambda x: (x.continent, x.country, x.index_name))
        return nodes_sorted

@app.get("/api/global/treemap/indices")
def global_treemap_indices(include_errors: bool = False):
    """
    Return a treemap-friendly JSON where weight is derived from last_close relative to sum of last_close.
    """
    nodes = list_global_indices(include_errors=include_errors)
    # compute total weight using last_close (skip None)
    total = sum([n.last_close for n in nodes if n.last_close is not None])
    payload = []
    for n in nodes:
        weight = (n.last_close / total) if (n.last_close is not None and total > 0) else None
        payload.append({
            "continent": n.continent,
            "country": n.country,
            "index_name": n.index_name,
            "symbol": n.symbol,
            "last_close": n.last_close,
            "pct_change_1d": n.pct_change_1d,
            "weight": weight,
            "last_error": n.last_error
        })
    # sort by weight descending (None last)
    payload_sorted = sorted(payload, key=lambda x: (x["weight"] is None, -(x["weight"] or 0)))
    return payload_sorted

@app.get("/api/global/index/{symbol}")
def get_global_index_by_symbol(symbol: str):
    """Get a single global index's latest OHLC and % change (symbol matches primary symbol)."""
    with db_session() as db:
        im = db.query(IndexMeta).filter(IndexMeta.symbol == symbol).first()
        if not im:
            raise HTTPException(status_code=404, detail="Symbol not found")

        ohlcs = db.query(IndexOHLC).filter(IndexOHLC.index_id == im.id).order_by(IndexOHLC.date.desc()).limit(2).all()
        last = ohlcs[0] if len(ohlcs) >= 1 else None
        prev = ohlcs[1] if len(ohlcs) >= 2 else None
        pct = None
        if last and prev and prev.close and prev.close != 0:
            pct = ((last.close - prev.close) / prev.close) * 100

        return {
            "symbol": im.symbol,
            "index_name": im.name,
            "country_id": im.country_id,
            "last_close": last.close if last else None,
            "last_open": last.open if last else None,
            "last_high": last.high if last else None,
            "last_low": last.low if last else None,
            "last_volume": last.volume if last else None,
            "pct_change_1d": pct,
            "last_error": im.last_error
        }

@app.get("/api/global/indices/history")
def api_all_indices_history(period: str = "1m"):
    """
    Fetch historical data for all global indices at once.
    Returns a dictionary with symbol as key and OHLC data as value.
    """
    if period not in PERIOD_INTERVAL_MAP:
        raise HTTPException(status_code=400, detail="Invalid period")

    config = PERIOD_INTERVAL_MAP[period]
    yf_period = config["period"]
    yf_interval = config["interval"]

    # Collect all symbols
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

    # Helper to safely convert to float or None
    def safe_float(val):
        try:
            return float(val) if pd.notna(val) else None
        except (ValueError, TypeError):
            return None

    # Fetch all data at once using yfinance
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

    # Process results
    result = {}
    
    for symbol in all_symbols:
        try:
            # Get data for this symbol
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
            
            # Reset index to get date as column
            symbol_data = symbol_data.reset_index()
            
            # Handle both 'Date' and 'Datetime' column names
            date_col = 'Date' if 'Date' in symbol_data.columns else 'Datetime'
            
            # Remove rows with missing dates
            symbol_data = symbol_data.dropna(subset=[date_col])
            
            ohlc_data = []
            for _, r in symbol_data.iterrows():
                # Format datetime based on interval
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

@app.get("/api/global/index/{symbol}/current")
def get_index_current_price(symbol: str):
    """Get current price and basic info for a specific index"""
    try:
        yf_tk = yf.Ticker(symbol)
        info = yf_tk.info
        
        # Get latest price data
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

# =============================== Global Indices Debug Routes ===============================
@app.get("/api/global/failed")
def list_failed_global_indices():
    with db_session() as db:
        failed = db.query(IndexMeta).filter(IndexMeta.last_error != None).all()
        return [{"symbol": f.symbol, "index": f.name, "country_id": f.country_id, "error": f.last_error} for f in failed]

@app.post("/api/global/update")
def manual_global_update():
    try:
        update_all_global_indices()
        return {"status": "ok", "updated_at": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============================== Debug Routes ===============================
@app.get("/api/companies")
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

@app.get("/api/market-status")
def get_market_status():
    """Check current market status"""
    return {
        "current_time": datetime.now().isoformat(),
        "is_weekday": is_weekday(),
        "is_market_hours": is_market_hours(),
        "should_update": should_update()
    }

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Nifty50 Real-Time & Global Indices API",
        "endpoints": {
            "nifty_treemap": "/api/treemap/nifty50",
            "indices_piechart": "/api/piechart/indices", 
            "indices_details": "/api/indices/details",
            "global_indices_list": "/api/global/index-list",
            "global_indices": "/api/global/indices",
            "global_treemap": "/api/global/treemap/indices",
            "global_index_detail": "/api/global/index/{symbol}",
            "global_all_indices_history": "/api/global/indices/history?period=1d|5d|1w|1m|6m|ytd|1y|5y|max",
            "global_index_current": "/api/global/index/{symbol}/current",
            "global_failed": "/api/global/failed",
            "global_manual_update": "/api/global/update (POST)",
            "companies": "/api/companies",
            "market_status": "/api/market-status"
        },
        "description": "Comprehensive API for Nifty50 analysis and global indices data"
    }