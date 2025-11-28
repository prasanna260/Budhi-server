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
#DATABASE_URL = "postgresql://postgres:ZpfLDFFOLJemAIEkOTBpEjCuBWYyIwSm@switchback.proxy.rlwy.net:19114/railway"
DB_FILE = "nifty_realtime.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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
NIFTY500_TICKERS = [
    "360ONE.NS", "3MINDIA.NS", "AADHARHFC.NS", "AARTIIND.NS", "AAVAS.NS", "ABB.NS", "ABBOTINDIA.NS", "ABCAPITAL.NS", "ABFRL.NS", "ABLBL.NS",
    "ABREL.NS", "ABSLAMC.NS", "ACC.NS", "ACE.NS", "ACMESOLAR.NS", "ADANIENSOL.NS", "ADANIENT.NS", "ADANIGREEN.NS", "ADANIPORTS.NS", "ADANIPOWER.NS",
    "ADVENTHTL.NS", "AEGISLOG.NS", "AEGISVOPAK.NS", "AFCONS.NS", "AFFLE.NS", "AGARWALEYE.NS", "AIAENG.NS", "AIIL.NS", "AJANTPHARM.NS", "AKUMS.NS",
    "AKZOINDIA.NS", "ALKEM.NS", "ALKYLAMINE.NS", "ALOKINDS.NS", "AMBER.NS", "AMBUJACEM.NS", "ANANDRATHI.NS", "ANANTRAJ.NS", "ANGELONE.NS", "APARINDS.NS",
    "APLAPOLLO.NS", "APLLTD.NS", "APOLLOHOSP.NS", "APOLLOTYRE.NS", "APTUS.NS", "ARE&M.NS", "ASAHIINDIA.NS", "ASHOKLEY.NS", "ASIANPAINT.NS", "ASTERDM.NS",
    "ASTRAL.NS", "ASTRAZEN.NS", "ATGL.NS", "ATHERENERG.NS", "ATUL.NS", "AUBANK.NS", "AUROPHARMA.NS", "AWL.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS",
    "BAJAJFINSV.NS", "BAJAJHFL.NS", "BAJAJHLDNG.NS", "BAJFINANCE.NS", "BALKRISIND.NS", "BALRAMCHIN.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BANKINDIA.NS", "BASF.NS",
    "BATAINDIA.NS", "BAYERCROP.NS", "BBTC.NS", "BDL.NS", "BEL.NS", "BEML.NS", "BERGEPAINT.NS", "BHARATFORG.NS", "BHARTIARTL.NS", "BHARTIHEXA.NS",
    "BHEL.NS", "BIKAJI.NS", "BIOCON.NS", "BLS.NS", "BLUEDART.NS", "BLUEJET.NS", "BLUESTARCO.NS", "BOSCHLTD.NS", "BPCL.NS", "BRIGADE.NS",
    "BRITANNIA.NS", "BSE.NS", "BSOFT.NS", "CAMPUS.NS", "CAMS.NS", "CANBK.NS", "CANFINHOME.NS", "CAPLIPOINT.NS", "CARBORUNIV.NS", "CASTROLIND.NS",
    "CCL.NS", "CDSL.NS", "CEATLTD.NS", "CENTRALBK.NS", "CENTURYPLY.NS", "CERA.NS", "CESC.NS", "CGCL.NS", "CGPOWER.NS", "CHALET.NS",
    "CHAMBLFERT.NS", "CHENNPETRO.NS", "CHOICEIN.NS", "CHOLAFIN.NS", "CHOLAHLDNG.NS", "CIPLA.NS", "CLEAN.NS", "COALINDIA.NS", "COCHINSHIP.NS", "COFORGE.NS",
    "COHANCE.NS", "COLPAL.NS", "CONCOR.NS", "CONCORDBIO.NS", "COROMANDEL.NS", "CRAFTSMAN.NS", "CREDITACC.NS", "CRISIL.NS", "CROMPTON.NS", "CUB.NS",
    "CUMMINSIND.NS", "CYIENT.NS", "DABUR.NS", "DALBHARAT.NS", "DATAPATTNS.NS", "DBREALTY.NS", "DCMSHRIRAM.NS", "DEEPAKFERT.NS", "DEEPAKNTR.NS", "DELHIVERY.NS",
    "DEVYANI.NS", "DIVISLAB.NS", "DIXON.NS", "DLF.NS", "DMART.NS", "DOMS.NS", "DRREDDY.NS", "DUMMYSKFIN.NS", "ECLERX.NS", "EICHERMOT.NS",
    "EIDPARRY.NS", "EIHOTEL.NS", "ELECON.NS", "ELGIEQUIP.NS", "EMAMILTD.NS", "EMCURE.NS", "ENDURANCE.NS", "ENGINERSIN.NS", "ENRIN.NS", "ERIS.NS",
    "ESCORTS.NS", "ETERNAL.NS", "EXIDEIND.NS", "FACT.NS", "FEDERALBNK.NS", "FINCABLES.NS", "FINPIPE.NS", "FIRSTCRY.NS", "FIVESTAR.NS", "FLUOROCHEM.NS",
    "FORCEMOT.NS", "FORTIS.NS", "FSL.NS", "GAIL.NS", "GESHIP.NS", "GICRE.NS", "GILLETTE.NS", "GLAND.NS", "GLAXO.NS", "GLENMARK.NS",
    "GMDCLTD.NS", "GMRAIRPORT.NS", "GODFRYPHLP.NS", "GODIGIT.NS", "GODREJAGRO.NS", "GODREJCP.NS", "GODREJIND.NS", "GODREJPROP.NS", "GPIL.NS", "GRANULES.NS",
    "GRAPHITE.NS", "GRASIM.NS", "GRAVITA.NS", "GRSE.NS", "GSPL.NS", "GUJGASLTD.NS", "GVT&D.NS", "HAL.NS", "HAPPSTMNDS.NS", "HAVELLS.NS",
    "HBLENGINE.NS", "HCLTECH.NS", "HDFCAMC.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEG.NS", "HEROMOTOCO.NS", "HEXT.NS", "HFCL.NS", "HINDALCO.NS",
    "HINDCOPPER.NS", "HINDPETRO.NS", "HINDUNILVR.NS", "HINDZINC.NS", "HOMEFIRST.NS", "HONASA.NS", "HONAUT.NS", "HSCL.NS", "HUDCO.NS", "HYUNDAI.NS",
    "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDBI.NS", "IDEA.NS", "IDFCFIRSTB.NS", "IEX.NS", "IFCI.NS", "IGIL.NS", "IGL.NS",
    "IIFL.NS", "IKS.NS", "INDGN.NS", "INDHOTEL.NS", "INDIACEM.NS", "INDIAMART.NS", "INDIANB.NS", "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS",
    "INFY.NS", "INOXINDIA.NS", "INOXWIND.NS", "INTELLECT.NS", "IOB.NS", "IOC.NS", "IPCALAB.NS", "IRB.NS", "IRCON.NS", "IRCTC.NS",
    "IREDA.NS", "IRFC.NS", "ITC.NS", "ITCHOTELS.NS", "ITI.NS", "J&KBANK.NS", "JBCHEPHARM.NS", "JBMA.NS", "JINDALSAW.NS", "JINDALSTEL.NS",
    "JIOFIN.NS", "JKCEMENT.NS", "JKTYRE.NS", "JMFINANCIL.NS", "JPPOWER.NS", "JSL.NS", "JSWENERGY.NS", "JSWINFRA.NS", "JSWSTEEL.NS", "JUBLFOOD.NS",
    "JUBLINGREA.NS", "JUBLPHARMA.NS", "JWL.NS", "JYOTHYLAB.NS", "JYOTICNC.NS", "KAJARIACER.NS", "KALYANKJIL.NS", "KARURVYSYA.NS", "KAYNES.NS", "KEC.NS",
    "KEI.NS", "KFINTECH.NS", "KIMS.NS", "KIRLOSBROS.NS", "KIRLOSENG.NS", "KOTAKBANK.NS", "KPIL.NS", "KPITTECH.NS", "KPRMILL.NS", "KSB.NS",
    "LALPATHLAB.NS", "LATENTVIEW.NS", "LAURUSLABS.NS", "LEMONTREE.NS", "LICHSGFIN.NS", "LICI.NS", "LINDEINDIA.NS", "LLOYDSME.NS", "LODHA.NS", "LT.NS",
    "LTF.NS", "LTFOODS.NS", "LTIM.NS", "LTTS.NS", "LUPIN.NS", "M&M.NS", "M&MFIN.NS", "MAHABANK.NS", "MAHSCOOTER.NS", "MAHSEAMLES.NS",
    "MANAPPURAM.NS", "MANKIND.NS", "MANYAVAR.NS", "MAPMYINDIA.NS", "MARICO.NS", "MARUTI.NS", "MAXHEALTH.NS", "MAZDOCK.NS", "MCX.NS", "MEDANTA.NS",
    "METROPOLIS.NS", "MFSL.NS", "MGL.NS", "MINDACORP.NS", "MMTC.NS", "MOTHERSON.NS", "MOTILALOFS.NS", "MPHASIS.NS", "MRF.NS", "MRPL.NS",
    "MSUMI.NS", "MUTHOOTFIN.NS", "NAM-INDIA.NS", "NATCOPHARM.NS", "NATIONALUM.NS", "NAUKRI.NS", "NAVA.NS", "NAVINFLUOR.NS", "NBCC.NS", "NCC.NS",
    "NESTLEIND.NS", "NETWEB.NS", "NEULANDLAB.NS", "NEWGEN.NS", "NH.NS", "NHPC.NS", "NIACL.NS", "NIVABUPA.NS", "NLCINDIA.NS", "NMDC.NS",
    "NSLNISP.NS", "NTPC.NS", "NTPCGREEN.NS", "NUVAMA.NS", "NUVOCO.NS", "NYKAA.NS", "OBEROIRLTY.NS", "OFSS.NS", "OIL.NS", "OLAELEC.NS",
    "OLECTRA.NS", "ONESOURCE.NS", "ONGC.NS", "PAGEIND.NS", "PATANJALI.NS", "PAYTM.NS", "PCBL.NS", "PERSISTENT.NS", "PETRONET.NS", "PFC.NS",
    "PFIZER.NS", "PGEL.NS", "PGHH.NS", "PHOENIXLTD.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "PNBHOUSING.NS", "POLICYBZR.NS", "POLYCAB.NS",
    "POLYMED.NS", "POONAWALLA.NS", "POWERGRID.NS", "POWERINDIA.NS", "PPLPHARMA.NS", "PRAJIND.NS", "PREMIERENE.NS", "PRESTIGE.NS", "PTCIL.NS", "PVRINOX.NS",
    "RADICO.NS", "RAILTEL.NS", "RAINBOW.NS", "RAMCOCEM.NS", "RBLBANK.NS", "RCF.NS", "RECLTD.NS", "REDINGTON.NS", "RELIANCE.NS", "RELINFRA.NS",
    "RHIM.NS", "RITES.NS", "RKFORGE.NS", "RPOWER.NS", "RRKABEL.NS", "RVNL.NS", "SAGILITY.NS", "SAIL.NS", "SAILIFE.NS", "SAMMAANCAP.NS",
    "SAPPHIRE.NS", "SARDAEN.NS", "SAREGAMA.NS", "SBFC.NS", "SBICARD.NS", "SBILIFE.NS", "SBIN.NS", "SCHAEFFLER.NS", "SCHNEIDER.NS", "SCI.NS",
    "SHREECEM.NS", "SHRIRAMFIN.NS", "SHYAMMETL.NS", "SIEMENS.NS", "SIGNATURE.NS", "SJVN.NS", "SKFINDIA.NS", "SOBHA.NS", "SOLARINDS.NS", "SONACOMS.NS",
    "SONATSOFTW.NS", "SRF.NS", "STARHEALTH.NS", "SUMICHEM.NS", "SUNDARMFIN.NS", "SUNDRMFAST.NS", "SUNPHARMA.NS", "SUNTV.NS", "SUPREMEIND.NS", "SUZLON.NS",
    "SWANCORP.NS", "SWIGGY.NS", "SYNGENE.NS", "SYRMA.NS", "TARIL.NS", "TATACHEM.NS", "TATACOMM.NS", "TATACONSUM.NS", "TATAELXSI.NS", "TATAINVEST.NS",
    "TATAPOWER.NS", "TATASTEEL.NS", "TATATECH.NS", "TBOTEK.NS", "TCS.NS", "TECHM.NS", "TECHNOE.NS", "TEJASNET.NS", "THELEELA.NS", "THERMAX.NS",
    "TIINDIA.NS", "TIMKEN.NS", "TITAGARH.NS", "TITAN.NS", "TMPV.NS", "TORNTPHARM.NS", "TORNTPOWER.NS", "TRENT.NS", "TRIDENT.NS", "TRITURBINE.NS",
    "TRIVENI.NS", "TTML.NS", "TVSMOTOR.NS", "UBL.NS", "UCOBANK.NS", "ULTRACEMCO.NS", "UNIONBANK.NS", "UNITDSPR.NS", "UNOMINDA.NS", "UPL.NS",
    "USHAMART.NS", "UTIAMC.NS", "VBL.NS", "VEDL.NS", "VENTIVE.NS", "VGUARD.NS", "VIJAYA.NS", "VMM.NS", "VOLTAS.NS", "VTL.NS",
    "WAAREEENER.NS", "WELCORP.NS", "WELSPUNLIV.NS", "WHIRLPOOL.NS", "WIPRO.NS", "WOCKPHARMA.NS", "YESBANK.NS", "ZEEL.NS", "ZENSARTECH.NS", "ZENTEC.NS",
    "ZFCVINDIA.NS", "ZYDUSLIFE.NS"
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
        # "Brazil": {"Ibovespa": "^BVSP"},
        "Argentina": {"MERVAL": "^MERV"},
    },
    "Europe": {
        "UK": {"FTSE 100": "^FTSE"},
        "Germany": {"DAX": "^GDAXI"},
        "France": {"CAC 40": "^FCHI"},
        "Italy": {"FTSE MIB": "FTSEMIB.MI"},
        # "Spain": {"IBEX 35": "^IBEX"},
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
    try:
        yf_tk = yf.Ticker(yf_symbol)
        
        # Try to get info, but don't fail if it's not available
        try:
            info = yf_tk.info
        except:
            info = {}
        
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

        # Update market cap if available
        mc = info.get("marketCap")
        if mc:
            try:
                company.market_cap = float(mc)
            except (ValueError, TypeError):
                pass

        # Get historical data
        hist = yf_tk.history(period="5d")
        
        if hist.empty:
            print(f"No historical data for {yf_symbol}")
            return
        
        for idx, row in hist.iterrows():
            try:
                d = idx.date()
                exists = db.query(OHLC).filter(OHLC.company_id == company.id, OHLC.date == d).first()
                if exists:
                    continue
                
                # Convert numpy types to Python native types with error handling
                def safe_convert(val):
                    try:
                        return float(val) if not math.isnan(val) else None
                    except (ValueError, TypeError):
                        return None
                
                o = OHLC(
                    company_id=company.id,
                    date=d,
                    open=safe_convert(row["Open"]),
                    high=safe_convert(row["High"]),
                    low=safe_convert(row["Low"]),
                    close=safe_convert(row["Close"]),
                    volume=safe_convert(row["Volume"])
                )
                db.add(o)
            except Exception as row_error:
                print(f"Error processing row for {yf_symbol} on {idx}: {str(row_error)}")
                continue
        
        db.commit()
        
    except Exception as e:
        db.rollback()
        print(f"Error updating ticker {yf_symbol}: {str(e)}")
        raise

def update_all_tickers():
    if not should_update():
        print(f"[Updater] Skipping update - outside market hours or weekend. Current time: {datetime.now()}")
        return
    
    with db_session() as db:
        print(f"[Updater] Running at {datetime.now()} (Market Hours)")
        for sym in NIFTY500_TICKERS:
            try:
                update_ticker(db, sym)
                print("Updated", sym)
            except Exception as e:
                print("Error updating", sym, e)
        print("==== Update complete ====")

def update_index_data(db, index_config: dict):
    try:
        yf_tk = yf.Ticker(index_config["yahoo_symbol"])
        
        # Try to get info, but don't fail if it's not available
        try:
            info = yf_tk.info
        except:
            info = {}
        
        hist = yf_tk.history(period="2d")
        
        if hist.empty:
            print(f"No data for {index_config['symbol']} (possibly delisted or rate limited)")
            return
        
        # Safe conversion function
        def safe_convert(val):
            try:
                return float(val) if not math.isnan(val) else None
            except (ValueError, TypeError):
                return None
        
        # Convert numpy types to Python native types
        current_close = safe_convert(hist['Close'].iloc[-1])
        if current_close is None:
            print(f"Invalid close price for {index_config['symbol']}")
            return
        
        previous_close = safe_convert(hist['Close'].iloc[0]) if len(hist) > 1 else current_close
        if previous_close is None or previous_close == 0:
            previous_close = current_close
        
        pct_change = ((current_close - previous_close) / previous_close) * 100 if previous_close != 0 else 0
        
        market_cap = info.get('marketCap')
        if market_cap is None:
            market_cap = current_close * 1000
        
        try:
            market_cap = float(market_cap)
        except (ValueError, TypeError):
            market_cap = current_close * 1000
        
        index = db.query(Index).filter(Index.symbol == index_config["symbol"]).first()
        if not index:
            index = Index(symbol=index_config["symbol"], name=index_config["name"])
            db.add(index)
            db.commit()
            db.refresh(index)
        
        index.market_cap = market_cap
        index.current_price = current_close
        index.previous_close = previous_close
        index.pct_change = float(pct_change)
        index.last_updated = datetime.utcnow()
        
        today = date.today()
        existing_history = db.query(IndexHistory).filter(
            IndexHistory.index_id == index.id,
            IndexHistory.date == today
        ).first()
        
        if not existing_history:
            # Convert volume to Python int, handling NaN
            volume_val = None
            if 'Volume' in hist.columns:
                volume_val = safe_convert(hist['Volume'].iloc[-1])
                if volume_val is not None:
                    try:
                        volume_val = int(volume_val)
                    except (ValueError, TypeError):
                        volume_val = None
            
            history = IndexHistory(
                index_id=index.id,
                date=today,
                close=current_close,
                volume=volume_val
            )
            db.add(history)
        
        db.commit()
        print(f"Updated index: {index_config['name']}")
        
    except Exception as e:
        db.rollback()
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

def initial_data_load():
    """Run initial data load in background"""
    import threading
    
    def load_data():
        with db_session() as db:
            print("Running initial company data load in background...")
            for sym in NIFTY500_TICKERS:
                try:
                    update_ticker(db, sym)
                    print("Initial load:", sym)
                except Exception as e:
                    print("Error in initial load", sym, e)
            
            print("Running initial indices data load in background...")
            for index_config in MAJOR_INDICES:
                try:
                    update_index_data(db, index_config)
                    print("Initial index load:", index_config["name"])
                except Exception as e:
                    print("Error in initial index load", index_config["name"], e)
            
            print("Initial data load complete!")
    
    # Start loading in background thread
    thread = threading.Thread(target=load_data, daemon=True)
    thread.start()
    print("Initial data load started in background thread...")


def start_scheduler():
    scheduler.add_job(update_all_tickers, "interval", minutes=5)
    scheduler.add_job(update_all_indices, "interval", minutes=5)
    scheduler.start()
    print("Scheduler started. Will update companies and indices every 5 minutes during market hours.")
    
    # Run initial load in background
    initial_data_load()

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

    result = {}
    
    # Fetch data with error handling - suppress yfinance warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    hist = None
    try:
        hist = yf.download(
            all_symbols,
            period=yf_period,
            interval=yf_interval,
            progress=False,
            auto_adjust=True,
            group_by='ticker',
            threads=True,  # Use threading for faster downloads
            ignore_tz=True  # Ignore timezone issues
        )
    except Exception as e:
        # If bulk download fails completely, try to continue with empty dataframe
        print(f"Bulk download encountered errors: {str(e)}")
        # Don't return early - we'll handle missing symbols below
    
    # Handle completely empty response
    if hist is None or hist.empty:
        print("No data returned from Yahoo Finance for any symbols")
        for symbol in all_symbols:
            result[symbol] = {
                "info": symbol_to_info[symbol],
                "data": [],
                "error": "No data available from Yahoo Finance"
            }
        return result

    # Process each symbol
    for symbol in all_symbols:
        try:
            # Get data for this symbol
            if len(all_symbols) == 1:
                symbol_data = hist
            else:
                # Check if symbol exists in the dataframe
                if symbol not in hist.columns.get_level_values(0):
                    result[symbol] = {
                        "info": symbol_to_info[symbol],
                        "data": [],
                        "error": "Symbol not found in response (possibly delisted or rate limited)"
                    }
                    continue
                
                symbol_data = hist[symbol]
            
            # Check if data is empty
            if symbol_data.empty or symbol_data.isna().all().all():
                result[symbol] = {
                    "info": symbol_to_info[symbol],
                    "data": [],
                    "error": "No data available (possibly delisted)"
                }
                continue
            
            # Reset index and get date column
            symbol_data = symbol_data.reset_index()
            date_col = 'Date' if 'Date' in symbol_data.columns else 'Datetime'
            
            # Remove rows with missing dates
            symbol_data = symbol_data.dropna(subset=[date_col])
            
            if symbol_data.empty:
                result[symbol] = {
                    "info": symbol_to_info[symbol],
                    "data": [],
                    "error": "No valid data after filtering"
                }
                continue
            
            # Build OHLC data
            ohlc_data = []
            for _, r in symbol_data.iterrows():
                try:
                    # Format datetime based on interval
                    if yf_interval in ["5m", "15m", "30m", "1h"]:
                        date_str = r[date_col].strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        date_str = r[date_col].strftime("%Y-%m-%d")
                    
                    ohlc_data.append({
                        "date": date_str,
                        "open": safe_float(r.get("Open")),
                        "high": safe_float(r.get("High")),
                        "low": safe_float(r.get("Low")),
                        "close": safe_float(r.get("Close")),
                        "volume": safe_float(r.get("Volume"))
                    })
                except Exception as row_error:
                    # Skip problematic rows
                    print(f"Error processing row for {symbol}: {str(row_error)}")
                    continue
            
            if not ohlc_data:
                result[symbol] = {
                    "info": symbol_to_info[symbol],
                    "data": [],
                    "error": "Failed to process any data rows"
                }
            else:
                result[symbol] = {
                    "info": symbol_to_info[symbol],
                    "data": ohlc_data
                }
            
        except KeyError as ke:
            result[symbol] = {
                "info": symbol_to_info[symbol],
                "data": [],
                "error": f"Symbol not found in response: {str(ke)}"
            }
        except Exception as e:
            result[symbol] = {
                "info": symbol_to_info[symbol],
                "data": [],
                "error": f"Processing error: {str(e)}"
            }
    
    return result

def get_index_current_price(symbol: str):
    import time
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            yf_tk = yf.Ticker(symbol)
            
            # Try to get historical data first (more reliable)
            hist = yf_tk.history(period="2d")
            
            if hist.empty:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed for {symbol}, retrying...")
                    time.sleep(retry_delay)
                    continue
                raise HTTPException(
                    status_code=404, 
                    detail="No data available for this symbol (possibly delisted or invalid)"
                )
            
            # Get info (may fail for some symbols, so we handle it separately)
            try:
                info = yf_tk.info
                name = info.get('shortName', symbol)
                currency = info.get('currency', 'USD')
            except:
                name = symbol
                currency = 'USD'
            
            current_price = float(hist['Close'].iloc[-1])
            previous_close = float(hist['Close'].iloc[0]) if len(hist) > 1 else current_price
            pct_change = ((current_price - previous_close) / previous_close) * 100 if previous_close != 0 else 0
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "previous_close": previous_close,
                "pct_change": float(pct_change),
                "name": name,
                "currency": currency,
                "last_updated": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}, retrying...")
                time.sleep(retry_delay)
                continue
            
            # Check for specific error types
            error_msg = str(e).lower()
            if 'rate limit' in error_msg or 'too many requests' in error_msg:
                raise HTTPException(
                    status_code=429, 
                    detail="Rate limited by Yahoo Finance. Please try again later."
                )
            elif 'delisted' in error_msg:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Symbol {symbol} appears to be delisted or invalid"
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to fetch data: {str(e)}"
                )

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
