from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.orm import sessionmaker, Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import jwt
import hashlib
import json
import os
from google.oauth2 import id_token
from google.auth.transport import requests
from google_auth_oauthlib.flow import Flow


# =========================================#
# CONFIGURATION
# =========================================
DATABASE_URL = "postgresql://postgres:ZpfLDFFOLJemAIEkOTBpEjCuBWYyIwSm@switchback.proxy.rlwy.net:19114/railway"
SECRET_KEY = "YOUR_SECRET_KEY_CHANGE_THIS"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Google OAuth Configuration
GOOGLE_CLIENT_SECRET_FILE = os.path.join(os.path.dirname(__file__), "client_secret_569540551378-mulbtgst82mumc131p7odv200dhv9ipo.apps.googleusercontent.com.json")
with open(GOOGLE_CLIENT_SECRET_FILE, "r") as f:
    GOOGLE_CLIENT_CONFIG = json.load(f)["web"]

GOOGLE_CLIENT_ID = GOOGLE_CLIENT_CONFIG["client_id"] or ""
GOOGLE_CLIENT_SECRET = GOOGLE_CLIENT_CONFIG["client_secret"]
GOOGLE_REDIRECT_URI = "http://localhost:5713/auth/google/callback"  # Frontend callback URL

app = FastAPI(title="BudhiTrade Backend")

# Allow your React frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================
# DATABASE SETUP
# =========================================
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

# =========================================
# MODELS
# =========================================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    mobile = Column(String(20), unique=False, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    google_id = Column(String(255), unique=True, nullable=True, index=True)

    role = Column(String(20), default="user", nullable=False)  # 'user' | 'admin'
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    profile = relationship("Profile", back_populates="user", uselist=False)
    wallet = relationship("Wallet", back_populates="user", uselist=False)
    kyc = relationship("KYC", back_populates="user", uselist=False)


class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    full_name = Column(String(100), nullable=True, default="")
    bio = Column(String(255), nullable=True, default="")

    user = relationship("User", back_populates="profile")


class Wallet(Base):
    __tablename__ = "wallets"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    balance = Column(Float, default=0.0)  # INR float per choice B
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="wallet")
    transactions = relationship("WalletTransaction", back_populates="wallet", cascade="all, delete-orphan")


class WalletTransaction(Base):
    __tablename__ = "wallet_transactions"
    id = Column(Integer, primary_key=True, index=True)
    wallet_id = Column(Integer, ForeignKey("wallets.id"), nullable=False)
    amount = Column(Float, nullable=False)  # +credit / -debit
    transaction_type = Column(String(50))   # "credit" | "debit"
    method = Column(String(50), nullable=True)  # "manual" | "upi" | "card" | "admin" | etc.
    reference_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    wallet = relationship("Wallet", back_populates="transactions")


class KYC(Base):
    __tablename__ = "kyc"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    pan_number = Column(String(20), nullable=True)
    aadhaar_number = Column(String(20), nullable=True)
    address_line = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    pincode = Column(String(20), nullable=True)
    document_front_url = Column(String(255), nullable=True)
    document_back_url = Column(String(255), nullable=True)
    status = Column(String(20), default="pending")  # "pending" | "approved" | "rejected"
    reviewed_by_admin_id = Column(Integer, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="kyc")


Base.metadata.create_all(bind=engine)

# =========================================
# SECURITY
# =========================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

def sha256_hash(password: str) -> str:
    """Hash password using SHA-256 to reduce length safely."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def hash_password(password: str) -> str:
    """First SHA-256 hash, then bcrypt hash."""
    sha_hashed = sha256_hash(password)
    return pwd_context.hash(sha_hashed)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify by SHA-256 hashing the input first, then bcrypt verification."""
    sha_hashed = sha256_hash(plain_password)
    return pwd_context.verify(sha_hashed, hashed_password)

# =========================================
# JWT HELPERS
# =========================================
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# =========================================
# DATABASE DEPENDENCY
# =========================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =========================================
# AUTH DEPENDENCY
# =========================================
from typing import Optional
from fastapi import Query

# Make security optional to support query parameter tokens
security_optional = HTTPBearer(auto_error=False)

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_optional),
    token: Optional[str] = Query(None),
    db: Session = Depends(get_db)
) -> User:
    # Try to get token from Authorization header first, then query parameter
    token_str = None
    if credentials:
        token_str = credentials.credentials
    elif token:
        token_str = token
    
    if not token_str:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. Provide token in Authorization header or query parameter."
        )
    
    identifier = decode_access_token(token_str)
    # Try to find user by username or email (for Google OAuth users)
    user = db.query(User).filter(
        (User.username == identifier) | (User.email == identifier)
    ).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

def require_admin(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")

# =========================================
# REQUEST SCHEMAS
# =========================================
class SignupRequest(BaseModel):
    username: str
    email: str
    password: str
    mobile: str | None = None


class LoginRequest(BaseModel):
    username: str
    password: str


class ProfileUpdate(BaseModel):
    full_name: str | None = None
    bio: str | None = None

class GoogleTokenRequest(BaseModel):
    token: str


class AddFundsRequest(BaseModel):
    amount: float = Field(gt=0, description="Amount in INR")
    method: str | None = "manual"
    reference_id: str | None = None


class WithdrawRequest(BaseModel):
    amount: float = Field(gt=0, description="Amount in INR")
    destination: str | None = None  # placeholder (UPI/bank masked)
    reference_id: str | None = None


class KYCSubmitRequest(BaseModel):
    pan_number: str | None = None
    aadhaar_number: str | None = None
    address_line: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None
    document_front_url: str | None = None
    document_back_url: str | None = None


class KYCDecisionRequest(BaseModel):
    approve: bool
    reason: str | None = None


# =========================================
# HELPERS
# =========================================
def _record_wallet_tx(db: Session, wallet: Wallet, amount: float, tx_type: str, method: str | None, reference_id: str | None):
    tx = WalletTransaction(
        wallet_id=wallet.id,
        amount=amount,
        transaction_type=tx_type,
        method=method,
        reference_id=reference_id,
    )
    wallet.updated_at = datetime.utcnow()
    db.add(tx)


# =========================================
# ROUTES
# =========================================
@app.get("/")
def root():
    return {"message": "BudhiTrade API is running!"}

@app.post("/signup")
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    if db.query(User).filter((User.username == request.username) | (User.email == request.email)).first():
        raise HTTPException(status_code=400, detail="Username or Email already exists")

    user = User(
        username=request.username,
        email=request.email,
        mobile=request.mobile,
        hashed_password=hash_password(request.password),
        role="user",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Create empty profile
    profile = Profile(user_id=user.id, full_name="", bio="")
    db.add(profile)

    # Create wallet with default 0.0 balance
    wallet = Wallet(user_id=user.id, balance=0.0)
    db.add(wallet)

    # Create initial KYC row with pending (optional)
    if not user.kyc:
        kyc = KYC(user_id=user.id, status="pending")
        db.add(kyc)

    db.commit()
    return {"message": "Signup successful", "user_id": user.id}


# ---------- LOGIN ----------
@app.post("/login")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Check if user has a password (not a Google OAuth user)
    if not user.hashed_password:
        raise HTTPException(status_code=401, detail="This account uses Google Sign-In. Please login with Google.")
    
    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not user.profile:
        profile = Profile(user_id=user.id, full_name="", bio="")
        db.add(profile)
        db.commit()
        db.refresh(user)
    if not user.wallet:
        wallet = Wallet(user_id=user.id, balance=0.0)
        db.add(wallet)
        db.commit()
        db.refresh(user)
    if not user.kyc:
        kyc = KYC(user_id=user.id, status="pending")
        db.add(kyc)
        db.commit()
        db.refresh(user)

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}


# ---------- PROFILE ----------
class ProfileResponse(BaseModel):
    username: str
    email: str
    mobile: str | None
    full_name: str
    bio: str

    class Config:
        from_attributes = True

@app.get("/profile", response_model=ProfileResponse)
def get_profile(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get the user with profile loaded
    from sqlalchemy.orm import joinedload
    db_user = db.query(User).options(joinedload(User.profile)).filter(User.id == user.id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
        
    prof = db_user.profile
    return {
        "username": db_user.username,
        "email": db_user.email,
        "mobile": db_user.mobile,
        "full_name": (prof.full_name if prof and prof.full_name else ""),
        "bio": (prof.bio if prof and prof.bio else "")
    }

@app.put("/profile")
def update_profile(update_data: ProfileUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    prof = user.profile
    if not prof:
        prof = Profile(user_id=user.id, full_name="", bio="")
        db.add(prof)
        db.flush()

    if update_data.full_name is not None:
        prof.full_name = update_data.full_name
    if update_data.bio is not None:
        prof.bio = update_data.bio

    db.commit()
    db.refresh(prof)
    return {"message": "Profile updated successfully"}

# =========================================
# GOOGLE OAUTH ROUTES
# =========================================
@app.get("/auth/google")
def google_auth():
    """Initiate Google OAuth flow"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [GOOGLE_REDIRECT_URI]
            }
        },
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"]
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI
    
    authorization_url, state = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent'
    )
    
    return {"auth_url": authorization_url, "state": state}

@app.post("/auth/google/token")
def google_token_exchange(request: GoogleTokenRequest, db: Session = Depends(get_db)):
    """Exchange Google ID token for JWT"""
    try:
        # Verify the Google ID token
        idinfo = id_token.verify_oauth2_token(
            request.token, 
            requests.Request(), 
            GOOGLE_CLIENT_ID
        )
        
        # Extract user information
        google_id = idinfo.get("sub")
        email = idinfo.get("email")
        name = idinfo.get("name", "")
        picture = idinfo.get("picture", "")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by Google")
        
        # Check if user exists by Google ID or email
        user = db.query(User).filter(
            (User.google_id == google_id) | (User.email == email)
        ).first()
        
        if not user:
            # Create new user
            # Generate username from email if not provided
            username_base = email.split("@")[0]
            username = username_base
            counter = 1
            while db.query(User).filter(User.username == username).first():
                username = f"{username_base}{counter}"
                counter += 1
            
            user = User(
                username=username,
                email=email,
                google_id=google_id,
                hashed_password=None
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            
            # Create profile with name from Google
            profile = Profile(user_id=user.id, full_name=name, bio="")
            db.add(profile)
            
            # Create wallet with default 0.0 balance
            wallet = Wallet(user_id=user.id, balance=0.0)
            db.add(wallet)
            
            # Create initial KYC row
            kyc = KYC(user_id=user.id, status="pending")
            db.add(kyc)
            
            db.commit()
        else:
            # Update existing user if needed
            if not user.google_id:
                user.google_id = google_id
                db.commit()
            
            # Update profile name if not set
            if user.profile and not user.profile.full_name and name:
                user.profile.full_name = name
                db.commit()
            
            db.refresh(user)
        
        # Generate JWT token
        access_token = create_access_token(data={"sub": user.username or user.email})
        return {"access_token": access_token, "token_type": "bearer"}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid Google token: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")


# ---------- WALLET ----------
@app.get("/wallet")
def get_wallet(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    wallet = user.wallet
    return {"balance": round(wallet.balance, 2), "updated_at": wallet.updated_at}


@app.post("/wallet/add")
def add_funds(request: AddFundsRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    wallet = user.wallet

    # In production, validate payment gateway webhook and idempotency here.
    wallet.balance = float(wallet.balance) + float(request.amount)
    _record_wallet_tx(db, wallet, amount=+request.amount, tx_type="credit", method=request.method or "manual", reference_id=request.reference_id)

    db.commit()
    db.refresh(wallet)
    return {"message": "Funds added", "new_balance": round(wallet.balance, 2)}


@app.post("/wallet/withdraw")
def withdraw_funds(request: WithdrawRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    wallet = user.wallet

    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be greater than zero")

    if wallet.balance < request.amount:
        raise HTTPException(status_code=400, detail="Insufficient wallet balance")

    # In production: enforce KYC approved for withdrawals
    # if user.kyc and user.kyc.status != "approved":
    #     raise HTTPException(status_code=403, detail="KYC not approved")

    wallet.balance = float(wallet.balance) - float(request.amount)
    _record_wallet_tx(db, wallet, amount=-request.amount, tx_type="debit", method="withdraw", reference_id=request.reference_id)

    db.commit()
    db.refresh(wallet)
    return {
        "message": "Withdrawal request created (mock). Process via payout later.",
        "new_balance": round(wallet.balance, 2),
    }


@app.get("/wallet/transactions")
def get_wallet_transactions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    wallet = user.wallet
    transactions = (
        db.query(WalletTransaction)
        .filter(WalletTransaction.wallet_id == wallet.id)
        .order_by(WalletTransaction.id.desc())
        .all()
    )

    return [
        {
            "amount": tx.amount,
            "type": tx.transaction_type,
            "method": tx.method,
            "reference_id": tx.reference_id,
            "created_at": tx.created_at,
        }
        for tx in transactions
    ]


# ---------- KYC ----------
@app.get("/kyc/status")
def kyc_status(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    kyc = user.kyc
    if not kyc:
        return {"status": "pending"}
    return {
        "status": kyc.status,
        "pan_number": kyc.pan_number,
        "aadhaar_number": kyc.aadhaar_number,
        "address_line": kyc.address_line,
        "city": kyc.city,
        "state": kyc.state,
        "pincode": kyc.pincode,
        "document_front_url": kyc.document_front_url,
        "document_back_url": kyc.document_back_url,
        "reviewed_at": kyc.reviewed_at,
    }


@app.post("/kyc/submit")
def kyc_submit(payload: KYCSubmitRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    kyc = user.kyc

    if not kyc:
        kyc = KYC(user_id=user.id)
        db.add(kyc)
        db.flush()

    # Users can re-submit → status goes back to pending until reviewed
    kyc.pan_number = payload.pan_number
    kyc.aadhaar_number = payload.aadhaar_number
    kyc.address_line = payload.address_line
    kyc.city = payload.city
    kyc.state = payload.state
    kyc.pincode = payload.pincode
    kyc.document_front_url = payload.document_front_url
    kyc.document_back_url = payload.document_back_url
    kyc.status = "pending"
    kyc.updated_at = datetime.utcnow()

    db.commit()
    return {"message": "KYC submitted. Awaiting review."}


# ---------- ADMIN (simple) ----------
@app.get("/admin/kyc/pending")
def admin_list_pending_kyc(admin: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(admin)

    rows = db.query(KYC).filter(KYC.status == "pending").order_by(KYC.created_at.asc()).all()
    return [
        {
            "kyc_id": r.id,
            "user_id": r.user_id,
            "username": db.query(User.username).filter(User.id == r.user_id).scalar(),
            "pan_number": r.pan_number,
            "aadhaar_number": r.aadhaar_number,
            "created_at": r.created_at,
        }
        for r in rows
    ]


@app.post("/admin/kyc/{user_id}/decide")
def admin_decide_kyc(user_id: int, decision: KYCDecisionRequest, admin: User = Depends(get_current_user), db: Session = Depends(get_db)):
    require_admin(admin)

    kyc = db.query(KYC).filter(KYC.user_id == user_id).first()
    if not kyc:
        raise HTTPException(status_code=404, detail="KYC not found for user")

    if decision.approve:
        kyc.status = "approved"
    else:
        kyc.status = "rejected"
    kyc.reviewed_by_admin_id = admin.id
    kyc.reviewed_at = datetime.utcnow()
    kyc.updated_at = datetime.utcnow()

    db.commit()
    return {"message": f"KYC {kyc.status} for user {user_id}"}

# =========================================
# KOTAK NEO BROKER CONNECT
# =========================================
from neo_api_client import NeoAPI
import traceback

class KotakInitRequest(BaseModel):
    consumer_key: str
    consumer_secret: str


class KotakLoginRequest(BaseModel):
    mobile_number: str
    password: str


class Kotak2FARequest(BaseModel):
    otp: str


# --- MODELS ---
class KotakConnection(Base):
    __tablename__ = "kotak_connections"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    consumer_key = Column(String(255), nullable=False)
    consumer_secret = Column(String(255), nullable=False)
    session_status = Column(String(50), default="disconnected")  # connected | pending_2fa | disconnected
    session_token = Column(String(255), nullable=True)
    mobile_number = Column(String(20), nullable=True)  # Store for session continuity
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", backref="kotak_connection")


Base.metadata.create_all(bind=engine)


# --- INTERNAL UTILS ---
# In-memory cache for NeoAPI client instances (keyed by user_id)
# This is needed because NeoAPI maintains session state between login and 2FA
_kotak_clients = {}

def get_kotak_connection(db: Session, user_id: int) -> KotakConnection:
    kc = db.query(KotakConnection).filter(KotakConnection.user_id == user_id).first()
    if not kc:
        kc = KotakConnection(user_id=user_id, consumer_key="", consumer_secret="", session_status="disconnected")
        db.add(kc)
        db.commit()
        db.refresh(kc)
    return kc


def get_or_create_kotak_client(kc: KotakConnection) -> NeoAPI:
    """Get existing client from cache or create new one"""
    user_id = kc.user_id
    
    # If client exists in cache and keys match, reuse it
    if user_id in _kotak_clients:
        return _kotak_clients[user_id]
    
    # Create new client and cache it
    client = NeoAPI(
        consumer_key=kc.consumer_key,
        consumer_secret=kc.consumer_secret,
        environment="prod",
        access_token=None,
        neo_fin_key=None
    )
    _kotak_clients[user_id] = client
    return client


def clear_kotak_client(user_id: int):
    """Remove client from cache"""
    if user_id in _kotak_clients:
        del _kotak_clients[user_id]


# ---------- INIT CONNECTION ----------
@app.post("/broker/kotak/init")
def kotak_init(req: KotakInitRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    kc = get_kotak_connection(db, user.id)
    kc.consumer_key = req.consumer_key.strip()
    kc.consumer_secret = req.consumer_secret.strip()
    kc.session_status = "disconnected"
    kc.session_token = None
    kc.updated_at = datetime.utcnow()
    
    # Clear any cached client when keys change
    clear_kotak_client(user.id)
    
    db.commit()
    return {"message": "Kotak API keys stored successfully. You can now login."}


# ---------- LOGIN ----------
@app.post("/broker/kotak/login")
def kotak_login(req: KotakLoginRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    kc = get_kotak_connection(db, user.id)
    if not kc.consumer_key or not kc.consumer_secret:
        raise HTTPException(status_code=400, detail="Broker keys not found. Run /broker/kotak/init first.")

    try:
        # Clear old client and create fresh one for new login
        clear_kotak_client(user.id)
        client = get_or_create_kotak_client(kc)
        
        resp = client.login(mobilenumber=req.mobile_number, password=req.password)
        
        # Store mobile number for session continuity
        kc.mobile_number = req.mobile_number
        kc.session_status = "pending_2fa"
        kc.updated_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Login successful. Please provide OTP using /broker/kotak/2fa.", "response": resp}
    except Exception as e:
        traceback.print_exc()
        clear_kotak_client(user.id)
        raise HTTPException(status_code=400, detail=f"Login failed: {str(e)}")


# ---------- OTP / 2FA ----------
@app.post("/broker/kotak/2fa")
def kotak_2fa(req: Kotak2FARequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    kc = get_kotak_connection(db, user.id)
    if kc.session_status != "pending_2fa":
        raise HTTPException(status_code=400, detail="Login not initiated or already connected.")

    try:
        # Reuse the same client instance from login
        client = get_or_create_kotak_client(kc)
        resp = client.session_2fa(OTP=req.otp)
        
        kc.session_status = "connected"
        kc.updated_at = datetime.utcnow()
        db.commit()
        
        return {"message": "Broker connected successfully.", "status": kc.session_status, "response": resp}
    except Exception as e:
        traceback.print_exc()
        clear_kotak_client(user.id)
        kc.session_status = "disconnected"
        db.commit()
        raise HTTPException(status_code=400, detail=f"2FA verification failed: {str(e)}")


# ---------- PORTFOLIO ----------
@app.get("/broker/kotak/portfolio")
def kotak_portfolio(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    kc = get_kotak_connection(db, user.id)
    if kc.session_status != "connected":
        raise HTTPException(status_code=401, detail="Broker not connected. Login and complete 2FA first.")

    try:
        client = get_or_create_kotak_client(kc)
        holdings = client.holdings()
        positions = client.positions()
        limits = client.limits(segment="ALL", exchange="ALL", product="ALL")
        return {
            "holdings": holdings,
            "positions": positions,
            "limits": limits
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to fetch portfolio: {str(e)}")


# ---------- DISCONNECT ----------
@app.post("/broker/kotak/disconnect")
def kotak_disconnect(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    kc = get_kotak_connection(db, user.id)
    kc.session_status = "disconnected"
    kc.session_token = None
    kc.updated_at = datetime.utcnow()
    
    # Clear cached client
    clear_kotak_client(user.id)
    
    db.commit()
    return {"message": "Disconnected from Kotak Neo."}
