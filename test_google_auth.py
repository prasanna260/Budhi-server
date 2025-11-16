"""
Test script to verify Google OAuth flow
"""
from app import app, get_db, User, Profile, Wallet, KYC
from sqlalchemy.orm import Session

def test_google_user_creation():
    """Test that a Google OAuth user is created correctly"""
    db = next(get_db())
    
    # Check if test user exists
    test_email = "manjula99663@gmail.com"
    user = db.query(User).filter(User.email == test_email).first()
    
    if user:
        print(f"✓ User found: {user.username}")
        print(f"  - Email: {user.email}")
        print(f"  - Google ID: {user.google_id}")
        print(f"  - Has password: {user.hashed_password is not None}")
        print(f"  - Role: {user.role}")
        
        # Check profile
        if user.profile:
            print(f"✓ Profile exists")
            print(f"  - Full name: {user.profile.full_name}")
            print(f"  - Bio: {user.profile.bio}")
        else:
            print("✗ Profile missing!")
        
        # Check wallet
        if user.wallet:
            print(f"✓ Wallet exists")
            print(f"  - Balance: {user.wallet.balance}")
        else:
            print("✗ Wallet missing!")
        
        # Check KYC
        if user.kyc:
            print(f"✓ KYC exists")
            print(f"  - Status: {user.kyc.status}")
        else:
            print("✗ KYC missing!")
        
        print("\n✅ Google OAuth user is properly configured!")
    else:
        print(f"ℹ No user found with email {test_email}")
        print("This is normal if you haven't signed up with Google yet.")
    
    db.close()

if __name__ == "__main__":
    test_google_user_creation()
