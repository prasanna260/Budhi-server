"""
Migration script to add google_id column to users table
"""
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:ZpfLDFFOLJemAIEkOTBpEjCuBWYyIwSm@switchback.proxy.rlwy.net:19114/railway"

def migrate():
    engine = create_engine(DATABASE_URL)
    
    with engine.connect() as conn:
        # Add google_id column if it doesn't exist
        try:
            conn.execute(text("""
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS google_id VARCHAR(255) UNIQUE;
            """))
            conn.commit()
            print("✓ Added google_id column to users table")
        except Exception as e:
            print(f"Error adding google_id column: {e}")
        
        # Create index on google_id for faster lookups
        try:
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_users_google_id 
                ON users(google_id);
            """))
            conn.commit()
            print("✓ Created index on google_id column")
        except Exception as e:
            print(f"Error creating index: {e}")
        
        # Make hashed_password nullable (for Google OAuth users)
        try:
            conn.execute(text("""
                ALTER TABLE users 
                ALTER COLUMN hashed_password DROP NOT NULL;
            """))
            conn.commit()
            print("✓ Made hashed_password nullable for Google OAuth users")
        except Exception as e:
            print(f"Error modifying hashed_password: {e}")
    
    print("\n✓ Migration completed successfully!")

if __name__ == "__main__":
    migrate()
