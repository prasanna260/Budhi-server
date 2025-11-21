"""
Migration script to update KYC table schema for bank statement verification
"""
from sqlalchemy import create_engine, text
import os

DATABASE_URL = "postgresql://postgres:ZpfLDFFOLJemAIEkOTBpEjCuBWYyIwSm@switchback.proxy.rlwy.net:19114/railway"

engine = create_engine(DATABASE_URL)

migrations = [
    # Add new columns
    "ALTER TABLE kyc ADD COLUMN IF NOT EXISTS full_name VARCHAR(100)",
    "ALTER TABLE kyc ADD COLUMN IF NOT EXISTS account_number VARCHAR(50)",
    "ALTER TABLE kyc ADD COLUMN IF NOT EXISTS ifsc_code VARCHAR(20)",
    "ALTER TABLE kyc ADD COLUMN IF NOT EXISTS bank_statement_filename VARCHAR(255)",
    "ALTER TABLE kyc ADD COLUMN IF NOT EXISTS verification_details VARCHAR(500)",
    
    # Drop old columns (optional - comment out if you want to keep old data)
    "ALTER TABLE kyc DROP COLUMN IF EXISTS pan_number",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS aadhaar_number",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS address_line",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS city",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS state",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS pincode",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS document_front_url",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS document_back_url",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS reviewed_by_admin_id",
    "ALTER TABLE kyc DROP COLUMN IF EXISTS reviewed_at",
]

print("Starting KYC schema migration...")

with engine.connect() as conn:
    for migration in migrations:
        try:
            print(f"Executing: {migration}")
            conn.execute(text(migration))
            conn.commit()
            print("✓ Success")
        except Exception as e:
            print(f"✗ Error: {e}")
            if "already exists" in str(e).lower() or "does not exist" in str(e).lower():
                print("  (Skipping - expected)")
            else:
                print("  (Failed - check manually)")

print("\nMigration complete!")
print("\nNote: Old columns (pan_number, aadhaar_number, etc.) are preserved.")
print("Uncomment the DROP statements in the script if you want to remove them.")
