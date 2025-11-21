"""
Test script for KYC submission with PDF upload
"""
import requests

# Configuration
API_URL = "http://localhost:8000"
JWT_TOKEN = "your_jwt_token_here"  # Get this from /login endpoint

# KYC data
kyc_data = {
    "full_name": "John Doe",
    "account_number": "1234567890",
    "ifsc_code": "SBIN0001234",
    "pdf_password": ""  # Leave empty if PDF is not password protected
}

# PDF file path
pdf_file_path = "path/to/bank_statement.pdf"

# Prepare the request
headers = {
    "Authorization": f"Bearer {JWT_TOKEN}"
}

files = {
    "bank_statement": open(pdf_file_path, "rb")
}

data = kyc_data

# Submit KYC
print("Submitting KYC...")
response = requests.post(
    f"{API_URL}/kyc/submit",
    headers=headers,
    files=files,
    data=data
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")

# Check KYC status
print("\nChecking KYC status...")
status_response = requests.get(
    f"{API_URL}/kyc/status",
    headers=headers
)

print(f"Status: {status_response.json()}")
