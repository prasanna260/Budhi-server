from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import httpx
import base64
import os

app = FastAPI()

# Your Sandbox credentials
API_KEY = "key_live_b9f5fbbff3904fbe989c6bd8ed5e0d8a"
API_SECRET = "secret_live_48fff3e8f9be482f91daef142f27c232"
JWT_TOKEN = "eyJ0eXAiOiJKV1MiLCJhbGciOiJSU0FTU0FfUFNTX1NIQV81MTIiLCJraWQiOiIwYzYwMGUzMS01MDAwLTRkYTItYjM3YS01ODdkYTA0ZTk4NTEifQ.eyJyZWZyZXNoX3Rva2VuIjoiZXlKMGVYQWlPaUpLVjFNaUxDSmhiR2NpT2lKU1UwRlRVMEZmVUZOVFgxTklRVjgxTVRJaUxDSnJhV1FpT2lJd1l6WXdNR1V6TVMwMU1EQXdMVFJrWVRJdFlqTTNZUzAxT0Rka1lUQTBaVGs0TlRFaWZRLmV5SnpkV0lpT2lKclpYbGZiR2wyWlY5aU9XWTFabUppWm1Zek9UQTBabUpsT1RnNVl6WmlaRGhsWkRWbE1HUTRZU0lzSW1Gd2FWOXJaWGtpT2lKclpYbGZiR2wyWlY5aU9XWTFabUppWm1Zek9UQTBabUpsT1RnNVl6WmlaRGhsWkRWbE1HUTRZU0lzSW5kdmNtdHpjR0ZqWlY5cFpDSTZJalJtTnpCaFlUYzRMVFkzTUdVdE5EQTNOaTFoTURkakxURTJZVE13WVdOaU9XTTFaQ0lzSW1GMVpDSTZJa0ZRU1NJc0ltbHVkR1Z1ZENJNklsSkZSbEpGVTBoZlZFOUxSVTRpTENKcGMzTWlPaUp3Y205a01TMWhjR2t1YzJGdVpHSnZlQzVqYnk1cGJpSXNJbVY0Y0NJNk1UYzVORGs0TWpFd09Dd2lhV0YwSWpveE56WXpORFEyTVRBNGZRLmo0ck4zR0llTDRob1haU2RaNFBOVzJzemgwSlRkVVVsRERGVE5nRjlZYjJFZXBCb0wyQkdZSWtjM0JoZV95djJKbGJnQ2ZZemFPMG92TVdxZm1HRHAyOW1IVkJGM1RzMlFXdndtMDd1OE9FSGdndEZsdlQtMk9idUQ0VTZhMERGdlBja1g2TUUxZkdrbVVnYUVJMDhIeHN3Rk5jQmFtXzVWSzNGYnhfeW82Y2pwTG42QjBjbzJUUzlfdUR3ZzRfcEZfWHhoU09DcndMRk9UaUF6T1NHb3o5STkwazBPR08wM1V0Qzg5VmpfY3RtUTF2TzZfUEhySDkyendLTkxnZmk4ZmhLNDhBa2NtclFQZGFxbDA2WERMamVNZUVSaXVjX0NmNS1CanFZOWNGNmNEUzdsNGRQUllYS0RSLXZudDNvVlhRRW5lSkJhUXV4RzdKcVRWZHN1ZyIsIndvcmtzcGFjZV9pZCI6IjRmNzBhYTc4LTY3MGUtNDA3Ni1hMDdjLTE2YTMwYWNiOWM1ZCIsInN1YiI6ImtleV9saXZlX2I5ZjVmYmJmZjM5MDRmYmU5ODljNmJkOGVkNWUwZDhhIiwiYXBpX2tleSI6ImtleV9saXZlX2I5ZjVmYmJmZjM5MDRmYmU5ODljNmJkOGVkNWUwZDhhIiwiYXVkIjoiQVBJIiwiaW50ZW50IjoiQUNDRVNTX1RPS0VOIiwiaXNzIjoicHJvZDEtYXBpLnNhbmRib3guY28uaW4iLCJpYXQiOjE3NjM0NDYxMDgsImV4cCI6MTc2MzUzMjUwOH0.iDAh6p897zmEm3LaBLB6EgHWvFxpp-ClW3kbw-oA0eN98NYMFUH0MSVXrA2LHCn5L1K-vTTqbJFRfag61CDQrbKE_Di4g5S7ll6nSSxyobegX5JY1haQ6pjig-Z9OVzA_UZvc2ZMJxTeGcqxaSNUMKXAV4OipF1EQjZVSYo-ja4vRXtI6l_uC5lHLQdEo8pRbQ33iaKHBg1n-e3rwQS47pHrgen9N5phcQTnIrrXaiVHJIkbs-hVvKBOC4kGcjBxOkEeAMCRsqLc8fIADu22YoIGVfdFwezIKW7ywf5mGCxu4iTAD_YPVUGjVKZ8984h93RhEagivKHfpUHAhAKTUA"  # Replace this with your actual JWT token
API_ENDPOINT = "https://api.sandbox.co.in/kyc/digilocker-sdk/sessions/create"

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for serving HTML)
# Create a 'static' folder and put your HTML file there
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    print("Note: Create a 'static' folder for your HTML files")

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    try:
        return FileResponse("static/index.html")
    except:
        return {"message": "Place your index.html in the 'static' folder"}

@app.post("/api/create-kyc-session")
async def create_kyc_session(request: Request):
    """Create a DigiLocker KYC session"""
    try:
        print("=" * 50)
        print("Creating KYC session...")
        print(f"API_KEY: {API_KEY[:20]}...")
        print(f"API_SECRET: {API_SECRET[:20]}...")
        
        # Create Basic Auth header
        credentials = f"{API_KEY}:{API_SECRET}"
        basic_auth = base64.b64encode(credentials.encode()).decode()
        print(f"Basic Auth (first 30 chars): {basic_auth[:30]}...")
        
        # Get the host from request
        host = request.headers.get("host", "localhost:8000")
        protocol = "https" if "https" in str(request.url) else "http"
        redirect_url = f"{protocol}://{host}/kyc-callback"
        print(f"Redirect URL: {redirect_url}")
        
        # Prepare request data
        headers = {
            "Authorization": JWT_TOKEN,
            "x-api-key": API_KEY,
            "x-api-version": "2.0",
            "Content-Type": "application/json",
        }
        
        payload = {
            "@entity": "in.co.sandbox.kyc.digilocker.sdk.session.request",
            "flow": "signin",
            "doc_types": [
                "aadhaar"
            ]
            # Note: redirect_url is not needed for SDK flow
            # The SDK handles the response via JavaScript events
        }
        
        print(f"Request headers: {headers}")
        print(f"Request payload: {payload}")
        print(f"Calling: {API_ENDPOINT}")
        
        # Call Sandbox API
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=30.0
            )
            
            print(f"Sandbox response status: {response.status_code}")
            print(f"Sandbox response headers: {dict(response.headers)}")
            print(f"Sandbox response body: {response.text}")
            print("=" * 50)
            
            if response.status_code != 200 and response.status_code != 201:
                error_detail = {
                    "error": "Failed to create session",
                    "status_code": response.status_code,
                    "details": response.text
                }
                print(f"ERROR: {error_detail}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_detail
                )
            
            response_data = response.json()
            
            # Extract session ID from the response
            session_id = None
            if 'data' in response_data and 'id' in response_data['data']:
                session_id = response_data['data']['id']
            elif 'session_id' in response_data:
                session_id = response_data['session_id']
            
            print(f"Success! Session ID: {session_id}")
            
            # Return in a format the frontend expects
            return {
                "session_id": session_id,
                "raw_response": response_data
            }
            
    except httpx.RequestError as e:
        error_msg = f"Network/Request error: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Network error",
                "message": str(e)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }
        )

@app.post("/kyc-callback")
async def kyc_callback(request: Request):
    """Handle DigiLocker webhook callbacks"""
    try:
        # Try to get JSON body
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type:
            body = await request.json()
        else:
            # Try to get form data
            form_data = await request.form()
            body = dict(form_data)
        
        print("=" * 50)
        print("KYC Callback received:")
        print(f"Headers: {dict(request.headers)}")
        print(f"Body: {body}")
        print("=" * 50)
        
        # Return success response in the format DigiLocker expects
        return {
            "code": 200,
            "message": "success",
            "data": {
                "status": "received"
            }
        }
        
    except Exception as e:
        print(f"Callback error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "code": 500,
            "message": "error",
            "error": str(e)
        }

@app.get("/kyc-callback")
async def kyc_callback_get(request: Request):
    """Handle DigiLocker GET callback (in case it uses GET)"""
    print("=" * 50)
    print("KYC GET Callback received:")
    print(f"Query params: {dict(request.query_params)}")
    print("=" * 50)
    
    return {
        "code": 200,
        "message": "success"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "KYC Backend is running"}

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔════════════════════════════════════════╗
    ║   FastAPI KYC Backend                  ║
    ╚════════════════════════════════════════╝
    
    Setup Instructions:
    1. pip install fastapi uvicorn httpx
    2. Create a 'static' folder
    3. Put your index.html in the 'static' folder
    4. python main.py
    5. Open http://localhost:8000
    
    """)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)