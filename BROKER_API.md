# Kotak Neo Broker Integration API

## Overview
The broker integration allows users to connect their Kotak Neo trading account to BudhiTrade and fetch their portfolio data.

## Authentication Flow

### 1. Initiate Connection (Send OTP)
```http
POST /broker/connect/initiate
Authorization: Bearer <your_jwt_token>
Content-Type: application/json

{
  "consumer_key": "your_kotak_consumer_key",
  "consumer_secret": "your_kotak_consumer_secret",
  "mobile_number": "7019073542",
  "password": "your_kotak_password"
}
```

**Response:**
```json
{
  "message": "OTP sent to your registered mobile number",
  "status": "pending_otp"
}
```

### 2. Verify OTP (Complete Connection)
```http
POST /broker/connect/verify
Authorization: Bearer <your_jwt_token>
Content-Type: application/json

{
  "otp": "123456"
}
```

**Response:**
```json
{
  "message": "Broker connected successfully",
  "status": "connected"
}
```

## Broker Management

### Check Connection Status
```http
GET /broker/status
Authorization: Bearer <your_jwt_token>
```

**Response:**
```json
{
  "is_connected": true,
  "broker_name": "kotak_neo",
  "status": "connected",
  "last_connected_at": "2025-11-12T10:30:00",
  "mobile_number": "9999"
}
```

### Disconnect Broker
```http
POST /broker/disconnect
Authorization: Bearer <your_jwt_token>
```

**Response:**
```json
{
  "message": "Broker disconnected successfully"
}
```

## Portfolio Data

### Get Holdings
```http
GET /broker/portfolio/holdings
Authorization: Bearer <your_jwt_token>
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "RELIANCE",
      "quantity": 10,
      "average_price": 2500.00,
      "current_price": 2550.00,
      ...
    }
  ]
}
```

### Get Positions
```http
GET /broker/portfolio/positions
Authorization: Bearer <your_jwt_token>
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "NIFTY",
      "quantity": 50,
      "buy_price": 19500.00,
      "current_price": 19550.00,
      ...
    }
  ]
}
```

### Get Complete Portfolio
```http
GET /broker/portfolio
Authorization: Bearer <your_jwt_token>
```

**Response:**
```json
{
  "success": true,
  "holdings": [...],
  "positions": [...]
}
```

### Get Trading Limits
```http
GET /broker/limits
Authorization: Bearer <your_jwt_token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "cash": 50000.00,
    "collateral": 10000.00,
    ...
  }
}
```

## Security Features

1. **Encryption**: All sensitive data (credentials, tokens) are encrypted using Fernet encryption
2. **JWT Authentication**: All endpoints require valid JWT token
3. **Secure Storage**: Credentials stored encrypted in database
4. **Session Management**: Automatic session token management

## Database Schema

New table `broker_connections`:
- Stores encrypted broker credentials
- Tracks connection status
- Manages session tokens
- One connection per user per broker

## Environment Variables

Add to your `.env` file:
```
ENCRYPTION_KEY=<generate_using_Fernet.generate_key()>
```

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid credentials, OTP failed)
- `401`: Unauthorized (invalid JWT)
- `404`: Not found (no broker connection)
- `500`: Server error (API failure)
