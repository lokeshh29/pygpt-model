# cURL Commands for PyGPT Model API

This document provides a comprehensive list of `curl` commands to test and interact with all available endpoints of the **PyGPT Model API** running locally (default: `http://localhost:8000`).

---

## 🚀 Base URL
```bash
http://localhost:8000
```
*(If running on a custom host or port, replace `localhost:8000` accordingly).*

---

## 📌 Quick Summary of Endpoints

| Endpoint | Method | Description | Expected Status |
| :--- | :--- | :--- | :--- |
| `/` | `GET` | Welcome message | `200 OK` |
| `/health` | `GET` | Health check status | `200 OK` |
| `/openapi.json` | `GET` | OpenAPI JSON schema specification | `200 OK` |
| `/docs` | `GET` | Swagger UI documentation HTML | `200 OK` |
| `/redoc` | `GET` | ReDoc documentation HTML | `200 OK` |

---

## 🛠️ Detailed cURL Commands

### 1. Root Endpoint (`GET /`)
Retrieves the welcome message from the server.

* **Basic cURL Command:**
  ```bash
  curl -X GET http://localhost:8000/
  ```

* **With JSON Header & Pretty Print (using `jq`):**
  ```bash
  curl -s -X GET http://localhost:8000/ -H "Accept: application/json" | jq .
  ```

* **Verbose Mode (Inspect HTTP Headers & Status Code):**
  ```bash
  curl -i -X GET http://localhost:8000/
  ```

* **Expected Response (`200 OK`):**
  ```json
  {
    "message": "Welcome to FastAPI server!"
  }
  ```

---

### 2. Health Check Endpoint (`GET /health`)
Checks the operational status of the server API.

* **Basic cURL Command:**
  ```bash
  curl -X GET http://localhost:8000/health
  ```

* **With Pretty Print:**
  ```bash
  curl -s -X GET http://localhost:8000/health | jq .
  ```

* **Include Response Headers:**
  ```bash
  curl -i -X GET http://localhost:8000/health
  ```

* **Expected Response (`200 OK`):**
  ```json
  {
    "status": "healthy"
  }
  ```

---

### 3. OpenAPI Schema (`GET /openapi.json`)
Fetches the complete OpenAPI schema specification in JSON format.

* **Basic cURL Command:**
  ```bash
  curl -X GET http://localhost:8000/openapi.json
  ```

* **Save OpenAPI Specification to a File:**
  ```bash
  curl -s -X GET http://localhost:8000/openapi.json -o openapi.json
  ```

---

### 4. Interactive API Documentation (`GET /docs` & `GET /redoc`)
FastAPI automatically serves interactive API documentation.

* **Swagger UI Docs (`GET /docs`):**
  ```bash
  curl -X GET http://localhost:8000/docs
  ```

* **ReDoc Documentation (`GET /redoc`):**
  ```bash
  curl -X GET http://localhost:8000/redoc
  ```

---

## ⚡ Batch Testing Script

You can run the following bash command to test all endpoints sequentially:

```bash
for endpoint in "/" "/health" "/openapi.json"; do
  echo -e "\n=== Testing GET http://localhost:8000${endpoint} ==="
  curl -s -w "\nHTTP Status: %{http_code}\n" "http://localhost:8000${endpoint}"
done
```

PowerShell equivalent:
```powershell
@("/", "/health", "/openapi.json") | ForEach-Object {
    Write-Host "`n=== Testing GET http://localhost:8000$_ ==="
    Invoke-RestMethod -Uri "http://localhost:8000$_" | ConvertTo-Json
}
```
