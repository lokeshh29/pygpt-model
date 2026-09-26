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
| `/model/info` | `GET` | PyGPT model size, architecture, context & capabilities | `200 OK` |
| `/tokenizer/encode` | `POST` | Tokenizes Python code into token strings and IDs | `200 OK` |
| `/tokenizer/decode` | `POST` | Decodes token IDs back into Python code text | `200 OK` |
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

---

### 2. Health Check Endpoint (`GET /health`)
Checks the operational status of the server API.

* **Basic cURL Command:**
  ```bash
  curl -X GET http://localhost:8000/health
  ```

---

### 3. Model Information Endpoint (`GET /model/info`)
Retrieves the full specification of the PyGPT model including model size, transformer architecture, context length, and target capabilities.

* **Basic cURL Command:**
  ```bash
  curl -X GET http://localhost:8000/model/info
  ```

---

### 4. Tokenizer Encode Endpoint (`POST /tokenizer/encode`)
Encodes a Python code string into syntactic token strings and numerical token IDs using `PyGPTTokenizer`.

* **Basic cURL Command:**
  ```bash
  curl -X POST http://localhost:8000/tokenizer/encode \
    -H "Content-Type: application/json" \
    -d '{"text": "def add(a: int, b: int) -> int:\n    return a + b", "add_special_tokens": true}'
  ```

* **Pretty Print with `jq`:**
  ```bash
  curl -s -X POST http://localhost:8000/tokenizer/encode \
    -H "Content-Type: application/json" \
    -d '{"text": "async def health(): return {\"status\": \"ok\"}", "add_special_tokens": false}' | jq .
  ```

* **Expected Response (`200 OK`):**
  ```json
  {
    "text": "async def health(): return {\"status\": \"ok\"}",
    "tokens": [
      "async", " ", "def", " ", "health", "(", ")", ":", " ", "return", " ", "{", "\"status\"", ":", " ", "\"ok\"", "}"
    ],
    "token_ids": [15, 237, 18, 237, 85, 236, 238, 234, 237, 34, 237, 248, 88, 234, 237, 89, 249],
    "token_count": 17
  }
  ```

---

### 5. Tokenizer Decode Endpoint (`POST /tokenizer/decode`)
Decodes a list of token IDs back into Python source code.

* **Basic cURL Command:**
  ```bash
  curl -X POST http://localhost:8000/tokenizer/decode \
    -H "Content-Type: application/json" \
    -d '{"token_ids": [18, 237, 85, 236, 238, 234, 237, 34, 237, 248, 88, 234, 237, 89, 249], "skip_special_tokens": true}'
  ```

---

### 6. OpenAPI Schema & Interactive Docs
* **OpenAPI Spec:** `curl -X GET http://localhost:8000/openapi.json`
* **Swagger UI:** `curl -X GET http://localhost:8000/docs`
* **ReDoc:** `curl -X GET http://localhost:8000/redoc`

---

## ⚡ Batch Testing Script

```bash
for endpoint in "/" "/health" "/model/info"; do
  echo -e "\n=== Testing GET http://localhost:8000${endpoint} ==="
  curl -s -w "\nHTTP Status: %{http_code}\n" "http://localhost:8000${endpoint}"
done

echo -e "\n=== Testing POST http://localhost:8000/tokenizer/encode ==="
curl -s -X POST http://localhost:8000/tokenizer/encode \
  -H "Content-Type: application/json" \
  -d '{"text": "import torch", "add_special_tokens": false}'
```
