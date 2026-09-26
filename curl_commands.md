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
| `/model/generate` | `POST` | Generates Python code completion from input prompt | `200 OK` |
| `/model/instruction` | `POST` | Executes coding instruction, program explanation, or bug fixing | `200 OK` |
| `/tokenizer/encode` | `POST` | Tokenizes Python code into token strings and IDs | `200 OK` |
| `/tokenizer/decode` | `POST` | Decodes token IDs back into Python code text | `200 OK` |
| `/openapi.json` | `GET` | OpenAPI JSON schema specification | `200 OK` |
| `/docs` | `GET` | Swagger UI documentation HTML | `200 OK` |

---

## 🛠️ Detailed cURL Commands

### 1. Code Generation Endpoint (`POST /model/generate`)
Generates Python code completion from a code prompt or prefix.

* **Basic cURL Command:**
  ```bash
  curl -X POST http://localhost:8000/model/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "def fibonacci(n: int) -> int:", "max_new_tokens": 100, "temperature": 0.5}'
  ```

---

### 2. Instruction Following Endpoint (`POST /model/instruction`)
Executes code generation, program explanation, or bug fixing instructions.

* **Task 1: Code Generation:**
  ```bash
  curl -X POST http://localhost:8000/model/instruction \
    -H "Content-Type: application/json" \
    -d '{
      "instruction": "Write a recursive function to calculate factorial of n",
      "task_type": "generate",
      "max_new_tokens": 150,
      "temperature": 0.5
    }'
  ```

* **Task 2: Program Explanation:**
  ```bash
  curl -X POST http://localhost:8000/model/instruction \
    -H "Content-Type: application/json" \
    -d '{
      "instruction": "Explain what this function does",
      "input_code": "def reverse_string(s: str): return s[::-1]",
      "task_type": "explain"
    }'
  ```

* **Task 3: Bug Fixing:**
  ```bash
  curl -X POST http://localhost:8000/model/instruction \
    -H "Content-Type: application/json" \
    -d '{
      "instruction": "Fix ZeroDivisionError bug",
      "input_code": "def compute_average(nums): return sum(nums) / len(nums)",
      "task_type": "fix_bug"
    }'
  ```

---

### 3. Model Information (`GET /model/info`)
  ```bash
  curl -X GET http://localhost:8000/model/info
  ```

---

### 4. Tokenizer Encode & Decode
* **Encode:**
  ```bash
  curl -X POST http://localhost:8000/tokenizer/encode \
    -H "Content-Type: application/json" \
    -d '{"text": "def add(a, b): return a + b"}'
  ```

* **Decode:**
  ```bash
  curl -X POST http://localhost:8000/tokenizer/decode \
    -H "Content-Type: application/json" \
    -d '{"token_ids": [18, 237, 85, 236, 238, 234]}'
  ```
