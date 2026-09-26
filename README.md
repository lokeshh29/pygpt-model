# pygpt-model

Custom model that is specific for python coding, debugging etc.

## Running the Server

### Using `uv`

1. **Install Dependencies** (already configured in `pyproject.toml`):
   ```bash
   uv sync
   ```

2. **Run the Development Server**:
   ```bash
   uv run uvicorn main:app --reload
   ```
   Or run `main.py` directly:
   ```bash
   uv run python main.py
   ```

## Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check endpoint returning `{"status": "healthy"}`
- `GET /docs`: Interactive Swagger API documentation

For detailed cURL commands to test each endpoint, refer to [curl_commands.md](curl_commands.md).

