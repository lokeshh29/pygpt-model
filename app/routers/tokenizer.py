from fastapi import APIRouter

from app.schemas.tokenizer import (
    DecodeRequest,
    DecodeResponse,
    EncodeRequest,
    EncodeResponse,
)
from app.tokenizer.tokenizer import pygpt_tokenizer

router = APIRouter(prefix="/tokenizer", tags=["Tokenizer Operations"])


@router.post("/encode", response_model=EncodeResponse)
def encode_text(request: EncodeRequest):
    """Encodes Python code string into token IDs using PyGPTTokenizer."""
    token_ids = pygpt_tokenizer.encode(
        request.text, add_special_tokens=request.add_special_tokens
    )
    tokens = [pygpt_tokenizer.id_to_token.get(tid, "<unk>") for tid in token_ids]
    return EncodeResponse(
        text=request.text,
        tokens=tokens,
        token_ids=token_ids,
        token_count=len(token_ids),
    )


@router.post("/decode", response_model=DecodeResponse)
def decode_tokens(request: DecodeRequest):
    """Decodes token IDs back into Python code text using PyGPTTokenizer."""
    text = pygpt_tokenizer.decode(
        request.token_ids, skip_special_tokens=request.skip_special_tokens
    )
    return DecodeResponse(
        text=text,
        token_ids=request.token_ids,
    )
