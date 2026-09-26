from typing import List
from pydantic import BaseModel, Field


class EncodeRequest(BaseModel):
    text: str = Field(..., description="Python code string to tokenize")
    add_special_tokens: bool = Field(False, description="Include BOS and EOS tokens")


class EncodeResponse(BaseModel):
    text: str
    tokens: List[str]
    token_ids: List[int]
    token_count: int


class DecodeRequest(BaseModel):
    token_ids: List[int] = Field(..., description="List of token IDs to decode")
    skip_special_tokens: bool = Field(True, description="Omit BOS, EOS, and PAD tokens")


class DecodeResponse(BaseModel):
    text: str
    token_ids: List[int]
