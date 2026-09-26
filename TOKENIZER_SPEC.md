# 🔤 PyGPT Tokenizer Specification

## 02. PyGPT Tokenizer & Vocabulary Definition

The **PyGPT Tokenizer** is a specialized, Python-aware lexer and subword tokenizer designed to encode Python code snippets, stack traces, and docstrings with high token efficiency and exact indentation fidelity.

---

## 🗂️ Vocabulary Coverage Summary

| Vocabulary Category | Included Token Types | Purpose |
| :--- | :--- | :--- |
| **Special Control Tokens** | `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<mask>`, `<code_start>`, `<code_end>`, `<indent>`, `<dedent>` | Sequence padding, boundary control, masking & structural indentation tracking |
| **Python Reserved Keywords** | `def`, `class`, `async`, `await`, `return`, `import`, `from`, `if`, `else`, `elif`, `for`, `while`, `try`, `except`, `finally`, `raise`, `with`, `yield`, `lambda`, `assert`, `pass`, `break`, `continue`, `global`, `nonlocal`, `in`, `is`, `and`, `or`, `not`, `True`, `False`, `None` | All 35 standard Python language keywords |
| **Multi-char Operators & Indents** | `->`, `:=`, `==`, `!=`, `<=`, `>=`, `+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`, `@=`, `&=`, `|=`, `^=`, `<<=`, `>>=`, `**`, `//`, `<<`, `>>`, `...`, `    ` (4-spaces), `        ` (8-spaces) | Single-token representation of complex operators & code indentation |
| **Python Built-in Functions & Types** | `print`, `len`, `range`, `int`, `str`, `dict`, `list`, `set`, `tuple`, `bool`, `float`, `type`, `isinstance`, `super`, `enumerate`, `zip`, `map`, `filter`, `open`, `dataclass`, `property`, `self`, `cls`, `args`, `kwargs` | Native Python standard library functions, types, and parameter conventions |
| **Framework Identifiers** | `FastAPI`, `Pydantic`, `BaseModel`, `Field`, `Uvicorn`, `PyTorch`, `torch`, `nn`, `Module`, `Tensor`, `cuda`, `np`, `pd`, `asyncio`, `pytest`, `uv` | Common Python ecosystem and web framework symbols |
| **Docstring & Natural Language** | `Args:`, `Returns:`, `Raises:`, `Yields:`, `Examples:`, `Note:`, `Warning:`, `TODO:`, `Parameters:`, ASCII characters `0-255` | Natural language documentation parsing and subword fallback |

---

## ⚙️ Tokenizer Features & Specifications

1. **Exact Reversibility (Lossless Decoding)**:
   - Guaranteed property: `tokenizer.decode(tokenizer.encode(code)) == code`
   - Preserves all indentation, spaces, line breaks (`\n`), and tab characters (`\t`).

2. **Regex Lexer Pre-Tokenization**:
   - Syntactic pre-splitting prevents cross-boundary token merging (e.g. `def foo()` will tokenize into `["def", " ", "foo", "(", ")"]` rather than merging keywords into identifier boundaries).

3. **Subword & Byte Fallback**:
   - For unknown user-defined variables or exotic characters, the tokenizer gracefully falls back to character and ASCII byte-level tokenization without losing information or raising `<unk>` errors.

4. **Vocabulary Persistence**:
   - The tokenizer exports its complete mapping to [`vocab.json`](vocab.json), allowing zero-dependency loading in production environments.

---

## 💻 Python Usage Example

```python
from tokenizer import pygpt_tokenizer

code_snippet = """
async def get_health_status() -> dict:
    return {"status": "healthy"}
"""

# Encode to token IDs
token_ids = pygpt_tokenizer.encode(code_snippet, add_special_tokens=True)
print(f"Token IDs: {token_ids}")

# Decode back to text
reconstructed_code = pygpt_tokenizer.decode(token_ids)
assert reconstructed_code == code_snippet
```
