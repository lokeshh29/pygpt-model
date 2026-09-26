import json
import re
from pathlib import Path
from typing import Dict, List, Optional


class PyGPTTokenizer:
    """
    Specialized Byte-Pair Encoding (BPE) & Keyword Tokenizer for PyGPT.
    Optimized for Python code syntax, indentation structure, reserved keywords,
    built-ins, ecosystem identifiers, and natural language docstrings.
    """

    # Special Control Tokens
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    MASK_TOKEN = "<mask>"
    CODE_START_TOKEN = "<code_start>"
    CODE_END_TOKEN = "<code_end>"
    INDENT_TOKEN = "<indent>"
    DEDENT_TOKEN = "<dedent>"

    SPECIAL_TOKENS = [
        PAD_TOKEN,
        UNK_TOKEN,
        BOS_TOKEN,
        EOS_TOKEN,
        MASK_TOKEN,
        CODE_START_TOKEN,
        CODE_END_TOKEN,
        INDENT_TOKEN,
        DEDENT_TOKEN,
    ]

    # Python Syntax Keywords
    PYTHON_KEYWORDS = [
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
        "while", "with", "yield"
    ]

    # Python Built-ins, Magic Methods & Common Identifiers
    PYTHON_BUILTINS = [
        "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
        "bytes", "callable", "chr", "classmethod", "compile", "complex",
        "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec",
        "filter", "float", "format", "frozenset", "getattr", "globals",
        "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance",
        "issubclass", "iter", "len", "list", "locals", "map", "max",
        "memoryview", "min", "next", "object", "oct", "open", "ord", "pow",
        "print", "property", "range", "repr", "reversed", "round", "set",
        "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
        "tuple", "type", "vars", "zip", "__import__", "__init__", "__str__",
        "__repr__", "__call__", "__getitem__", "__setitem__", "__delitem__",
        "__len__", "__iter__", "__next__", "__enter__", "__exit__", "__eq__",
        "__ne__", "__lt__", "__gt__", "__le__", "__ge__", "__add__", "__sub__",
        "self", "cls", "args", "kwargs", "main", "app"
    ]

    # Multi-character Python Operators & Delimiters
    PYTHON_OPERATORS = [
        "->", ":=", "==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "//=", "%=",
        "**=", "@=", "&=", "|=", "^=", "<<=", ">>=", "**", "//", "<<", ">>", "...",
        "    ", "        ", "            "  # Common 4/8/12 space indentations
    ]

    # Single-character Delimiters & Punctuation
    PYTHON_SYMBOLS = list(":=()[]{},.;+-*/%@&|^~<>!?\\#$'\"`\n\t ")

    # Framework & Library Identifiers
    FRAMEWORK_IDENTIFIERS = [
        "FastAPI", "Pydantic", "BaseModel", "Field", "Uvicorn", "PyTorch",
        "torch", "nn", "Module", "Tensor", "cuda", "device", "numpy", "np",
        "pandas", "pd", "DataFrame", "Series", "asyncio", "pytest", "uvicorn",
        "uv", "requests", "json", "typing", "Optional", "List", "Dict", "Set",
        "Tuple", "Union", "Any", "Callable", "Iterable", "Generic", "TypeVar"
    ]

    # Natural Language Docstring Keywords
    DOCSTRING_TOKENS = [
        "Args:", "Returns:", "Raises:", "Yields:", "Attributes:", "Examples:",
        "Note:", "Warning:", "TODO:", "FIXME:", "Parameters:"
    ]

    def __init__(self, vocab_size: int = 32000):
        self.target_vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self._build_initial_vocabulary()

    def _build_initial_vocabulary(self) -> None:
        """Constructs the seed vocabulary covering special tokens, Python syntax, and subwords."""
        vocab_list: List[str] = []

        # 1. Special Control Tokens
        for token in self.SPECIAL_TOKENS:
            if token not in vocab_list:
                vocab_list.append(token)

        # 2. Multi-character Python Operators & Indentation
        for token in self.PYTHON_OPERATORS:
            if token not in vocab_list:
                vocab_list.append(token)

        # 3. Python Keywords
        for token in self.PYTHON_KEYWORDS:
            if token not in vocab_list:
                vocab_list.append(token)

        # 4. Built-ins, Identifiers, Frameworks & Docstrings
        for token in self.PYTHON_BUILTINS + self.FRAMEWORK_IDENTIFIERS + self.DOCSTRING_TOKENS:
            if token not in vocab_list:
                vocab_list.append(token)

        # 5. Printable Single ASCII Bytes/Characters
        for i in range(256):
            ch = chr(i)
            if ch not in vocab_list:
                vocab_list.append(ch)

        # Populate mappings
        for idx, token in enumerate(vocab_list):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _tokenize_regex(self, text: str) -> List[str]:
        """Regex-based lexer to split code into syntactic Python primitives."""
        pattern = re.compile(
            r"(?:"
            r"->|:=|==|!=|<=|>=|\+=|-=|\*=|\/=|//=|%=|\*\*=|@=|&=|\|=|\^=|<<=|>>=|\*\*|//|<<|>>|\.\.\.|"
            r" {12}| {8}| {4}|"
            r"<[a-z_]+>|"
            r"\w+|"
            r"[^\w\s]|"
            r"\n|\t|\s+"
            r")"
        )
        return pattern.findall(text)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes Python code string into a list of token IDs."""
        tokens = self._tokenize_regex(text)
        token_ids: List[int] = []

        if add_special_tokens:
            token_ids.append(self.token_to_id[self.BOS_TOKEN])

        for tok in tokens:
            if tok in self.token_to_id:
                token_ids.append(self.token_to_id[tok])
            else:
                for char in tok:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
                    else:
                        token_ids.append(self.token_to_id[self.UNK_TOKEN])

        if add_special_tokens:
            token_ids.append(self.token_to_id[self.EOS_TOKEN])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodes token IDs back into Python code text."""
        decoded_tokens: List[str] = []
        special_ids = {self.token_to_id[t] for t in self.SPECIAL_TOKENS if t in self.token_to_id}

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            token_str = self.id_to_token.get(tid, self.UNK_TOKEN)
            decoded_tokens.append(token_str)

        return "".join(decoded_tokens)

    def save_vocabulary(self, file_path: Optional[str] = None) -> None:
        """Saves current vocabulary to a JSON file."""
        if file_path is None:
            file_path = str(Path(__file__).parent / "vocab.json")
        data = {
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "special_tokens": self.SPECIAL_TOKENS,
            "token_to_id": self.token_to_id,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_vocabulary(cls, file_path: Optional[str] = None) -> "PyGPTTokenizer":
        """Loads a PyGPTTokenizer instance from a JSON vocabulary file."""
        if file_path is None:
            file_path = str(Path(__file__).parent / "vocab.json")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data.get("target_vocab_size", 32000))
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in data["token_to_id"].items()}
        return tokenizer


# Global tokenizer instance
pygpt_tokenizer = PyGPTTokenizer()
pygpt_tokenizer.save_vocabulary()
