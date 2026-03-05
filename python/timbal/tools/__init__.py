# pyright: reportUnsupportedDunderAll=false

__all__ = [
    "Bash",
    "CalaSearch",
    "Edit",
    "GmailAddLabel",
    "GmailListLabels",
    "GmailRemoveLabel",
    "GmailReply",
    "GmailSearch",
    "GmailSend",
    "PineconeCreateIndex",
    "PineconeDeleteVectors",
    "PineconeFetchVectors",
    "PineconeIndexStats",
    "PineconeListIndexes",
    "PineconeQuery",
    "PineconeUpsertVectors",
    "Read",
    "WebSearch",
    "Write",
]

_LAZY_IMPORTS = {
    "Bash": ".bash",
    "CalaSearch": ".cala",
    "Edit": ".edit",
    "GmailAddLabel": ".gmail",
    "GmailListLabels": ".gmail",
    "GmailRemoveLabel": ".gmail",
    "GmailReply": ".gmail",
    "GmailSearch": ".gmail",
    "GmailSend": ".gmail",
    "PineconeCreateIndex": ".pinecone",
    "PineconeDeleteVectors": ".pinecone",
    "PineconeFetchVectors": ".pinecone",
    "PineconeIndexStats": ".pinecone",
    "PineconeListIndexes": ".pinecone",
    "PineconeQuery": ".pinecone",
    "PineconeUpsertVectors": ".pinecone",
    "Read": ".read",
    "WebSearch": ".web_search",
    "Write": ".write",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache to bypass __getattr__ on subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
