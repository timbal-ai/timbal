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
    "GoogleMapsNearbySearch",
    "GoogleMapsPlaceDetails",
    "GoogleMapsTextSearch",
    "GoogleMapsValidateAddress",
    "OutlookArchive",
    "OutlookCreateDraft",
    "OutlookForward",
    "OutlookGetAttachments",
    "OutlookReadEmails",
    "OutlookSend",
    "OutlookTrash",
    "OutlookUpdateEmail",
    "PineconeCreateIndex",
    "PineconeDeleteVectors",
    "PineconeFetchVectors",
    "PineconeIndexStats",
    "PineconeListIndexes",
    "PineconeQuery",
    "PineconeUpsertVectors",
    "Read",
    "TavilyCrawl",
    "TavilyExtract",
    "TavilyMap",
    "TavilySearch",
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
    "GoogleMapsNearbySearch": ".google_maps",
    "GoogleMapsPlaceDetails": ".google_maps",
    "GoogleMapsTextSearch": ".google_maps",
    "GoogleMapsValidateAddress": ".google_maps",
    "OutlookArchive": ".outlook",
    "OutlookCreateDraft": ".outlook",
    "OutlookForward": ".outlook",
    "OutlookGetAttachments": ".outlook",
    "OutlookReadEmails": ".outlook",
    "OutlookSend": ".outlook",
    "OutlookTrash": ".outlook",
    "OutlookUpdateEmail": ".outlook",
    "PineconeCreateIndex": ".pinecone",
    "PineconeDeleteVectors": ".pinecone",
    "PineconeFetchVectors": ".pinecone",
    "PineconeIndexStats": ".pinecone",
    "PineconeListIndexes": ".pinecone",
    "PineconeQuery": ".pinecone",
    "PineconeUpsertVectors": ".pinecone",
    "Read": ".read",
    "TavilyCrawl": ".tavily",
    "TavilyExtract": ".tavily",
    "TavilyMap": ".tavily",
    "TavilySearch": ".tavily",
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
