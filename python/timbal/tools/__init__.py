# pyright: reportUnsupportedDunderAll=false

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bash import Bash
    from .cala import CalaSearch
    from .edit import Edit
    from .firecrawl import FirecrawlCrawl, FirecrawlExtract, FirecrawlMap, FirecrawlScrape, FirecrawlSearch
    from .gmail import GmailAddLabel, GmailListLabels, GmailRemoveLabel, GmailReply, GmailSearch, GmailSend
    from .google_calendar import (
        GoogleCalendarCheckFreeSlots,
        GoogleCalendarCreateEvent,
        GoogleCalendarDeleteEvent,
        GoogleCalendarListEvents,
        GoogleCalendarUpdateAttendeeStatus,
        GoogleCalendarUpdateEvent,
    )
    from .google_docs import GoogleDocsCreate
    from .google_drive import (
        GoogleDriveCreateFile,
        GoogleDriveCreateFolder,
        GoogleDriveCreateSharedDrive,
        GoogleDriveGetDownloadLink,
        GoogleDriveGetFile,
        GoogleDriveSearchFiles,
        GoogleDriveSearchFolders,
        GoogleDriveUploadFile,
    )
    from .google_sheets import (
        GoogleSheetsAddSheet,
        GoogleSheetsAppendValues,
        GoogleSheetsBatchGet,
        GoogleSheetsBatchUpdate,
        GoogleSheetsClearValues,
        GoogleSheetsCopySheet,
        GoogleSheetsCreateSheet,
        GoogleSheetsGetSheetNames,
        GoogleSheetsGetSpreadsheetInfo,
        GoogleSheetsLookupRow,
        GoogleSheetsShareSpreadsheet,
    )
    from .google_maps import (
        GoogleMapsNearbySearch,
        GoogleMapsPlaceDetails,
        GoogleMapsTextSearch,
        GoogleMapsValidateAddress,
    )
    from .outlook import (
        OutlookArchive,
        OutlookCreateDraft,
        OutlookForward,
        OutlookGetAttachments,
        OutlookReadEmails,
        OutlookSend,
        OutlookTrash,
        OutlookUpdateEmail,
    )
    from .pinecone import (
        PineconeCreateIndex,
        PineconeDeleteVectors,
        PineconeFetchVectors,
        PineconeIndexStats,
        PineconeListIndexes,
        PineconeQuery,
        PineconeUpsertVectors,
    )
    from .read import Read
    from .scraperapi import (
        ScraperAPIAmazonProduct,
        ScraperAPIAmazonSearch,
        ScraperAPIAsyncScrape,
        ScraperAPIGoogleSearch,
        ScraperAPIScrape,
    )
    from .tavily import TavilyCrawl, TavilyExtract, TavilyMap, TavilySearch
    from .web_search import WebSearch
    from .write import Write

__all__ = [
    "Bash",
    "CalaSearch",
    "Edit",
    "FirecrawlCrawl",
    "FirecrawlExtract",
    "FirecrawlMap",
    "FirecrawlScrape",
    "FirecrawlSearch",
    "GmailAddLabel",
    "GmailListLabels",
    "GmailRemoveLabel",
    "GmailReply",
    "GmailSearch",
    "GmailSend",
    "GoogleCalendarCheckFreeSlots",
    "GoogleCalendarCreateEvent",
    "GoogleCalendarDeleteEvent",
    "GoogleCalendarListEvents",
    "GoogleCalendarUpdateAttendeeStatus",
    "GoogleCalendarUpdateEvent",
    "GoogleDocsCreate",
    "GoogleDriveCreateFile",
    "GoogleDriveCreateFolder",
    "GoogleDriveCreateSharedDrive",
    "GoogleDriveGetDownloadLink",
    "GoogleDriveGetFile",
    "GoogleDriveSearchFiles",
    "GoogleDriveSearchFolders",
    "GoogleDriveUploadFile",
    "GoogleSheetsAddSheet",
    "GoogleSheetsAppendValues",
    "GoogleSheetsBatchGet",
    "GoogleSheetsBatchUpdate",
    "GoogleSheetsClearValues",
    "GoogleSheetsCopySheet",
    "GoogleSheetsCreateSheet",
    "GoogleSheetsGetSheetNames",
    "GoogleSheetsGetSpreadsheetInfo",
    "GoogleSheetsLookupRow",
    "GoogleSheetsShareSpreadsheet",
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
    "ScraperAPIAmazonProduct",
    "ScraperAPIAmazonSearch",
    "ScraperAPIAsyncScrape",
    "ScraperAPIGoogleSearch",
    "ScraperAPIScrape",
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
    "FirecrawlCrawl": ".firecrawl",
    "FirecrawlExtract": ".firecrawl",
    "FirecrawlMap": ".firecrawl",
    "FirecrawlScrape": ".firecrawl",
    "FirecrawlSearch": ".firecrawl",
    "GmailAddLabel": ".gmail",
    "GmailListLabels": ".gmail",
    "GmailRemoveLabel": ".gmail",
    "GmailReply": ".gmail",
    "GmailSearch": ".gmail",
    "GmailSend": ".gmail",
    "GoogleCalendarCheckFreeSlots": ".google_calendar",
    "GoogleCalendarCreateEvent": ".google_calendar",
    "GoogleCalendarDeleteEvent": ".google_calendar",
    "GoogleCalendarListEvents": ".google_calendar",
    "GoogleCalendarUpdateAttendeeStatus": ".google_calendar",
    "GoogleCalendarUpdateEvent": ".google_calendar",
    "GoogleDocsCreate": ".google_docs",
    "GoogleDriveCreateFile": ".google_drive",
    "GoogleDriveCreateFolder": ".google_drive",
    "GoogleDriveCreateSharedDrive": ".google_drive",
    "GoogleDriveGetDownloadLink": ".google_drive",
    "GoogleDriveGetFile": ".google_drive",
    "GoogleDriveSearchFiles": ".google_drive",
    "GoogleDriveSearchFolders": ".google_drive",
    "GoogleDriveUploadFile": ".google_drive",
    "GoogleSheetsAddSheet": ".google_sheets",
    "GoogleSheetsAppendValues": ".google_sheets",
    "GoogleSheetsBatchGet": ".google_sheets",
    "GoogleSheetsBatchUpdate": ".google_sheets",
    "GoogleSheetsClearValues": ".google_sheets",
    "GoogleSheetsCopySheet": ".google_sheets",
    "GoogleSheetsCreateSheet": ".google_sheets",
    "GoogleSheetsGetSheetNames": ".google_sheets",
    "GoogleSheetsGetSpreadsheetInfo": ".google_sheets",
    "GoogleSheetsLookupRow": ".google_sheets",
    "GoogleSheetsShareSpreadsheet": ".google_sheets",
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
    "ScraperAPIAmazonProduct": ".scraperapi",
    "ScraperAPIAmazonSearch": ".scraperapi",
    "ScraperAPIAsyncScrape": ".scraperapi",
    "ScraperAPIGoogleSearch": ".scraperapi",
    "ScraperAPIScrape": ".scraperapi",
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
