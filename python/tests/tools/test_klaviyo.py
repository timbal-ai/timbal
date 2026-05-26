"""Live integration smoke tests for Klaviyo tools.

Requires ``KLAVIYO_API_KEY`` in the environment (e.g. repo-root ``.env`` loaded by
``tests/conftest.py``). These tests call the real Klaviyo API and may create profiles,
lists, catalog items, and events in the connected account.

Optional env for campaign create:
- ``KLAVIYO_FROM_EMAIL`` — verified sender email in the Klaviyo account

Run explicitly::

    uv run pytest python/tests/tools/test_klaviyo.py -m integration -v

Default ``uv run pytest`` excludes ``integration`` tests.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from timbal.core.tool import Tool
from timbal.tools.klaviyo import (
    KlaviyoAddProfilesToList,
    KlaviyoCreateCampaign,
    KlaviyoCreateCatalogItem,
    KlaviyoCreateEvent,
    KlaviyoCreateList,
    KlaviyoCreatePlacedOrderEvent,
    KlaviyoCreateProfile,
    KlaviyoGetCatalogItem,
    KlaviyoGetListProfiles,
    KlaviyoGetProfile,
    KlaviyoGetProfileLists,
    KlaviyoGetProfilePredictiveAnalytics,
    KlaviyoGetProfileSubscriptions,
    KlaviyoListCampaigns,
    KlaviyoListCatalogItems,
    KlaviyoListEvents,
    KlaviyoListFlowActions,
    KlaviyoListFlows,
    KlaviyoListLists,
    KlaviyoListMetrics,
    KlaviyoListProfiles,
    KlaviyoListSegments,
    KlaviyoQueryCampaignValues,
    KlaviyoRemoveProfilesFromList,
    KlaviyoSubscribeProfiles,
    KlaviyoTriggerFlowViaEvent,
    KlaviyoUnsubscribeProfiles,
    KlaviyoUpdateCampaign,
    KlaviyoUpdateFlowStatus,
    KlaviyoUpdateProfile,
)


def _skip_if_klaviyo_not_configured() -> None:
    if not os.getenv("KLAVIYO_API_KEY"):
        pytest.skip("Klaviyo integration: set KLAVIYO_API_KEY in the environment or .env")


async def _invoke(tool: Tool, **kwargs: Any) -> Any:
    result = await tool(**kwargs).collect()
    if result.error:
        message = result.error.get("message", result.error) if isinstance(result.error, dict) else result.error
        raise AssertionError(f"{tool.name} failed: {message}")
    return result.output


@pytest.mark.integration
@pytest.mark.asyncio
async def test_klaviyo_live_smoke() -> None:
    """End-to-end smoke test for core and extended Klaviyo tools (options B/C/D)."""
    _skip_if_klaviyo_not_configured()

    # --- read-only endpoints ---
    profiles = await _invoke(KlaviyoListProfiles())
    assert "data" in profiles

    lists = await _invoke(KlaviyoListLists())
    assert "data" in lists

    metrics = await _invoke(KlaviyoListMetrics())
    assert "data" in metrics
    metric_id = metrics["data"][0]["id"]

    segments = await _invoke(KlaviyoListSegments())
    assert "data" in segments

    campaigns = await _invoke(KlaviyoListCampaigns())
    assert "data" in campaigns

    flows = await _invoke(KlaviyoListFlows())
    assert "data" in flows
    flow_id = flows["data"][0]["id"]
    original_flow_status = flows["data"][0]["attributes"].get("status", "draft")

    events = await _invoke(KlaviyoListEvents(), sort="-datetime")
    assert "data" in events

    catalog_items = await _invoke(KlaviyoListCatalogItems())
    assert "data" in catalog_items

    # --- profile write path ---
    test_email = f"timbal-test-{int(time.time())}@example.com"
    created = await _invoke(
        KlaviyoCreateProfile(),
        email=test_email,
        first_name="Timbal",
        last_name="Test",
        properties={"source": "timbal_integration_test"},
    )
    profile_id = created["data"]["id"]

    fetched = await _invoke(KlaviyoGetProfile(), profile_id=profile_id)
    assert fetched["data"]["attributes"]["email"] == test_email

    updated = await _invoke(
        KlaviyoUpdateProfile(),
        profile_id=profile_id,
        first_name="TimbalUpdated",
    )
    assert updated["data"]["attributes"]["first_name"] == "TimbalUpdated"

    event_result = await _invoke(
        KlaviyoCreateEvent(),
        metric_name="Timbal Integration Test",
        profile_email=test_email,
        properties={"test_run": True},
    )
    assert event_result == {"status": "accepted"}

    # --- list membership write path ---
    created_list = await _invoke(
        KlaviyoCreateList(),
        name=f"Timbal Test {int(time.time())}",
    )
    list_id = created_list["data"]["id"]

    add_result = await _invoke(
        KlaviyoAddProfilesToList(),
        list_id=list_id,
        profile_ids=[profile_id],
    )
    assert add_result == {"status": "success"}

    list_profiles = await _invoke(KlaviyoGetListProfiles(), list_id=list_id)
    assert any(p["id"] == profile_id for p in list_profiles["data"])

    profile_lists = await _invoke(KlaviyoGetProfileLists(), profile_id=profile_id)
    assert list_id in [item["id"] for item in profile_lists["data"]]

    # --- Option B: subscriptions ---
    subscribe_job = await _invoke(
        KlaviyoSubscribeProfiles(),
        emails=[test_email],
        list_id=list_id,
        profile_ids=[profile_id],
        custom_source="timbal_integration_test",
    )
    assert subscribe_job.get("status") == "accepted" or "data" in subscribe_job

    subscriptions = await _invoke(KlaviyoGetProfileSubscriptions(), profile_id=profile_id)
    assert "data" in subscriptions

    # Unsubscribe scoped to list_id to avoid global unsubscribe
    unsubscribe_job = await _invoke(
        KlaviyoUnsubscribeProfiles(),
        emails=[test_email],
        list_id=list_id,
    )
    assert unsubscribe_job.get("status") == "accepted" or "data" in unsubscribe_job

    remove_result = await _invoke(
        KlaviyoRemoveProfilesFromList(),
        list_id=list_id,
        profile_ids=[profile_id],
    )
    assert remove_result == {"status": "success"}

    # --- Option C: catalog + placed order + predictive analytics ---
    external_id = f"timbal-item-{int(time.time())}"
    catalog_created = await _invoke(
        KlaviyoCreateCatalogItem(),
        external_id=external_id,
        title="Timbal Test Product",
        description="Integration test catalog item",
        url="https://example.com/products/timbal-test",
        price=9.99,
    )
    assert catalog_created["data"]["attributes"]["external_id"] == external_id

    catalog_fetched = await _invoke(KlaviyoGetCatalogItem(), item_id=external_id)
    assert catalog_fetched["data"]["attributes"]["title"] == "Timbal Test Product"

    placed_order = await _invoke(
        KlaviyoCreatePlacedOrderEvent(),
        profile_email=test_email,
        profile_id=profile_id,
        order_id=f"order-{int(time.time())}",
        value=19.99,
        properties={"ItemNames": ["Timbal Test Product"]},
        unique_id=f"timbal-order-{int(time.time())}",
    )
    assert placed_order == {"status": "accepted"}

    predictive = await _invoke(KlaviyoGetProfilePredictiveAnalytics(), profile_id=profile_id)
    assert "data" in predictive

    # --- Option D: campaigns, reporting, flows ---
    flow_actions = await _invoke(KlaviyoListFlowActions(), flow_id=flow_id)
    assert "data" in flow_actions

    # Restore original status after setting draft (idempotent for draft flows)
    await _invoke(KlaviyoUpdateFlowStatus(), flow_id=flow_id, status="draft")
    if original_flow_status != "draft":
        await _invoke(KlaviyoUpdateFlowStatus(), flow_id=flow_id, status=original_flow_status)

    trigger_event = await _invoke(
        KlaviyoTriggerFlowViaEvent(),
        metric_name="Timbal Integration Test",
        profile_email=test_email,
        profile_id=profile_id,
        properties={"flow_trigger_test": True},
        unique_id=f"timbal-flow-trigger-{int(time.time())}",
    )
    assert trigger_event == {"status": "accepted"}

    campaign_values = await _invoke(
        KlaviyoQueryCampaignValues(),
        conversion_metric_id=metric_id,
        timeframe_key="last_7_days",
        statistics=["opens", "recipients"],
    )
    assert "data" in campaign_values

    if campaigns["data"]:
        existing_campaign_id = campaigns["data"][0]["id"]
        renamed = await _invoke(
            KlaviyoUpdateCampaign(),
            campaign_id=existing_campaign_id,
            name=f"Timbal rename check {int(time.time())}",
        )
        assert "data" in renamed

    from_email = os.getenv("KLAVIYO_FROM_EMAIL")
    if from_email:
        draft_campaign = await _invoke(
            KlaviyoCreateCampaign(),
            name=f"Timbal Draft {int(time.time())}",
            audience_list_ids=[list_id],
            subject="Timbal integration test",
            from_email=from_email,
            from_label="Timbal Test",
        )
        assert draft_campaign["data"]["id"]

    # confirm filter lookup works for the profile we created
    filtered = await _invoke(
        KlaviyoListProfiles(),
        filter=f'equals(email,"{test_email}")',
    )
    assert any(p["id"] == profile_id for p in filtered["data"])
