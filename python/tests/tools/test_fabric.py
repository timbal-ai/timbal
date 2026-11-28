"""Unit tests for Microsoft Fabric tools (mocked, no network).

``_ROUTES`` pins the HTTP method, path and fixed query of every tool. It was generated from, and
checked against, the official Fabric OpenAPI specs (microsoft/fabric-rest-api-specs).

Live smoke tests are marked ``integration`` and need ``FABRIC_ACCESS_TOKEN`` (Microsoft Entra
bearer token for https://api.fabric.microsoft.com) and optionally ``FABRIC_WORKSPACE_ID``::

    uv run pytest python/tests/tools/test_fabric.py -m integration -v
"""

import base64
import inspect
import json
import os
import typing
from types import SimpleNamespace
from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr
from timbal.codegen.tool_discovery import get_framework_tools
from timbal.errors import CredentialNotAvailable
from timbal.platform.integrations import Integration
from timbal.tools import (
    FabricCreateItem,
    FabricCreateOrUpdateApacheAirflowJobFile,
    FabricDeployApacheAirflowJobRequirements,
    FabricGetItemShortcut,
    FabricGetOperationState,
    FabricListItems,
    FabricListWorkspaces,
    FabricRunItemJob,
    FabricSetWarehouseAuditActionsAndGroups,
)
from timbal.tools import fabric as fabric_module
from timbal.tools.fabric import (
    _clean_params,
    _get_token_from_client_credentials,
    _parse_response,
    _raise_for_fabric,
    _resolve_token,
    _service_principal_parts,
    _upload_bytes,
)

FABRIC_TOOL_COUNT = 395
BASE_URL = "https://api.fabric.microsoft.com/v1"

# (tool name, HTTP method, path template with python parameter names, fixed query or None)
_ROUTES: list[tuple[str, str, str, dict[str, str] | None]] = [
    ("fabric_list_capacities", "GET", "/capacities", None),
    ("fabric_get_capacity", "GET", "/capacities/{capacity_id}", None),
    ("fabric_get_capacity_surge_protection", "GET", "/capacities/{capacity_id}/surgeProtection", None),
    ("fabric_update_capacity_surge_protection", "PATCH", "/capacities/{capacity_id}/surgeProtection", None),
    ("fabric_search_catalog", "POST", "/catalog/search", None),
    ("fabric_list_connections", "GET", "/connections", None),
    ("fabric_create_connection", "POST", "/connections", None),
    ("fabric_list_supported_connection_types", "GET", "/connections/supportedConnectionTypes", None),
    ("fabric_delete_connection", "DELETE", "/connections/{connection_id}", None),
    ("fabric_get_connection", "GET", "/connections/{connection_id}", None),
    ("fabric_update_connection", "PATCH", "/connections/{connection_id}", None),
    ("fabric_list_connection_role_assignments", "GET", "/connections/{connection_id}/roleAssignments", None),
    ("fabric_add_connection_role_assignment", "POST", "/connections/{connection_id}/roleAssignments", None),
    (
        "fabric_delete_connection_role_assignment",
        "DELETE",
        "/connections/{connection_id}/roleAssignments/{connection_role_assignment_id}",
        None,
    ),
    (
        "fabric_get_connection_role_assignment",
        "GET",
        "/connections/{connection_id}/roleAssignments/{connection_role_assignment_id}",
        None,
    ),
    (
        "fabric_update_connection_role_assignment",
        "PATCH",
        "/connections/{connection_id}/roleAssignments/{connection_role_assignment_id}",
        None,
    ),
    ("fabric_test_connection", "POST", "/connections/{connection_id}/testConnection", None),
    ("fabric_list_deployment_pipelines", "GET", "/deploymentPipelines", None),
    ("fabric_create_deployment_pipeline", "POST", "/deploymentPipelines", None),
    ("fabric_delete_deployment_pipeline", "DELETE", "/deploymentPipelines/{deployment_pipeline_id}", None),
    ("fabric_get_deployment_pipeline", "GET", "/deploymentPipelines/{deployment_pipeline_id}", None),
    ("fabric_update_deployment_pipeline", "PATCH", "/deploymentPipelines/{deployment_pipeline_id}", None),
    (
        "fabric_deploy_deployment_pipeline_stage_content",
        "POST",
        "/deploymentPipelines/{deployment_pipeline_id}/deploy",
        None,
    ),
    (
        "fabric_list_deployment_pipeline_operations",
        "GET",
        "/deploymentPipelines/{deployment_pipeline_id}/operations",
        None,
    ),
    (
        "fabric_get_deployment_pipeline_operation",
        "GET",
        "/deploymentPipelines/{deployment_pipeline_id}/operations/{operation_id}",
        None,
    ),
    (
        "fabric_list_deployment_pipeline_role_assignments",
        "GET",
        "/deploymentPipelines/{deployment_pipeline_id}/roleAssignments",
        None,
    ),
    (
        "fabric_add_deployment_pipeline_role_assignment",
        "POST",
        "/deploymentPipelines/{deployment_pipeline_id}/roleAssignments",
        None,
    ),
    (
        "fabric_delete_deployment_pipeline_role_assignment",
        "DELETE",
        "/deploymentPipelines/{deployment_pipeline_id}/roleAssignments/{principal_id}",
        None,
    ),
    ("fabric_list_deployment_pipeline_stages", "GET", "/deploymentPipelines/{deployment_pipeline_id}/stages", None),
    (
        "fabric_get_deployment_pipeline_stage",
        "GET",
        "/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}",
        None,
    ),
    (
        "fabric_update_deployment_pipeline_stage",
        "PATCH",
        "/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}",
        None,
    ),
    (
        "fabric_assign_deployment_pipeline_workspace_to_stage",
        "POST",
        "/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}/assignWorkspace",
        None,
    ),
    (
        "fabric_list_deployment_pipeline_stage_items",
        "GET",
        "/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}/items",
        None,
    ),
    (
        "fabric_unassign_deployment_pipeline_workspace_from_stage",
        "POST",
        "/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}/unassignWorkspace",
        None,
    ),
    ("fabric_list_domains", "GET", "/domains", None),
    ("fabric_get_domain", "GET", "/domains/{domain_id}", None),
    (
        "fabric_get_external_data_share_invitation_details",
        "GET",
        "/externalDataShares/invitations/{invitation_id}",
        None,
    ),
    (
        "fabric_accept_external_data_share_invitation",
        "POST",
        "/externalDataShares/invitations/{invitation_id}/accept",
        None,
    ),
    ("fabric_list_gateways", "GET", "/gateways", None),
    ("fabric_create_gateway", "POST", "/gateways", None),
    ("fabric_delete_gateway", "DELETE", "/gateways/{gateway_id}", None),
    ("fabric_get_gateway", "GET", "/gateways/{gateway_id}", None),
    ("fabric_update_gateway", "PATCH", "/gateways/{gateway_id}", None),
    ("fabric_check_gateway_status", "POST", "/gateways/{gateway_id}/checkStatus", None),
    ("fabric_list_gateway_members", "GET", "/gateways/{gateway_id}/members", None),
    ("fabric_delete_gateway_member", "DELETE", "/gateways/{gateway_id}/members/{gateway_member_id}", None),
    ("fabric_update_gateway_member", "PATCH", "/gateways/{gateway_id}/members/{gateway_member_id}", None),
    (
        "fabric_check_gateway_member_status",
        "POST",
        "/gateways/{gateway_id}/members/{gateway_member_id}/checkStatus",
        None,
    ),
    ("fabric_restart_gateway", "POST", "/gateways/{gateway_id}/restart", None),
    ("fabric_list_gateway_role_assignments", "GET", "/gateways/{gateway_id}/roleAssignments", None),
    ("fabric_add_gateway_role_assignment", "POST", "/gateways/{gateway_id}/roleAssignments", None),
    (
        "fabric_delete_gateway_role_assignment",
        "DELETE",
        "/gateways/{gateway_id}/roleAssignments/{gateway_role_assignment_id}",
        None,
    ),
    (
        "fabric_get_gateway_role_assignment",
        "GET",
        "/gateways/{gateway_id}/roleAssignments/{gateway_role_assignment_id}",
        None,
    ),
    (
        "fabric_update_gateway_role_assignment",
        "PATCH",
        "/gateways/{gateway_id}/roleAssignments/{gateway_role_assignment_id}",
        None,
    ),
    ("fabric_shutdown_gateway", "POST", "/gateways/{gateway_id}/shutdown", None),
    ("fabric_get_operation_state", "GET", "/operations/{operation_id}", None),
    ("fabric_get_operation_result", "GET", "/operations/{operation_id}/result", None),
    ("fabric_list_tags", "GET", "/tags", None),
    ("fabric_list_workspaces", "GET", "/workspaces", None),
    ("fabric_create_workspace", "POST", "/workspaces", None),
    ("fabric_delete_workspace", "DELETE", "/workspaces/{workspace_id}", None),
    ("fabric_get_workspace", "GET", "/workspaces/{workspace_id}", None),
    ("fabric_update_workspace", "PATCH", "/workspaces/{workspace_id}", None),
    ("fabric_apply_workspace_tags", "POST", "/workspaces/{workspace_id}/applyTags", None),
    ("fabric_assign_workspace_to_capacity", "POST", "/workspaces/{workspace_id}/assignToCapacity", None),
    ("fabric_assign_workspace_to_domain", "POST", "/workspaces/{workspace_id}/assignToDomain", None),
    ("fabric_deprovision_workspace_identity", "POST", "/workspaces/{workspace_id}/deprovisionIdentity", None),
    ("fabric_get_workspace_encryption", "GET", "/workspaces/{workspace_id}/encryption", None),
    ("fabric_assign_workspace_encryption", "POST", "/workspaces/{workspace_id}/encryption/assign", None),
    ("fabric_reset_workspace_encryption", "POST", "/workspaces/{workspace_id}/encryption/reset", None),
    ("fabric_list_folders", "GET", "/workspaces/{workspace_id}/folders", None),
    ("fabric_create_folder", "POST", "/workspaces/{workspace_id}/folders", None),
    ("fabric_delete_folder", "DELETE", "/workspaces/{workspace_id}/folders/{folder_id}", None),
    ("fabric_get_folder", "GET", "/workspaces/{workspace_id}/folders/{folder_id}", None),
    ("fabric_update_folder", "PATCH", "/workspaces/{workspace_id}/folders/{folder_id}", None),
    ("fabric_move_folder", "POST", "/workspaces/{workspace_id}/folders/{folder_id}/move", None),
    ("fabric_git_commit", "POST", "/workspaces/{workspace_id}/git/commitToGit", None),
    ("fabric_git_connect", "POST", "/workspaces/{workspace_id}/git/connect", None),
    ("fabric_git_get_connection", "GET", "/workspaces/{workspace_id}/git/connection", None),
    ("fabric_git_disconnect", "POST", "/workspaces/{workspace_id}/git/disconnect", None),
    ("fabric_git_initialize_connection", "POST", "/workspaces/{workspace_id}/git/initializeConnection", None),
    ("fabric_git_get_my_credentials", "GET", "/workspaces/{workspace_id}/git/myGitCredentials", None),
    ("fabric_git_update_my_credentials", "PATCH", "/workspaces/{workspace_id}/git/myGitCredentials", None),
    ("fabric_git_get_status", "GET", "/workspaces/{workspace_id}/git/status", None),
    ("fabric_git_update_from_git", "POST", "/workspaces/{workspace_id}/git/updateFromGit", None),
    ("fabric_git_list_workspace_relations", "GET", "/workspaces/{workspace_id}/git/workspaceRelations", None),
    ("fabric_git_create_workspace_relation", "POST", "/workspaces/{workspace_id}/git/workspaceRelations", None),
    (
        "fabric_git_delete_workspace_relation",
        "DELETE",
        "/workspaces/{workspace_id}/git/workspaceRelations/{workspace_relation_id}",
        None,
    ),
    ("fabric_list_items", "GET", "/workspaces/{workspace_id}/items", None),
    ("fabric_create_item", "POST", "/workspaces/{workspace_id}/items", None),
    ("fabric_bulk_export_item_definitions", "POST", "/workspaces/{workspace_id}/items/bulkExportDefinitions", None),
    ("fabric_bulk_import_item_definitions", "POST", "/workspaces/{workspace_id}/items/bulkImportDefinitions", None),
    ("fabric_bulk_move_items", "POST", "/workspaces/{workspace_id}/items/bulkMove", None),
    ("fabric_delete_item", "DELETE", "/workspaces/{workspace_id}/items/{item_id}", None),
    ("fabric_get_item", "GET", "/workspaces/{workspace_id}/items/{item_id}", None),
    ("fabric_update_item", "PATCH", "/workspaces/{workspace_id}/items/{item_id}", None),
    ("fabric_apply_item_tags", "POST", "/workspaces/{workspace_id}/items/{item_id}/applyTags", None),
    ("fabric_list_item_connections", "GET", "/workspaces/{workspace_id}/items/{item_id}/connections", None),
    ("fabric_list_item_data_access_roles", "GET", "/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles", None),
    ("fabric_upsert_item_data_access_role", "POST", "/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles", None),
    ("fabric_set_item_data_access_roles", "PUT", "/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles", None),
    (
        "fabric_delete_item_data_access_role",
        "DELETE",
        "/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles/{role_name}",
        None,
    ),
    (
        "fabric_get_item_data_access_role",
        "GET",
        "/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles/{role_name}",
        None,
    ),
    (
        "fabric_list_item_external_data_shares",
        "GET",
        "/workspaces/{workspace_id}/items/{item_id}/externalDataShares",
        None,
    ),
    (
        "fabric_create_item_external_data_share",
        "POST",
        "/workspaces/{workspace_id}/items/{item_id}/externalDataShares",
        None,
    ),
    (
        "fabric_delete_item_external_data_share",
        "DELETE",
        "/workspaces/{workspace_id}/items/{item_id}/externalDataShares/{external_data_share_id}",
        None,
    ),
    (
        "fabric_get_item_external_data_share",
        "GET",
        "/workspaces/{workspace_id}/items/{item_id}/externalDataShares/{external_data_share_id}",
        None,
    ),
    (
        "fabric_revoke_item_external_data_share",
        "POST",
        "/workspaces/{workspace_id}/items/{item_id}/externalDataShares/{external_data_share_id}/revoke",
        None,
    ),
    ("fabric_get_item_definition", "POST", "/workspaces/{workspace_id}/items/{item_id}/getDefinition", None),
    (
        "fabric_assign_item_default_identity",
        "POST",
        "/workspaces/{workspace_id}/items/{item_id}/identities/default/assign",
        {"beta": "true"},
    ),
    ("fabric_list_item_job_instances", "GET", "/workspaces/{workspace_id}/items/{item_id}/jobs/instances", None),
    (
        "fabric_get_item_job_instance",
        "GET",
        "/workspaces/{workspace_id}/items/{item_id}/jobs/instances/{job_instance_id}",
        None,
    ),
    (
        "fabric_cancel_item_job_instance",
        "POST",
        "/workspaces/{workspace_id}/items/{item_id}/jobs/instances/{job_instance_id}/cancel",
        None,
    ),
    ("fabric_run_item_job", "POST", "/workspaces/{workspace_id}/items/{item_id}/jobs/{job_type}/instances", None),
    ("fabric_list_item_schedules", "GET", "/workspaces/{workspace_id}/items/{item_id}/jobs/{job_type}/schedules", None),
    (
        "fabric_create_item_schedule",
        "POST",
        "/workspaces/{workspace_id}/items/{item_id}/jobs/{job_type}/schedules",
        None,
    ),
    (
        "fabric_delete_item_schedule",
        "DELETE",
        "/workspaces/{workspace_id}/items/{item_id}/jobs/{job_type}/schedules/{schedule_id}",
        None,
    ),
    (
        "fabric_get_item_schedule",
        "GET",
        "/workspaces/{workspace_id}/items/{item_id}/jobs/{job_type}/schedules/{schedule_id}",
        None,
    ),
    (
        "fabric_update_item_schedule",
        "PATCH",
        "/workspaces/{workspace_id}/items/{item_id}/jobs/{job_type}/schedules/{schedule_id}",
        None,
    ),
    ("fabric_update_item_logical_id", "PATCH", "/workspaces/{workspace_id}/items/{item_id}/logicalId", None),
    ("fabric_move_item", "POST", "/workspaces/{workspace_id}/items/{item_id}/move", None),
    (
        "fabric_get_item_downstream_relations",
        "GET",
        "/workspaces/{workspace_id}/items/{item_id}/relations/downstream",
        {"beta": "true"},
    ),
    (
        "fabric_get_item_upstream_relations",
        "GET",
        "/workspaces/{workspace_id}/items/{item_id}/relations/upstream",
        {"beta": "true"},
    ),
    ("fabric_list_item_shortcuts", "GET", "/workspaces/{workspace_id}/items/{item_id}/shortcuts", None),
    ("fabric_create_item_shortcut", "POST", "/workspaces/{workspace_id}/items/{item_id}/shortcuts", None),
    (
        "fabric_bulk_create_item_shortcuts",
        "POST",
        "/workspaces/{workspace_id}/items/{item_id}/shortcuts/bulkCreate",
        None,
    ),
    (
        "fabric_delete_item_shortcut",
        "DELETE",
        "/workspaces/{workspace_id}/items/{item_id}/shortcuts/{shortcut_path}/{shortcut_name}",
        None,
    ),
    (
        "fabric_get_item_shortcut",
        "GET",
        "/workspaces/{workspace_id}/items/{item_id}/shortcuts/{shortcut_path}/{shortcut_name}",
        None,
    ),
    ("fabric_unapply_item_tags", "POST", "/workspaces/{workspace_id}/items/{item_id}/unapplyTags", None),
    ("fabric_update_item_definition", "POST", "/workspaces/{workspace_id}/items/{item_id}/updateDefinition", None),
    (
        "fabric_list_workspace_managed_private_endpoints",
        "GET",
        "/workspaces/{workspace_id}/managedPrivateEndpoints",
        None,
    ),
    (
        "fabric_create_workspace_managed_private_endpoint",
        "POST",
        "/workspaces/{workspace_id}/managedPrivateEndpoints",
        None,
    ),
    (
        "fabric_delete_workspace_managed_private_endpoint",
        "DELETE",
        "/workspaces/{workspace_id}/managedPrivateEndpoints/{managed_private_endpoint_id}",
        None,
    ),
    (
        "fabric_get_workspace_managed_private_endpoint",
        "GET",
        "/workspaces/{workspace_id}/managedPrivateEndpoints/{managed_private_endpoint_id}",
        None,
    ),
    (
        "fabric_get_workspace_communication_policy",
        "GET",
        "/workspaces/{workspace_id}/networking/communicationPolicy",
        None,
    ),
    (
        "fabric_set_workspace_communication_policy",
        "PUT",
        "/workspaces/{workspace_id}/networking/communicationPolicy",
        None,
    ),
    (
        "fabric_get_workspace_inbound_azure_resource_rules",
        "GET",
        "/workspaces/{workspace_id}/networking/communicationPolicy/inbound/azureResources",
        None,
    ),
    (
        "fabric_set_workspace_inbound_azure_resource_rules",
        "PUT",
        "/workspaces/{workspace_id}/networking/communicationPolicy/inbound/azureResources",
        None,
    ),
    (
        "fabric_get_workspace_inbound_external_data_shares_policy",
        "GET",
        "/workspaces/{workspace_id}/networking/communicationPolicy/inbound/externalDataShares",
        None,
    ),
    (
        "fabric_set_workspace_inbound_external_data_shares_policy",
        "PUT",
        "/workspaces/{workspace_id}/networking/communicationPolicy/inbound/externalDataShares",
        None,
    ),
    (
        "fabric_get_workspace_firewall_rules",
        "GET",
        "/workspaces/{workspace_id}/networking/communicationPolicy/inbound/firewall",
        None,
    ),
    (
        "fabric_set_workspace_firewall_rules",
        "PUT",
        "/workspaces/{workspace_id}/networking/communicationPolicy/inbound/firewall",
        None,
    ),
    (
        "fabric_get_workspace_outbound_cloud_connection_rules",
        "GET",
        "/workspaces/{workspace_id}/networking/communicationPolicy/outbound/connections",
        None,
    ),
    (
        "fabric_set_workspace_outbound_cloud_connection_rules",
        "PUT",
        "/workspaces/{workspace_id}/networking/communicationPolicy/outbound/connections",
        None,
    ),
    (
        "fabric_get_workspace_outbound_gateway_rules",
        "GET",
        "/workspaces/{workspace_id}/networking/communicationPolicy/outbound/gateways",
        None,
    ),
    (
        "fabric_set_workspace_outbound_gateway_rules",
        "PUT",
        "/workspaces/{workspace_id}/networking/communicationPolicy/outbound/gateways",
        None,
    ),
    (
        "fabric_get_workspace_outbound_git_policy",
        "GET",
        "/workspaces/{workspace_id}/networking/communicationPolicy/outbound/git",
        None,
    ),
    (
        "fabric_set_workspace_outbound_git_policy",
        "PUT",
        "/workspaces/{workspace_id}/networking/communicationPolicy/outbound/git",
        None,
    ),
    (
        "fabric_export_onelake_lifecycle_policy",
        "POST",
        "/workspaces/{workspace_id}/onelake/lifecycle/exportPolicy",
        None,
    ),
    (
        "fabric_import_onelake_lifecycle_policy",
        "POST",
        "/workspaces/{workspace_id}/onelake/lifecycle/importPolicy",
        None,
    ),
    ("fabric_reset_onelake_shortcut_cache", "POST", "/workspaces/{workspace_id}/onelake/resetShortcutCache", None),
    ("fabric_get_onelake_settings", "GET", "/workspaces/{workspace_id}/onelake/settings", None),
    (
        "fabric_modify_onelake_settings_access_time_tracking",
        "POST",
        "/workspaces/{workspace_id}/onelake/settings/modifyAccessTimeTracking",
        None,
    ),
    (
        "fabric_modify_onelake_settings_default_tier",
        "POST",
        "/workspaces/{workspace_id}/onelake/settings/modifyDefaultTier",
        None,
    ),
    (
        "fabric_modify_onelake_settings_diagnostics",
        "POST",
        "/workspaces/{workspace_id}/onelake/settings/modifyDiagnostics",
        None,
    ),
    (
        "fabric_modify_onelake_settings_immutability_policy",
        "POST",
        "/workspaces/{workspace_id}/onelake/settings/modifyImmutabilityPolicy",
        None,
    ),
    ("fabric_provision_workspace_identity", "POST", "/workspaces/{workspace_id}/provisionIdentity", None),
    ("fabric_list_recoverable_items", "GET", "/workspaces/{workspace_id}/recoverableItems", None),
    ("fabric_delete_recoverable_item", "DELETE", "/workspaces/{workspace_id}/recoverableItems/{item_id}", None),
    ("fabric_recover_recoverable_item", "POST", "/workspaces/{workspace_id}/recoverableItems/{item_id}/recover", None),
    ("fabric_list_workspace_role_assignments", "GET", "/workspaces/{workspace_id}/roleAssignments", None),
    ("fabric_add_workspace_role_assignment", "POST", "/workspaces/{workspace_id}/roleAssignments", None),
    (
        "fabric_delete_workspace_role_assignment",
        "DELETE",
        "/workspaces/{workspace_id}/roleAssignments/{workspace_role_assignment_id}",
        None,
    ),
    (
        "fabric_get_workspace_role_assignment",
        "GET",
        "/workspaces/{workspace_id}/roleAssignments/{workspace_role_assignment_id}",
        None,
    ),
    (
        "fabric_update_workspace_role_assignment",
        "PATCH",
        "/workspaces/{workspace_id}/roleAssignments/{workspace_role_assignment_id}",
        None,
    ),
    ("fabric_unapply_workspace_tags", "POST", "/workspaces/{workspace_id}/unapplyTags", None),
    ("fabric_unassign_workspace_from_capacity", "POST", "/workspaces/{workspace_id}/unassignFromCapacity", None),
    ("fabric_unassign_workspace_from_domain", "POST", "/workspaces/{workspace_id}/unassignFromDomain", None),
    (
        "fabric_admin_list_capacities_tenant_setting_overrides",
        "GET",
        "/admin/capacities/delegatedTenantSettingOverrides",
        None,
    ),
    (
        "fabric_admin_list_capacity_tenant_setting_overrides",
        "GET",
        "/admin/capacities/{capacity_id}/delegatedTenantSettingOverrides",
        None,
    ),
    (
        "fabric_admin_delete_capacity_tenant_setting_override",
        "DELETE",
        "/admin/capacities/{capacity_id}/delegatedTenantSettingOverrides/{tenant_setting_name}",
        None,
    ),
    (
        "fabric_admin_update_capacity_tenant_setting_override",
        "POST",
        "/admin/capacities/{capacity_id}/delegatedTenantSettingOverrides/{tenant_setting_name}/update",
        None,
    ),
    ("fabric_admin_list_domains", "GET", "/admin/domains", {"preview": "false"}),
    ("fabric_admin_create_domain", "POST", "/admin/domains", {"preview": "false"}),
    (
        "fabric_admin_list_domains_tenant_setting_overrides",
        "GET",
        "/admin/domains/delegatedTenantSettingOverrides",
        None,
    ),
    ("fabric_admin_delete_domain", "DELETE", "/admin/domains/{domain_id}", None),
    ("fabric_admin_get_domain", "GET", "/admin/domains/{domain_id}", {"preview": "false"}),
    ("fabric_admin_update_domain", "PATCH", "/admin/domains/{domain_id}", {"preview": "false"}),
    ("fabric_admin_assign_domain_workspaces_by_ids", "POST", "/admin/domains/{domain_id}/assignWorkspaces", None),
    (
        "fabric_admin_assign_domain_workspaces_by_capacities",
        "POST",
        "/admin/domains/{domain_id}/assignWorkspacesByCapacities",
        None,
    ),
    (
        "fabric_admin_assign_domain_workspaces_by_principals",
        "POST",
        "/admin/domains/{domain_id}/assignWorkspacesByPrincipals",
        None,
    ),
    ("fabric_admin_list_domain_role_assignments", "GET", "/admin/domains/{domain_id}/roleAssignments", None),
    (
        "fabric_admin_bulk_assign_domain_role_assignments",
        "POST",
        "/admin/domains/{domain_id}/roleAssignments/bulkAssign",
        None,
    ),
    (
        "fabric_admin_bulk_unassign_domain_role_assignments",
        "POST",
        "/admin/domains/{domain_id}/roleAssignments/bulkUnassign",
        None,
    ),
    (
        "fabric_admin_sync_domain_role_assignments_to_subdomains",
        "POST",
        "/admin/domains/{domain_id}/roleAssignments/syncToSubdomains",
        None,
    ),
    ("fabric_admin_unassign_all_domain_workspaces", "POST", "/admin/domains/{domain_id}/unassignAllWorkspaces", None),
    ("fabric_admin_unassign_domain_workspaces_by_ids", "POST", "/admin/domains/{domain_id}/unassignWorkspaces", None),
    ("fabric_admin_list_domain_workspaces", "GET", "/admin/domains/{domain_id}/workspaces", None),
    ("fabric_admin_list_items", "GET", "/admin/items", None),
    ("fabric_admin_bulk_remove_item_labels", "POST", "/admin/items/bulkRemoveLabels", None),
    ("fabric_admin_bulk_remove_item_sharing_links", "POST", "/admin/items/bulkRemoveSharingLinks", None),
    ("fabric_admin_bulk_set_item_labels", "POST", "/admin/items/bulkSetLabels", None),
    ("fabric_admin_list_external_data_shares", "GET", "/admin/items/externalDataShares", None),
    ("fabric_admin_remove_all_item_sharing_links", "POST", "/admin/items/removeAllSharingLinks", None),
    ("fabric_admin_list_tags", "GET", "/admin/tags", None),
    ("fabric_admin_bulk_create_tags", "POST", "/admin/tags/bulkCreateTags", None),
    ("fabric_admin_delete_tag", "DELETE", "/admin/tags/{tag_id}", None),
    ("fabric_admin_update_tag", "PATCH", "/admin/tags/{tag_id}", None),
    ("fabric_admin_list_tenant_settings", "GET", "/admin/tenantsettings", None),
    ("fabric_admin_update_tenant_setting", "POST", "/admin/tenantsettings/{tenant_setting_name}/update", None),
    ("fabric_admin_list_user_access_entities", "GET", "/admin/users/{user_id}/access", None),
    ("fabric_admin_list_workloads", "GET", "/admin/workloads", None),
    ("fabric_admin_list_workload_assignments", "GET", "/admin/workloads/assignments", None),
    ("fabric_admin_create_workload_assignment", "POST", "/admin/workloads/assignments", None),
    ("fabric_admin_delete_workload_assignment", "DELETE", "/admin/workloads/assignments/{assignment_id}", None),
    ("fabric_admin_list_workspaces", "GET", "/admin/workspaces", None),
    (
        "fabric_admin_list_workspaces_tenant_setting_overrides",
        "GET",
        "/admin/workspaces/delegatedTenantSettingOverrides",
        None,
    ),
    ("fabric_admin_list_workspace_git_connections", "GET", "/admin/workspaces/discoverGitConnections", None),
    (
        "fabric_admin_list_workspace_communication_policies",
        "GET",
        "/admin/workspaces/networking/communicationpolicies",
        None,
    ),
    ("fabric_admin_get_workspace", "GET", "/admin/workspaces/{workspace_id}", None),
    (
        "fabric_admin_grant_workspace_temporary_access",
        "POST",
        "/admin/workspaces/{workspace_id}/grantAdminTemporaryAccess",
        None,
    ),
    ("fabric_admin_get_item", "GET", "/admin/workspaces/{workspace_id}/items/{item_id}", None),
    (
        "fabric_admin_revoke_item_external_data_share",
        "POST",
        "/admin/workspaces/{workspace_id}/items/{item_id}/externalDataShares/{external_data_share_id}/revoke",
        None,
    ),
    ("fabric_admin_list_item_access_details", "GET", "/admin/workspaces/{workspace_id}/items/{item_id}/users", None),
    (
        "fabric_admin_remove_workspace_temporary_access",
        "POST",
        "/admin/workspaces/{workspace_id}/removeAdminTemporaryAccess",
        None,
    ),
    ("fabric_admin_restore_workspace", "POST", "/admin/workspaces/{workspace_id}/restore", None),
    ("fabric_admin_list_workspace_access_details", "GET", "/admin/workspaces/{workspace_id}/users", None),
    (
        "fabric_list_airflow_pool_templates",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/poolTemplates",
        {"beta": "true"},
    ),
    (
        "fabric_create_airflow_pool_template",
        "POST",
        "/workspaces/{workspace_id}/apacheAirflowJobs/poolTemplates",
        {"beta": "true"},
    ),
    (
        "fabric_delete_airflow_pool_template",
        "DELETE",
        "/workspaces/{workspace_id}/apacheAirflowJobs/poolTemplates/{pool_template_id}",
        {"beta": "true"},
    ),
    (
        "fabric_get_airflow_pool_template",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/poolTemplates/{pool_template_id}",
        {"beta": "true"},
    ),
    (
        "fabric_get_airflow_workspace_settings",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/settings",
        {"beta": "true"},
    ),
    (
        "fabric_update_airflow_workspace_settings",
        "PATCH",
        "/workspaces/{workspace_id}/apacheAirflowJobs/settings",
        {"beta": "true"},
    ),
    (
        "fabric_get_apache_airflow_job_environment",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment",
        {"beta": "true"},
    ),
    (
        "fabric_get_apache_airflow_job_compute",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/compute",
        {"beta": "true"},
    ),
    (
        "fabric_deploy_apache_airflow_job_requirements",
        "POST",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/deployRequirements",
        {"beta": "true"},
    ),
    (
        "fabric_list_apache_airflow_job_libraries",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/libraries",
        {"beta": "true"},
    ),
    (
        "fabric_get_apache_airflow_job_settings",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/settings",
        {"beta": "true"},
    ),
    (
        "fabric_start_apache_airflow_job_environment",
        "POST",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/start",
        {"beta": "true"},
    ),
    (
        "fabric_stop_apache_airflow_job_environment",
        "POST",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/stop",
        {"beta": "true"},
    ),
    (
        "fabric_update_apache_airflow_job_compute",
        "POST",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/updateCompute",
        {"beta": "true"},
    ),
    (
        "fabric_update_apache_airflow_job_settings",
        "POST",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/updateSettings",
        {"beta": "true"},
    ),
    (
        "fabric_list_apache_airflow_job_files",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/files",
        {"beta": "true"},
    ),
    (
        "fabric_delete_apache_airflow_job_file",
        "DELETE",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/files/{file_path}",
        {"beta": "true"},
    ),
    (
        "fabric_get_apache_airflow_job_file",
        "GET",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/files/{file_path}",
        {"beta": "true"},
    ),
    (
        "fabric_create_or_update_apache_airflow_job_file",
        "PUT",
        "/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/files/{file_path}",
        {"beta": "true"},
    ),
    ("fabric_reset_copy_job", "POST", "/workspaces/{workspace_id}/copyJobs/{copy_job_id}/resetCopyJob", None),
    (
        "fabric_list_data_agent_datasources",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources",
        None,
    ),
    (
        "fabric_get_data_agent_datasource",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources/{datasource_id}",
        None,
    ),
    (
        "fabric_list_data_agent_datasource_elements",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources/{datasource_id}/elements",
        None,
    ),
    (
        "fabric_list_data_agent_datasource_fewshots",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources/{datasource_id}/fewshots",
        None,
    ),
    (
        "fabric_get_data_agent_datasource_fewshot",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources/{datasource_id}/fewshots/{few_shot_id}",
        None,
    ),
    ("fabric_get_data_agent_settings", "GET", "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/settings", None),
    (
        "fabric_list_data_agent_staging_datasources",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources",
        None,
    ),
    (
        "fabric_create_data_agent_staging_datasource",
        "POST",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources",
        None,
    ),
    (
        "fabric_delete_data_agent_staging_datasource",
        "DELETE",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}",
        None,
    ),
    (
        "fabric_get_data_agent_staging_datasource",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}",
        None,
    ),
    (
        "fabric_update_data_agent_staging_datasource",
        "PATCH",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}",
        None,
    ),
    (
        "fabric_delete_data_agent_staging_datasource_element",
        "DELETE",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/elements",
        None,
    ),
    (
        "fabric_list_data_agent_staging_datasource_elements",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/elements",
        None,
    ),
    (
        "fabric_update_data_agent_staging_datasource_element",
        "PATCH",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/elements",
        None,
    ),
    (
        "fabric_list_data_agent_staging_datasource_fewshots",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots",
        None,
    ),
    (
        "fabric_create_data_agent_staging_datasource_fewshot",
        "POST",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots",
        None,
    ),
    (
        "fabric_delete_data_agent_staging_all_datasource_fewshots",
        "POST",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots/deleteAll",
        None,
    ),
    (
        "fabric_delete_data_agent_staging_datasource_fewshot",
        "DELETE",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots/{few_shot_id}",
        None,
    ),
    (
        "fabric_get_data_agent_staging_datasource_fewshot",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots/{few_shot_id}",
        None,
    ),
    (
        "fabric_update_data_agent_staging_datasource_fewshot",
        "PATCH",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots/{few_shot_id}",
        None,
    ),
    (
        "fabric_publish_data_agent",
        "POST",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/publish",
        None,
    ),
    (
        "fabric_reset_data_agent_staging_changes",
        "POST",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/reset",
        None,
    ),
    (
        "fabric_get_data_agent_staging_settings",
        "GET",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/settings",
        None,
    ),
    (
        "fabric_update_data_agent_staging_settings",
        "PATCH",
        "/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/settings",
        None,
    ),
    (
        "fabric_run_data_build_tool_job",
        "POST",
        "/workspaces/{workspace_id}/dataBuildToolJobs/{data_build_tool_job_id}/jobs/execute/instances",
        None,
    ),
    (
        "fabric_create_data_build_tool_job_schedule",
        "POST",
        "/workspaces/{workspace_id}/dataBuildToolJobs/{data_build_tool_job_id}/jobs/execute/schedules",
        None,
    ),
    (
        "fabric_list_data_pipeline_job_instances",
        "GET",
        "/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/instances",
        None,
    ),
    (
        "fabric_run_data_pipeline",
        "POST",
        "/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/instances",
        None,
    ),
    (
        "fabric_get_data_pipeline_job_instance",
        "GET",
        "/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/instances/{job_instance_id}",
        None,
    ),
    (
        "fabric_list_data_pipeline_schedules",
        "GET",
        "/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules",
        None,
    ),
    (
        "fabric_create_data_pipeline_schedule",
        "POST",
        "/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules",
        None,
    ),
    (
        "fabric_delete_data_pipeline_schedule",
        "DELETE",
        "/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules/{schedule_id}",
        None,
    ),
    (
        "fabric_get_data_pipeline_schedule",
        "GET",
        "/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules/{schedule_id}",
        None,
    ),
    (
        "fabric_update_data_pipeline_schedule",
        "PATCH",
        "/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules/{schedule_id}",
        None,
    ),
    ("fabric_upgrade_dataflows_gen1", "POST", "/workspaces/{workspace_id}/dataflows/gen1Upgrade", None),
    (
        "fabric_list_dataflow_gen1_upgrade_readiness",
        "GET",
        "/workspaces/{workspace_id}/dataflows/gen1UpgradeReadinessResults",
        None,
    ),
    ("fabric_execute_dataflow_query", "POST", "/workspaces/{workspace_id}/dataflows/{dataflow_id}/executeQuery", None),
    (
        "fabric_run_dataflow_apply_changes",
        "POST",
        "/workspaces/{workspace_id}/dataflows/{dataflow_id}/jobs/applyChanges/instances",
        None,
    ),
    (
        "fabric_create_dataflow_apply_changes_schedule",
        "POST",
        "/workspaces/{workspace_id}/dataflows/{dataflow_id}/jobs/applyChanges/schedules",
        None,
    ),
    ("fabric_run_dataflow", "POST", "/workspaces/{workspace_id}/dataflows/{dataflow_id}/jobs/execute/instances", None),
    (
        "fabric_create_dataflow_schedule",
        "POST",
        "/workspaces/{workspace_id}/dataflows/{dataflow_id}/jobs/execute/schedules",
        None,
    ),
    (
        "fabric_discover_dataflow_parameters",
        "GET",
        "/workspaces/{workspace_id}/dataflows/{dataflow_id}/parameters",
        None,
    ),
    (
        "fabric_list_environment_published_libraries",
        "GET",
        "/workspaces/{workspace_id}/environments/{environment_id}/libraries",
        {"beta": "false"},
    ),
    (
        "fabric_export_environment_published_external_libraries",
        "GET",
        "/workspaces/{workspace_id}/environments/{environment_id}/libraries/exportExternalLibraries",
        None,
    ),
    (
        "fabric_get_environment_published_spark_compute",
        "GET",
        "/workspaces/{workspace_id}/environments/{environment_id}/sparkcompute",
        {"beta": "false"},
    ),
    (
        "fabric_cancel_environment_publish",
        "POST",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/cancelPublish",
        None,
    ),
    (
        "fabric_list_environment_staging_libraries",
        "GET",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries",
        {"beta": "false"},
    ),
    (
        "fabric_export_environment_staging_external_libraries",
        "GET",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/exportExternalLibraries",
        None,
    ),
    (
        "fabric_import_environment_staging_external_libraries",
        "POST",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/importExternalLibraries",
        None,
    ),
    (
        "fabric_remove_environment_staging_external_library",
        "POST",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/removeExternalLibrary",
        None,
    ),
    (
        "fabric_delete_environment_staging_custom_library",
        "DELETE",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/{library_name}",
        None,
    ),
    (
        "fabric_upload_environment_staging_custom_library",
        "POST",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/{library_name}",
        None,
    ),
    (
        "fabric_publish_environment",
        "POST",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/publish",
        {"beta": "false"},
    ),
    (
        "fabric_get_environment_staging_spark_compute",
        "GET",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/sparkcompute",
        {"beta": "false"},
    ),
    (
        "fabric_update_environment_staging_spark_compute",
        "PATCH",
        "/workspaces/{workspace_id}/environments/{environment_id}/staging/sparkcompute",
        {"beta": "false"},
    ),
    (
        "fabric_get_eventstream_destination",
        "GET",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/destinations/{destination_id}",
        None,
    ),
    (
        "fabric_get_eventstream_destination_connection",
        "GET",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/destinations/{destination_id}/connection",
        None,
    ),
    (
        "fabric_pause_eventstream_destination",
        "POST",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/destinations/{destination_id}/pause",
        None,
    ),
    (
        "fabric_resume_eventstream_destination",
        "POST",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/destinations/{destination_id}/resume",
        None,
    ),
    ("fabric_pause_eventstream", "POST", "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/pause", None),
    ("fabric_resume_eventstream", "POST", "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/resume", None),
    (
        "fabric_get_eventstream_source",
        "GET",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/sources/{source_id}",
        None,
    ),
    (
        "fabric_get_eventstream_source_connection",
        "GET",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/sources/{source_id}/connection",
        None,
    ),
    (
        "fabric_pause_eventstream_source",
        "POST",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/sources/{source_id}/pause",
        None,
    ),
    (
        "fabric_resume_eventstream_source",
        "POST",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/sources/{source_id}/resume",
        None,
    ),
    (
        "fabric_get_eventstream_topology",
        "GET",
        "/workspaces/{workspace_id}/eventstreams/{eventstream_id}/topology",
        None,
    ),
    (
        "fabric_execute_graph_model_query",
        "POST",
        "/workspaces/{workspace_id}/graphModels/{graph_model_id}/executeQuery",
        {"beta": "true"},
    ),
    (
        "fabric_get_graph_model_queryable_graph_type",
        "GET",
        "/workspaces/{workspace_id}/graphModels/{graph_model_id}/getQueryableGraphType",
        {"beta": "true"},
    ),
    (
        "fabric_refresh_graph_model",
        "POST",
        "/workspaces/{workspace_id}/graphModels/{graph_model_id}/jobs/refreshGraph/instances",
        None,
    ),
    (
        "fabric_list_kql_database_shortcuts",
        "GET",
        "/workspaces/{workspace_id}/kqlDatabases/{kql_database_id}/shortcuts",
        None,
    ),
    (
        "fabric_create_kql_database_shortcut",
        "POST",
        "/workspaces/{workspace_id}/kqlDatabases/{kql_database_id}/shortcuts",
        None,
    ),
    (
        "fabric_delete_kql_database_shortcut",
        "DELETE",
        "/workspaces/{workspace_id}/kqlDatabases/{kql_database_id}/shortcuts/{shortcut_name}",
        None,
    ),
    (
        "fabric_get_kql_database_shortcut",
        "GET",
        "/workspaces/{workspace_id}/kqlDatabases/{kql_database_id}/shortcuts/{shortcut_name}",
        None,
    ),
    (
        "fabric_run_lakehouse_refresh_materialized_lake_views",
        "POST",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/refreshMaterializedLakeViews/instances",
        None,
    ),
    (
        "fabric_create_lakehouse_refresh_materialized_lake_views_schedule",
        "POST",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/refreshMaterializedLakeViews/schedules",
        None,
    ),
    (
        "fabric_delete_lakehouse_refresh_materialized_lake_views_schedule",
        "DELETE",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/refreshMaterializedLakeViews/schedules/{schedule_id}",
        None,
    ),
    (
        "fabric_update_lakehouse_refresh_materialized_lake_views_schedule",
        "PATCH",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/refreshMaterializedLakeViews/schedules/{schedule_id}",
        None,
    ),
    (
        "fabric_run_lakehouse_table_maintenance",
        "POST",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/tableMaintenance/instances",
        None,
    ),
    (
        "fabric_list_lakehouse_livy_sessions",
        "GET",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/livySessions",
        None,
    ),
    (
        "fabric_get_lakehouse_livy_session",
        "GET",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/livySessions/{livy_id}",
        None,
    ),
    (
        "fabric_list_lakehouse_mlv_execution_definitions",
        "GET",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions",
        None,
    ),
    (
        "fabric_create_lakehouse_mlv_execution_definition",
        "POST",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions",
        None,
    ),
    (
        "fabric_delete_lakehouse_mlv_execution_definition",
        "DELETE",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions/{mlv_execution_definition_id}",
        None,
    ),
    (
        "fabric_get_lakehouse_mlv_execution_definition",
        "GET",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions/{mlv_execution_definition_id}",
        None,
    ),
    (
        "fabric_update_lakehouse_mlv_execution_definition",
        "PATCH",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions/{mlv_execution_definition_id}",
        None,
    ),
    (
        "fabric_load_lakehouse_schema_table",
        "POST",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/schemas/{schema_name}/tables/{table_name}/load",
        {"beta": "true"},
    ),
    ("fabric_list_lakehouse_tables", "GET", "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/tables", None),
    (
        "fabric_load_lakehouse_table",
        "POST",
        "/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/tables/{table_name}/load",
        None,
    ),
    ("fabric_discover_azure_databricks_catalogs", "GET", "/workspaces/{workspace_id}/azureDatabricks/catalogs", None),
    (
        "fabric_discover_azure_databricks_catalog_schemas",
        "GET",
        "/workspaces/{workspace_id}/azureDatabricks/catalogs/{catalog_name}/schemas",
        None,
    ),
    (
        "fabric_discover_azure_databricks_catalog_schema_tables",
        "GET",
        "/workspaces/{workspace_id}/azureDatabricks/catalogs/{catalog_name}/schemas/{schema_name}/tables",
        None,
    ),
    (
        "fabric_refresh_mirrored_azure_databricks_catalog_metadata",
        "POST",
        "/workspaces/{workspace_id}/mirroredAzureDatabricksCatalogs/{mirrored_azure_databricks_catalog_id}/refreshCatalogMetadata",
        None,
    ),
    (
        "fabric_list_catalog_mirroring_scopes",
        "GET",
        "/workspaces/{workspace_id}/catalogmirroring/scopes",
        {"beta": "true"},
    ),
    (
        "fabric_list_catalog_mirroring_tables",
        "GET",
        "/workspaces/{workspace_id}/catalogmirroring/tables",
        {"beta": "true"},
    ),
    (
        "fabric_get_mirrored_catalog_mirroring_status",
        "GET",
        "/workspaces/{workspace_id}/mirroredCatalogs/{mirrored_catalog_id}/mirroringStatus",
        {"beta": "true"},
    ),
    (
        "fabric_refresh_mirrored_catalog_metadata",
        "POST",
        "/workspaces/{workspace_id}/mirroredCatalogs/{mirrored_catalog_id}/refreshCatalogMetadata",
        {"beta": "true"},
    ),
    (
        "fabric_get_mirrored_catalog_tables_mirroring_status",
        "GET",
        "/workspaces/{workspace_id}/mirroredCatalogs/{mirrored_catalog_id}/tablesMirroringStatus",
        {"beta": "true"},
    ),
    (
        "fabric_get_mirrored_database_mirroring_status",
        "POST",
        "/workspaces/{workspace_id}/mirroredDatabases/{mirrored_database_id}/getMirroringStatus",
        None,
    ),
    (
        "fabric_get_mirrored_database_tables_mirroring_status",
        "POST",
        "/workspaces/{workspace_id}/mirroredDatabases/{mirrored_database_id}/getTablesMirroringStatus",
        None,
    ),
    (
        "fabric_start_mirrored_database_mirroring",
        "POST",
        "/workspaces/{workspace_id}/mirroredDatabases/{mirrored_database_id}/startMirroring",
        None,
    ),
    (
        "fabric_stop_mirrored_database_mirroring",
        "POST",
        "/workspaces/{workspace_id}/mirroredDatabases/{mirrored_database_id}/stopMirroring",
        None,
    ),
    ("fabric_score_ml_model_endpoint", "POST", "/workspaces/{workspace_id}/mlModels/{model_id}/endpoint/score", None),
    ("fabric_get_ml_model_endpoint", "GET", "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint", None),
    ("fabric_update_ml_model_endpoint", "PATCH", "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint", None),
    (
        "fabric_list_ml_model_endpoint_versions",
        "GET",
        "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions",
        None,
    ),
    (
        "fabric_deactivate_all_ml_model_endpoint_versions",
        "POST",
        "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/deactivateAll",
        None,
    ),
    (
        "fabric_get_ml_model_endpoint_version",
        "GET",
        "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{name}",
        None,
    ),
    (
        "fabric_update_ml_model_endpoint_version",
        "PATCH",
        "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{name}",
        None,
    ),
    (
        "fabric_activate_ml_model_endpoint_version",
        "POST",
        "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{name}/activate",
        None,
    ),
    (
        "fabric_deactivate_ml_model_endpoint_version",
        "POST",
        "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{name}/deactivate",
        None,
    ),
    (
        "fabric_score_ml_model_endpoint_version",
        "POST",
        "/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{name}/score",
        None,
    ),
    (
        "fabric_run_notebook",
        "POST",
        "/workspaces/{workspace_id}/notebooks/{notebook_id}/jobs/execute/instances",
        {"beta": "false"},
    ),
    (
        "fabric_get_notebook_job_instance",
        "GET",
        "/workspaces/{workspace_id}/notebooks/{notebook_id}/jobs/execute/instances/{job_instance_id}",
        {"beta": "true"},
    ),
    (
        "fabric_list_notebook_livy_sessions",
        "GET",
        "/workspaces/{workspace_id}/notebooks/{notebook_id}/livySessions",
        None,
    ),
    (
        "fabric_get_notebook_livy_session",
        "GET",
        "/workspaces/{workspace_id}/notebooks/{notebook_id}/livySessions/{livy_id}",
        None,
    ),
    ("fabric_nl_to_kql", "POST", "/workspaces/{workspace_id}/realTimeIntelligence/nltokql", {"beta": "true"}),
    (
        "fabric_bind_semantic_model_connection",
        "POST",
        "/workspaces/{workspace_id}/semanticModels/{semantic_model_id}/bindConnection",
        None,
    ),
    ("fabric_list_capacity_spark_custom_pools", "GET", "/capacities/{capacity_id}/spark/pools", {"beta": "true"}),
    ("fabric_create_capacity_spark_custom_pool", "POST", "/capacities/{capacity_id}/spark/pools", {"beta": "true"}),
    (
        "fabric_delete_capacity_spark_custom_pool",
        "DELETE",
        "/capacities/{capacity_id}/spark/pools/{pool_id}",
        {"beta": "true"},
    ),
    (
        "fabric_get_capacity_spark_custom_pool",
        "GET",
        "/capacities/{capacity_id}/spark/pools/{pool_id}",
        {"beta": "true"},
    ),
    (
        "fabric_update_capacity_spark_custom_pool",
        "PATCH",
        "/capacities/{capacity_id}/spark/pools/{pool_id}",
        {"beta": "true"},
    ),
    ("fabric_get_capacity_spark_settings", "GET", "/capacities/{capacity_id}/spark/settings", {"beta": "true"}),
    ("fabric_update_capacity_spark_settings", "PATCH", "/capacities/{capacity_id}/spark/settings", {"beta": "true"}),
    ("fabric_list_workspace_livy_sessions", "GET", "/workspaces/{workspace_id}/spark/livySessions", None),
    ("fabric_list_workspace_spark_custom_pools", "GET", "/workspaces/{workspace_id}/spark/pools", None),
    ("fabric_create_workspace_spark_custom_pool", "POST", "/workspaces/{workspace_id}/spark/pools", None),
    ("fabric_delete_workspace_spark_custom_pool", "DELETE", "/workspaces/{workspace_id}/spark/pools/{pool_id}", None),
    ("fabric_get_workspace_spark_custom_pool", "GET", "/workspaces/{workspace_id}/spark/pools/{pool_id}", None),
    ("fabric_update_workspace_spark_custom_pool", "PATCH", "/workspaces/{workspace_id}/spark/pools/{pool_id}", None),
    ("fabric_get_workspace_spark_settings", "GET", "/workspaces/{workspace_id}/spark/settings", None),
    ("fabric_update_workspace_spark_settings", "PATCH", "/workspaces/{workspace_id}/spark/settings", None),
    (
        "fabric_run_spark_job_definition",
        "POST",
        "/workspaces/{workspace_id}/sparkJobDefinitions/{spark_job_definition_id}/jobs/sparkjob/instances",
        None,
    ),
    (
        "fabric_list_spark_job_definition_livy_sessions",
        "GET",
        "/workspaces/{workspace_id}/sparkJobDefinitions/{spark_job_definition_id}/livySessions",
        None,
    ),
    (
        "fabric_get_spark_job_definition_livy_session",
        "GET",
        "/workspaces/{workspace_id}/sparkJobDefinitions/{spark_job_definition_id}/livySessions/{livy_id}",
        None,
    ),
    (
        "fabric_list_sql_database_restorable_deleted_databases",
        "GET",
        "/workspaces/{workspace_id}/sqlDatabases/restorableDeletedDatabases",
        None,
    ),
    (
        "fabric_revalidate_sql_database_cmk",
        "POST",
        "/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/revalidateCMK",
        None,
    ),
    (
        "fabric_get_sql_database_audit_settings",
        "GET",
        "/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/settings/sqlAudit",
        None,
    ),
    (
        "fabric_update_sql_database_audit_settings",
        "PATCH",
        "/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/settings/sqlAudit",
        None,
    ),
    (
        "fabric_start_sql_database_mirroring",
        "POST",
        "/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/startMirroring",
        None,
    ),
    (
        "fabric_stop_sql_database_mirroring",
        "POST",
        "/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/stopMirroring",
        None,
    ),
    (
        "fabric_get_sql_endpoint_audit_settings",
        "GET",
        "/workspaces/{workspace_id}/sqlEndpoints/{item_id}/settings/sqlAudit",
        None,
    ),
    (
        "fabric_update_sql_endpoint_audit_settings",
        "PATCH",
        "/workspaces/{workspace_id}/sqlEndpoints/{item_id}/settings/sqlAudit",
        None,
    ),
    (
        "fabric_set_sql_endpoint_audit_actions_and_groups",
        "POST",
        "/workspaces/{workspace_id}/sqlEndpoints/{item_id}/settings/sqlAudit/setAuditActionsAndGroups",
        None,
    ),
    (
        "fabric_get_sql_endpoint_connection_string",
        "GET",
        "/workspaces/{workspace_id}/sqlEndpoints/{sql_endpoint_id}/connectionString",
        None,
    ),
    (
        "fabric_refresh_sql_endpoint_metadata",
        "POST",
        "/workspaces/{workspace_id}/sqlEndpoints/{sql_endpoint_id}/refreshMetadata",
        None,
    ),
    (
        "fabric_get_warehouse_sql_pools_configuration",
        "GET",
        "/workspaces/{workspace_id}/warehouses/sqlPoolsConfiguration",
        {"beta": "true"},
    ),
    (
        "fabric_update_warehouse_sql_pools_configuration",
        "PATCH",
        "/workspaces/{workspace_id}/warehouses/sqlPoolsConfiguration",
        {"beta": "true"},
    ),
    (
        "fabric_get_warehouse_audit_settings",
        "GET",
        "/workspaces/{workspace_id}/warehouses/{item_id}/settings/sqlAudit",
        None,
    ),
    (
        "fabric_update_warehouse_audit_settings",
        "PATCH",
        "/workspaces/{workspace_id}/warehouses/{item_id}/settings/sqlAudit",
        None,
    ),
    (
        "fabric_set_warehouse_audit_actions_and_groups",
        "POST",
        "/workspaces/{workspace_id}/warehouses/{item_id}/settings/sqlAudit/setAuditActionsAndGroups",
        None,
    ),
    (
        "fabric_get_warehouse_connection_string",
        "GET",
        "/workspaces/{workspace_id}/warehouses/{warehouse_id}/connectionString",
        None,
    ),
    (
        "fabric_list_warehouse_restore_points",
        "GET",
        "/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints",
        None,
    ),
    (
        "fabric_create_warehouse_restore_point",
        "POST",
        "/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints",
        None,
    ),
    (
        "fabric_delete_warehouse_restore_point",
        "DELETE",
        "/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints/{restore_point_id}",
        None,
    ),
    (
        "fabric_get_warehouse_restore_point",
        "GET",
        "/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints/{restore_point_id}",
        None,
    ),
    (
        "fabric_update_warehouse_restore_point",
        "PATCH",
        "/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints/{restore_point_id}",
        None,
    ),
    (
        "fabric_restore_warehouse_to_restore_point",
        "POST",
        "/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints/{restore_point_id}/restore",
        None,
    ),
]

_RAW_UPLOAD_ARGS = {
    "fabric_create_or_update_apache_airflow_job_file": {"content": "x"},
    "fabric_upload_environment_staging_custom_library": {"content": "x"},
    "fabric_import_environment_staging_external_libraries": {"content": "x"},
}


async def _run(cls: type, **kwargs: Any) -> Any:
    return await cls(token="tok")(**kwargs).collect()


def _tool(**kwargs: object) -> SimpleNamespace:
    defaults = {
        "integration": None,
        "api_key": None,
        "token": None,
        "tenant_id": None,
        "client_id": None,
        "client_secret": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class _Response:
    def __init__(
        self,
        status_code: int = 200,
        body: Any = None,
        headers: dict[str, str] | None = None,
        raw: bytes | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        if raw is not None:
            self.content = raw
        else:
            self.content = json.dumps(body).encode() if body is not None else b""
        self.headers = {"content-type": "application/json"} if body is not None else {}
        self.headers.update(headers or {})
        self.text = self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("no json body")
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@pytest.fixture(autouse=True)
def _clear_fabric_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "FABRIC_API_KEY",
        "FABRIC_ACCESS_TOKEN",
        "FABRIC_TENANT_ID",
        "FABRIC_CLIENT_ID",
        "FABRIC_CLIENT_SECRET",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def http(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace httpx.AsyncClient with a scripted fake and make asyncio.sleep instant."""
    state = SimpleNamespace(calls=[], posts=[], responses=[], sleeps=[])

    class _Client:
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...

        async def __aenter__(self) -> Self:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def request(self, method: str, url: str, **kwargs: Any) -> _Response:
            state.calls.append({"method": method, "url": url, **kwargs})
            return state.responses.pop(0) if state.responses else _Response(200, {})

        async def post(self, url: str, **kwargs: Any) -> _Response:
            state.posts.append({"url": url, **kwargs})
            return _Response(200, {"access_token": "sp-token"})

    async def _sleep(seconds: float) -> None:
        state.sleeps.append(seconds)

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    monkeypatch.setattr(fabric_module, "asyncio", SimpleNamespace(sleep=_sleep))
    return state


class TestFabricAuth:
    async def test_explicit_api_key(self) -> None:
        assert await _resolve_token(_tool(api_key=SecretStr("app-bearer"))) == "app-bearer"

    async def test_explicit_oauth_token(self) -> None:
        assert await _resolve_token(_tool(token=SecretStr("user-oauth"))) == "user-oauth"

    async def test_explicit_api_key_beats_explicit_token(self) -> None:
        tool = _tool(api_key=SecretStr("app-bearer"), token=SecretStr("user-oauth"))
        assert await _resolve_token(tool) == "app-bearer"

    async def test_oauth_token_from_integration(self) -> None:
        integration = MagicMock(spec=Integration)
        integration.resolve = AsyncMock(return_value={"token": "platform-user-token"})
        assert await _resolve_token(_tool(integration=integration)) == "platform-user-token"

    async def test_oauth_access_token_alias_from_integration(self) -> None:
        integration = MagicMock(spec=Integration)
        integration.resolve = AsyncMock(return_value={"access_token": "platform-user-access"})
        assert await _resolve_token(_tool(integration=integration)) == "platform-user-access"

    async def test_api_key_from_integration(self) -> None:
        integration = MagicMock(spec=Integration)
        integration.resolve = AsyncMock(return_value={"api_key": "platform-app-key"})
        assert await _resolve_token(_tool(integration=integration)) == "platform-app-key"

    async def test_integration_oauth_beats_api_key(self) -> None:
        integration = MagicMock(spec=Integration)
        integration.resolve = AsyncMock(return_value={"token": "user-oauth", "api_key": "app-key"})
        assert await _resolve_token(_tool(integration=integration)) == "user-oauth"

    async def test_service_principal_from_integration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        integration = MagicMock(spec=Integration)
        integration.resolve = AsyncMock(return_value={"tenant_id": "tid", "client_id": "cid", "client_secret": "sec"})
        exchange = AsyncMock(return_value="sp-access-token")
        monkeypatch.setattr("timbal.tools.fabric._get_token_from_client_credentials", exchange)
        assert await _resolve_token(_tool(integration=integration)) == "sp-access-token"
        exchange.assert_awaited_once_with("tid", "cid", "sec")

    async def test_service_principal_from_tool_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "timbal.tools.fabric._get_token_from_client_credentials", AsyncMock(return_value="sp-access-token")
        )
        tool = _tool(tenant_id="tid", client_id="cid", client_secret=SecretStr("sec"))
        assert await _resolve_token(tool) == "sp-access-token"

    async def test_client_credentials_request_uses_fabric_scope(self, http: SimpleNamespace) -> None:
        assert await _get_token_from_client_credentials("tid", "cid", "sec") == "sp-token"
        post = http.posts[0]
        assert post["url"] == "https://login.microsoftonline.com/tid/oauth2/v2.0/token"
        assert post["data"]["scope"] == "https://api.fabric.microsoft.com/.default"
        assert post["data"]["grant_type"] == "client_credentials"

    async def test_env_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FABRIC_API_KEY", "env-app-key")
        assert await _resolve_token(_tool()) == "env-app-key"

    async def test_env_access_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FABRIC_ACCESS_TOKEN", "env-user-token")
        assert await _resolve_token(_tool()) == "env-user-token"

    async def test_env_api_key_beats_env_access_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FABRIC_API_KEY", "env-app-key")
        monkeypatch.setenv("FABRIC_ACCESS_TOKEN", "env-user-token")
        assert await _resolve_token(_tool()) == "env-app-key"

    def test_service_principal_parts_incomplete(self) -> None:
        assert _service_principal_parts(_tool(tenant_id="tid", client_id="cid"), {}) is None

    async def test_missing_credentials_raises(self) -> None:
        with pytest.raises(CredentialNotAvailable) as exc_info:
            await _resolve_token(_tool())
        assert exc_info.value.provider_name == "Fabric"

    def test_config_exposes_all_auth_fields(self) -> None:
        config = FabricListWorkspaces(integration=Integration("fabric", "org-int-1")).get_config()
        for field in ("integration", "api_key", "token", "tenant_id", "client_id", "client_secret"):
            assert field in config
        assert config["integration"]["value"] == "org-int-1"


class TestFabricHelpers:
    def test_clean_params(self) -> None:
        assert _clean_params(None) is None
        assert _clean_params({"a": None}) is None
        assert _clean_params({"a": True, "b": False, "c": ["x", "y"], "d": 5, "e": None}) == {
            "a": "true",
            "b": "false",
            "c": "x,y",
            "d": 5,
        }

    def test_upload_bytes_text_and_base64(self) -> None:
        assert _upload_bytes("héllo", None) == "héllo".encode()
        assert _upload_bytes(None, base64.b64encode(b"\x00\x01").decode()) == b"\x00\x01"

    @pytest.mark.parametrize(("content", "content_base64"), [(None, None), ("a", "YQ==")])
    def test_upload_bytes_requires_exactly_one(self, content: str | None, content_base64: str | None) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            _upload_bytes(content, content_base64)

    def test_raise_for_fabric_includes_code_message_and_request_id(self) -> None:
        response = _Response(404, {"errorCode": "ItemNotFound", "message": "Item gone", "requestId": "req-1"})
        with pytest.raises(ValueError, match=r"404: ItemNotFound: Item gone \(requestId req-1\)"):
            _raise_for_fabric(response)

    def test_raise_for_fabric_unauthorized_hints_scopes(self) -> None:
        with pytest.raises(ValueError, match="delegated scopes"):
            _raise_for_fabric(_Response(401, {"errorCode": "Unauthorized", "message": "bad token"}))

    def test_raise_for_fabric_falls_back_to_text(self) -> None:
        with pytest.raises(ValueError, match="502: upstream exploded"):
            _raise_for_fabric(_Response(502, raw=b"upstream exploded"))

    def test_raise_for_fabric_ok_is_silent(self) -> None:
        _raise_for_fabric(_Response(200, {}))

    def test_parse_response_empty_and_204(self) -> None:
        assert _parse_response(_Response(204)) == {"status": "success", "status_code": 204}
        assert _parse_response(_Response(200, raw=b"")) == {"status": "success", "status_code": 200}

    def test_parse_response_json(self) -> None:
        assert _parse_response(_Response(200, {"id": "1"})) == {"id": "1"}

    def test_parse_response_binary_and_text(self) -> None:
        binary = _Response(200, raw=b"\x00\x01", headers={"content-type": "application/octet-stream"})
        parsed = _parse_response(binary)
        assert base64.b64decode(parsed["content_base64"]) == b"\x00\x01"
        assert "content" not in parsed
        text = _parse_response(_Response(200, raw=b"a: 1", headers={"content-type": "text/plain"}))
        assert text["content"] == "a: 1"


class TestFabricRequests:
    async def test_list_sends_bearer_query_and_no_body(self, http: SimpleNamespace) -> None:
        http.responses = [_Response(200, {"value": [{"id": "1"}]})]
        out = await _run(
            FabricListItems,
            workspace_id="ws-1",
            type="Lakehouse",
            recursive=False,
            include=["tags", "sensitivityLabel"],
        )
        assert out.status.code == "success", out.error
        assert out.output == {"value": [{"id": "1"}]}
        call = http.calls[0]
        assert (call["method"], call["url"]) == ("GET", f"{BASE_URL}/workspaces/ws-1/items")
        assert call["headers"] == {"Authorization": "Bearer tok"}
        assert call["params"] == {"type": "Lakehouse", "recursive": "false", "include": "tags,sensitivityLabel"}
        assert "json" not in call

    async def test_continuation_token_is_forwarded(self, http: SimpleNamespace) -> None:
        await _run(FabricListWorkspaces, continuation_token="abc")
        assert http.calls[0]["params"] == {"continuationToken": "abc"}

    async def test_post_body_maps_wire_names_and_drops_none(self, http: SimpleNamespace) -> None:
        http.responses = [_Response(201, {"id": "new"})]
        out = await _run(FabricCreateItem, workspace_id="ws-1", display_name="Sales", type="Lakehouse", folder_id="f-1")
        assert out.output == {"id": "new"}
        assert http.calls[0]["json"] == {"displayName": "Sales", "type": "Lakehouse", "folderId": "f-1"}

    async def test_path_params_are_quoted_and_shortcut_path_keeps_slashes(self, http: SimpleNamespace) -> None:
        await _run(
            FabricGetItemShortcut,
            workspace_id="ws-1",
            item_id="i-1",
            shortcut_path="Files/sub dir",
            shortcut_name="a b",
        )
        assert http.calls[0]["url"] == f"{BASE_URL}/workspaces/ws-1/items/i-1/shortcuts/Files/sub%20dir/a%20b"
        await _run(FabricRunItemJob, workspace_id="ws-1", item_id="i-1", job_type="Default Job")
        assert http.calls[1]["url"].endswith("/items/i-1/jobs/Default%20Job/instances")

    async def test_error_response_surfaces_fabric_error(self, http: SimpleNamespace) -> None:
        http.responses = [_Response(404, {"errorCode": "WorkspaceNotFound", "message": "nope"})]
        out = await _run(FabricListItems, workspace_id="missing")
        assert out.status.code == "error"
        assert "WorkspaceNotFound" in out.error["message"]

    async def test_rate_limit_retries_using_retry_after(self, http: SimpleNamespace) -> None:
        http.responses = [
            _Response(429, {"errorCode": "TooManyRequests", "message": "slow"}, headers={"Retry-After": "7"}),
            _Response(200, {"value": []}),
        ]
        out = await _run(FabricListWorkspaces)
        assert out.output == {"value": []}
        assert len(http.calls) == 2
        assert http.sleeps == [7.0]

    async def test_rate_limit_gives_up_after_retries(self, http: SimpleNamespace) -> None:
        http.responses = [_Response(429, {"errorCode": "TooManyRequests", "message": "slow"}) for _ in range(10)]
        out = await _run(FabricListWorkspaces)
        assert out.status.code == "error"
        assert len(http.calls) == fabric_module._MAX_RATE_LIMIT_RETRIES + 1

    async def test_missing_credentials_is_reported(self) -> None:
        out = await FabricListWorkspaces()().collect()
        assert out.status.code == "error"


class TestFabricLongRunningOperations:
    OP = "https://api.fabric.microsoft.com/v1/operations/op-1"

    def _accepted(self, location: str | None = None, **extra: str) -> _Response:
        headers = {"Location": location or self.OP, "x-ms-operation-id": "op-1", "Retry-After": "1", **extra}
        return _Response(202, headers=headers)

    async def test_polls_until_succeeded_then_returns_result(self, http: SimpleNamespace) -> None:
        http.responses = [
            self._accepted(),
            _Response(200, {"status": "Running"}),
            _Response(200, {"status": "Succeeded"}),
            _Response(200, {"id": "lakehouse-1", "displayName": "Sales"}),
        ]
        out = await _run(FabricCreateItem, workspace_id="ws-1", display_name="Sales", type="Lakehouse")
        assert out.output == {"id": "lakehouse-1", "displayName": "Sales"}
        assert [c["url"] for c in http.calls[1:]] == [self.OP, self.OP, f"{self.OP}/result"]

    async def test_failed_operation_raises_with_error(self, http: SimpleNamespace) -> None:
        http.responses = [
            self._accepted(),
            _Response(200, {"status": "Failed", "error": {"errorCode": "Boom", "message": "it broke"}}),
        ]
        out = await _run(FabricCreateItem, workspace_id="ws-1", display_name="x", type="Notebook")
        assert out.status.code == "error"
        assert "Failed" in out.error["message"] and "it broke" in out.error["message"]

    async def test_result_not_available_returns_operation_state(self, http: SimpleNamespace) -> None:
        http.responses = [
            self._accepted(),
            _Response(200, {"status": "Succeeded", "percentComplete": 100}),
            _Response(404, {"errorCode": "OperationHasNoResult", "message": "none"}),
        ]
        out = await _run(FabricCreateItem, workspace_id="ws-1", display_name="x", type="Notebook")
        assert out.output == {"status": "Succeeded", "percentComplete": 100}

    async def test_still_running_returns_handle(self, http: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(fabric_module, "_LRO_MAX_WAIT_SECONDS", 3.0)
        http.responses = [self._accepted()] + [_Response(200, {"status": "Running"}) for _ in range(10)]
        out = await _run(FabricCreateItem, workspace_id="ws-1", display_name="x", type="Notebook")
        assert out.output["status"] == "Running"
        assert out.output["operation_id"] == "op-1"
        assert "fabric_get_operation_state" in out.output["message"]

    async def test_job_run_returns_job_instance_without_polling(self, http: SimpleNamespace) -> None:
        location = f"{BASE_URL}/workspaces/ws-1/items/i-1/jobs/instances/job-9"
        http.responses = [self._accepted(location)]
        out = await _run(FabricRunItemJob, workspace_id="ws-1", item_id="i-1", job_type="RunNotebook")
        assert out.output["status"] == "Accepted"
        assert out.output["job_instance_id"] == "job-9"
        assert len(http.calls) == 1

    async def test_foreign_location_is_never_followed(self, http: SimpleNamespace) -> None:
        http.responses = [self._accepted("https://evil.example.com/v1/operations/op-1")]
        out = await _run(FabricCreateItem, workspace_id="ws-1", display_name="x", type="Notebook")
        assert out.output["status"] == "Accepted"
        assert len(http.calls) == 1

    async def test_non_operation_location_returns_final_state(self, http: SimpleNamespace) -> None:
        pipeline_op = f"{BASE_URL}/deploymentPipelines/p-1/operations/op-1"
        http.responses = [self._accepted(pipeline_op), _Response(200, {"status": "Succeeded", "id": "op-1"})]
        out = await _run(FabricCreateItem, workspace_id="ws-1", display_name="x", type="Notebook")
        assert out.output == {"status": "Succeeded", "id": "op-1"}
        assert len(http.calls) == 2

    async def test_accepted_without_location(self, http: SimpleNamespace) -> None:
        http.responses = [_Response(202)]
        out = await _run(FabricCreateItem, workspace_id="ws-1", display_name="x", type="Notebook")
        assert out.output["status"] == "Accepted"
        assert out.output["location"] is None

    async def test_operation_state_tool(self, http: SimpleNamespace) -> None:
        http.responses = [_Response(200, {"status": "Running", "percentComplete": 40})]
        out = await _run(FabricGetOperationState, operation_id="op-1")
        assert out.output["percentComplete"] == 40
        assert http.calls[0]["url"] == f"{BASE_URL}/operations/op-1"


class TestFabricRawPayloads:
    async def test_binary_upload_sends_bytes_with_octet_stream(self, http: SimpleNamespace) -> None:
        payload = base64.b64encode(b"\x00dag\xff").decode()
        await _run(
            FabricCreateOrUpdateApacheAirflowJobFile,
            workspace_id="ws-1",
            apache_airflow_job_id="a-1",
            file_path="dags/x.py",
            content_base64=payload,
        )
        call = http.calls[0]
        assert call["method"] == "PUT"
        assert call["url"] == f"{BASE_URL}/workspaces/ws-1/apacheAirflowJobs/a-1/files/dags/x.py"
        assert call["params"] == {"beta": "true"}
        assert call["content"] == b"\x00dag\xff"
        assert call["headers"]["Content-Type"] == "application/octet-stream"

    async def test_upload_requires_content(self, http: SimpleNamespace) -> None:
        out = await _run(
            FabricCreateOrUpdateApacheAirflowJobFile,
            workspace_id="ws-1",
            apache_airflow_job_id="a-1",
            file_path="dags/x.py",
        )
        assert out.status.code == "error"
        assert http.calls == []

    async def test_text_upload_uses_text_plain(self, http: SimpleNamespace) -> None:
        await _run(
            FabricDeployApacheAirflowJobRequirements,
            workspace_id="ws-1",
            apache_airflow_job_id="a-1",
            content="pandas==2.2.0",
        )
        call = http.calls[0]
        assert call["content"] == b"pandas==2.2.0"
        assert call["headers"]["Content-Type"] == "text/plain"

    async def test_array_body_is_sent_as_json_array(self, http: SimpleNamespace) -> None:
        await _run(
            FabricSetWarehouseAuditActionsAndGroups,
            workspace_id="ws-1",
            item_id="w-1",
            audit_actions_and_groups=["BATCH_COMPLETED_GROUP"],
        )
        assert http.calls[0]["json"] == ["BATCH_COMPLETED_GROUP"]

    async def test_download_returns_base64(self, http: SimpleNamespace) -> None:
        from timbal.tools import FabricGetApacheAirflowJobFile

        http.responses = [
            _Response(200, raw=b"print('hi')", headers={"content-type": "application/octet-stream"}),
        ]
        out = await _run(
            FabricGetApacheAirflowJobFile, workspace_id="ws-1", apache_airflow_job_id="a-1", file_path="dags/x.py"
        )
        assert base64.b64decode(out.output["content_base64"]) == b"print('hi')"


class TestFabricRegistry:
    def test_lazy_import_from_timbal_tools(self) -> None:
        from timbal.tools import FabricListItems as exported

        assert exported is FabricListItems

    def test_route_table_covers_every_exported_tool(self) -> None:
        import timbal.tools as tools_module

        exported = {n for n in tools_module.__all__ if n.startswith("Fabric")}
        assert len(exported) == FABRIC_TOOL_COUNT == len(_ROUTES)
        names = {getattr(tools_module, cls).model_fields["name"].default for cls in exported}
        assert names == {r[0] for r in _ROUTES}

    def test_tool_names_are_unique_prefixed_and_within_provider_limits(self) -> None:
        names = [r[0] for r in _ROUTES]
        assert len(set(names)) == len(names)
        assert all(n.startswith("fabric_") and len(n) <= 64 for n in names)

    def test_framework_discovery_registers_fabric_provider(self) -> None:
        tools = get_framework_tools(no_cache=True)
        fabric_tools = [ft for ft in tools.values() if ft.provider == "fabric"]
        assert len(fabric_tools) == FABRIC_TOOL_COUNT
        assert {"fabric_list_workspaces", "fabric_run_notebook", "fabric_admin_list_workspaces"} <= {
            ft.name for ft in fabric_tools
        }


def _all_fabric_classes() -> dict[str, type]:
    import timbal.tools as tools_module

    classes = {}
    for cls_name in tools_module.__all__:
        if cls_name.startswith("Fabric"):
            cls = getattr(tools_module, cls_name)
            classes[cls.model_fields["name"].default] = cls
    return classes


_CLASSES = _all_fabric_classes()


def _dummy(annotation: Any, name: str) -> Any:
    origin = typing.get_origin(annotation) or annotation
    return {int: 1, bool: True, float: 1.5, list: ["x"], dict: {"k": "v"}}.get(origin, f"id-{name}")


@pytest.mark.parametrize(("tool_name", "method", "template", "fixed_query"), _ROUTES, ids=[r[0] for r in _ROUTES])
async def test_route(
    tool_name: str, method: str, template: str, fixed_query: dict[str, str] | None, http: SimpleNamespace
) -> None:
    tool = _CLASSES[tool_name](token="tok")
    kwargs: dict[str, Any] = {}
    for name, param in inspect.signature(tool.handler).parameters.items():
        kwargs[name] = _dummy(param.annotation, name) if param.default.is_required() else None
    kwargs.update(_RAW_UPLOAD_ARGS.get(tool_name, {}))

    await tool.handler(**kwargs)

    call = http.calls[0]
    assert call["method"] == method
    assert call["url"] == BASE_URL + template.format(**{k: v for k, v in kwargs.items() if v is not None})
    assert call["headers"]["Authorization"] == "Bearer tok"
    for key, value in (fixed_query or {}).items():
        assert call["params"][key] == value


@pytest.mark.integration
async def test_live_list_workspaces() -> None:
    token = os.getenv("FABRIC_ACCESS_TOKEN")
    if not token:
        pytest.skip("Set FABRIC_ACCESS_TOKEN (Microsoft Entra bearer token for https://api.fabric.microsoft.com).")
    out = await FabricListWorkspaces(token=token).collect()
    assert out.status.code == "success", out.error
    assert "value" in out.output


@pytest.mark.integration
async def test_live_list_items() -> None:
    token = os.getenv("FABRIC_ACCESS_TOKEN")
    workspace_id = os.getenv("FABRIC_WORKSPACE_ID")
    if not (token and workspace_id):
        pytest.skip("Set FABRIC_ACCESS_TOKEN and FABRIC_WORKSPACE_ID.")
    out = await FabricListItems(token=token).collect(workspace_id=workspace_id)
    assert out.status.code == "success", out.error
    assert "value" in out.output
