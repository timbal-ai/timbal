"""Microsoft Fabric REST API tools: Admin APIs (tenant settings, domains, tags, workspaces and items at tenant scope).

Auth, long running operations and request handling live in ``fabric.py``.
"""

from typing import Any

from pydantic import Field

from .fabric import _drop_none, _fabric_request, _FabricTool, _quote


class FabricAdminListCapacitiesTenantSettingOverrides(_FabricTool):
    name: str = "fabric_admin_list_capacities_tenant_setting_overrides"
    description: str | None = (
        "Returns list of tenant setting overrides that override at the capacities. A maximum of 10,000 records can be "
        "returned per request. With the continuation token provided in the response, you can get the next 10,000 "
        "records."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_capacities_tenant_setting_overrides(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/capacities/delegatedTenantSettingOverrides",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_capacities_tenant_setting_overrides, **kwargs)


class FabricAdminListCapacityTenantSettingOverrides(_FabricTool):
    name: str = "fabric_admin_list_capacity_tenant_setting_overrides"
    description: str | None = "Returns list of tenant setting overrides that override for given capacity Id."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_capacity_tenant_setting_overrides(
            capacity_id: str = Field(..., description="The capacity ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/capacities/{capacity_id}/delegatedTenantSettingOverrides",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_capacity_tenant_setting_overrides, **kwargs)


class FabricAdminDeleteCapacityTenantSettingOverride(_FabricTool):
    name: str = "fabric_admin_delete_capacity_tenant_setting_override"
    description: str | None = "Remove given tenant setting override for given capacity Id."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_delete_capacity_tenant_setting_override(
            capacity_id: str = Field(..., description="The capacity ID."),
            tenant_setting_name: str = Field(..., description="The name of tenant setting."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/admin/capacities/{capacity_id}/delegatedTenantSettingOverrides/{_quote(tenant_setting_name)}",
            )

        super().__init__(handler=_admin_delete_capacity_tenant_setting_override, **kwargs)


class FabricAdminUpdateCapacityTenantSettingOverride(_FabricTool):
    name: str = "fabric_admin_update_capacity_tenant_setting_override"
    description: str | None = "Update given tenant setting override for given capacity Id."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_update_capacity_tenant_setting_override(
            capacity_id: str = Field(..., description="The capacity ID."),
            tenant_setting_name: str = Field(..., description="The name of tenant setting."),
            enabled: bool = Field(
                ..., description="The status of the tenant setting. False - Disabled, True - Enabled."
            ),
            enabled_security_groups: list[Any] | None = Field(None, description="A list of enabled security groups."),
            excluded_security_groups: list[Any] | None = Field(None, description="A list of excluded security groups."),
            delegate_to_workspace: bool | None = Field(
                None,
                description=(
                    "Indicates whether the tenant setting can be delegated to a workspace admin. False - Workspace admin "
                    "cannot override the tenant setting. True - Workspace admin can override the tenant setting."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/capacities/{capacity_id}/delegatedTenantSettingOverrides/{_quote(tenant_setting_name)}/update",
                body=_drop_none(
                    {
                        "enabled": enabled,
                        "enabledSecurityGroups": enabled_security_groups,
                        "excludedSecurityGroups": excluded_security_groups,
                        "delegateToWorkspace": delegate_to_workspace,
                    }
                ),
            )

        super().__init__(handler=_admin_update_capacity_tenant_setting_override, **kwargs)


class FabricAdminListDomains(_FabricTool):
    name: str = "fabric_admin_list_domains"
    description: str | None = "Returns info for all domains. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_domains(
            non_empty_only: bool | None = Field(
                None,
                description=(
                    "When true, only return domains with workspaces assigned that contain one or more items the user has "
                    "at least read access to. Default: false."
                ),
            ),
            with_assigned_workspaces_only: bool | None = Field(
                None,
                description=(
                    "When true, only return domains that have at least one workspace assigned to them, or to any of their "
                    "subdomains. Default: false."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/domains",
                params={
                    "nonEmptyOnly": non_empty_only,
                    "withAssignedWorkspacesOnly": with_assigned_workspaces_only,
                    "preview": "false",
                },
            )

        super().__init__(handler=_admin_list_domains, **kwargs)


class FabricAdminCreateDomain(_FabricTool):
    name: str = "fabric_admin_create_domain"
    description: str | None = "Creates a new domain. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_create_domain(
            display_name: str = Field(
                ..., description="The domain display name. The display name cannot contain more than 40 characters."
            ),
            description: str | None = Field(
                None, description="The domain description. The description cannot contain more than 256 characters."
            ),
            parent_domain_id: str | None = Field(None, description="The domain parent object ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/admin/domains",
                params={"preview": "false"},
                body=_drop_none(
                    {"displayName": display_name, "description": description, "parentDomainId": parent_domain_id}
                ),
            )

        super().__init__(handler=_admin_create_domain, **kwargs)


class FabricAdminListDomainsTenantSettingOverrides(_FabricTool):
    name: str = "fabric_admin_list_domains_tenant_setting_overrides"
    description: str | None = (
        "Returns list of domain delegation setting overrides. A maximum of 10,000 records can be returned per "
        "request. With the continuation token provided in the response, you can get the next 10,000 records."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_domains_tenant_setting_overrides(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/domains/delegatedTenantSettingOverrides",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_domains_tenant_setting_overrides, **kwargs)


class FabricAdminDeleteDomain(_FabricTool):
    name: str = "fabric_admin_delete_domain"
    description: str | None = "Deletes the specified domain."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_delete_domain(
            domain_id: str = Field(..., description="The domain ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/admin/domains/{domain_id}",
            )

        super().__init__(handler=_admin_delete_domain, **kwargs)


class FabricAdminGetDomain(_FabricTool):
    name: str = "fabric_admin_get_domain"
    description: str | None = "Returns the specified domain info. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_get_domain(
            domain_id: str = Field(..., description="The domain ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/domains/{domain_id}",
                params={"preview": "false"},
            )

        super().__init__(handler=_admin_get_domain, **kwargs)


class FabricAdminUpdateDomain(_FabricTool):
    name: str = "fabric_admin_update_domain"
    description: str | None = "Updates the specified domain info. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_update_domain(
            domain_id: str = Field(..., description="The domain ID."),
            display_name: str | None = Field(
                None, description="The domain display name. The display name cannot contain more than 40 characters."
            ),
            description: str | None = Field(
                None, description="The domain description. The description cannot contain more than 256 characters."
            ),
            default_label_id: str | None = Field(
                None,
                description=(
                    "The domain default sensitivity label. To remove the defaultLabelId from a domain, set its value to "
                    'an empty UUID in your request: "00000000-0000-0000-0000-000000000000".'
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/admin/domains/{domain_id}",
                params={"preview": "false"},
                body=_drop_none(
                    {"displayName": display_name, "description": description, "defaultLabelId": default_label_id}
                ),
            )

        super().__init__(handler=_admin_update_domain, **kwargs)


class FabricAdminAssignDomainWorkspacesByIds(_FabricTool):
    name: str = "fabric_admin_assign_domain_workspaces_by_ids"
    description: str | None = (
        "Assign workspaces to the specified domain by workspace ID. Preexisting domain assignments will be overridden "
        "unless bulk reassignment is blocked by domain management tenant settings."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_assign_domain_workspaces_by_ids(
            domain_id: str = Field(..., description="The domain ID."),
            workspaces_ids: list[Any] | None = Field(
                None, description="The workspace IDs that will be assigned to that domain."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/domains/{domain_id}/assignWorkspaces",
                body=_drop_none({"workspacesIds": workspaces_ids}),
            )

        super().__init__(handler=_admin_assign_domain_workspaces_by_ids, **kwargs)


class FabricAdminAssignDomainWorkspacesByCapacities(_FabricTool):
    name: str = "fabric_admin_assign_domain_workspaces_by_capacities"
    description: str | None = (
        "Assign all workspaces that reside on the specified capacities to the specified domain. Preexisting domain "
        "assignments will be overridden unless bulk reassignment is blocked by domain management tenant settings."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_assign_domain_workspaces_by_capacities(
            domain_id: str = Field(..., description="The domain ID."),
            capacities_ids: list[Any] | None = Field(None, description="The capacity IDs."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/domains/{domain_id}/assignWorkspacesByCapacities",
                body=_drop_none({"capacitiesIds": capacities_ids}),
            )

        super().__init__(handler=_admin_assign_domain_workspaces_by_capacities, **kwargs)


class FabricAdminAssignDomainWorkspacesByPrincipals(_FabricTool):
    name: str = "fabric_admin_assign_domain_workspaces_by_principals"
    description: str | None = (
        "Assign workspaces to the specified domain, when one of the specified principals has admin permission in the "
        "workspace. Preexisting domain assignments will be overridden unless bulk reassignment is blocked by domain "
        "management tenant settings."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_assign_domain_workspaces_by_principals(
            domain_id: str = Field(..., description="The domain ID."),
            principals: list[Any] | None = Field(None, description="The principals that are admins of the workspaces."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/domains/{domain_id}/assignWorkspacesByPrincipals",
                body=_drop_none({"principals": principals}),
            )

        super().__init__(handler=_admin_assign_domain_workspaces_by_principals, **kwargs)


class FabricAdminListDomainRoleAssignments(_FabricTool):
    name: str = "fabric_admin_list_domain_role_assignments"
    description: str | None = "Returns a list of domain role assignments."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_domain_role_assignments(
            domain_id: str = Field(..., description="The domain ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/domains/{domain_id}/roleAssignments",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_domain_role_assignments, **kwargs)


class FabricAdminBulkAssignDomainRoleAssignments(_FabricTool):
    name: str = "fabric_admin_bulk_assign_domain_role_assignments"
    description: str | None = "Assign the specified admins or contributors to the domain."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_bulk_assign_domain_role_assignments(
            domain_id: str = Field(..., description="The domain ID."),
            type: str = Field(
                ...,
                description=(
                    "Represents the domain members by the principal's request type. Additional request types may be added "
                    "over time. Allowed values: Admin, Contributor."
                ),
            ),
            principals: list[Any] | None = Field(
                None, description="The principals that will be assigned to the domain role."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/domains/{domain_id}/roleAssignments/bulkAssign",
                body=_drop_none({"type": type, "principals": principals}),
            )

        super().__init__(handler=_admin_bulk_assign_domain_role_assignments, **kwargs)


class FabricAdminBulkUnassignDomainRoleAssignments(_FabricTool):
    name: str = "fabric_admin_bulk_unassign_domain_role_assignments"
    description: str | None = "Unassign the specified admins or contributors from the domain."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_bulk_unassign_domain_role_assignments(
            domain_id: str = Field(..., description="The domain ID."),
            type: str = Field(
                ...,
                description=(
                    "Represents the domain members by the principal's request type. Additional request types may be added "
                    "over time. Allowed values: Admin, Contributor."
                ),
            ),
            principals: list[Any] | None = Field(
                None, description="The principals that will be unassigned from the domain role."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/domains/{domain_id}/roleAssignments/bulkUnassign",
                body=_drop_none({"type": type, "principals": principals}),
            )

        super().__init__(handler=_admin_bulk_unassign_domain_role_assignments, **kwargs)


class FabricAdminSyncDomainRoleAssignmentsToSubdomains(_FabricTool):
    name: str = "fabric_admin_sync_domain_role_assignments_to_subdomains"
    description: str | None = "Sync the role assignments from the specified domain to its subdomains."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_sync_domain_role_assignments_to_subdomains(
            domain_id: str = Field(..., description="The domain ID."),
            role: str = Field(
                ...,
                description=(
                    "Represents the domain members by the principal's request type. Additional request types may be added "
                    "over time. Allowed values: Admin, Contributor."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/domains/{domain_id}/roleAssignments/syncToSubdomains",
                body=_drop_none({"role": role}),
            )

        super().__init__(handler=_admin_sync_domain_role_assignments_to_subdomains, **kwargs)


class FabricAdminUnassignAllDomainWorkspaces(_FabricTool):
    name: str = "fabric_admin_unassign_all_domain_workspaces"
    description: str | None = "Unassign all workspaces from the specified domain."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_unassign_all_domain_workspaces(
            domain_id: str = Field(..., description="The domain ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/domains/{domain_id}/unassignAllWorkspaces",
            )

        super().__init__(handler=_admin_unassign_all_domain_workspaces, **kwargs)


class FabricAdminUnassignDomainWorkspacesByIds(_FabricTool):
    name: str = "fabric_admin_unassign_domain_workspaces_by_ids"
    description: str | None = "Unassign workspaces from the specified domain by workspace ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_unassign_domain_workspaces_by_ids(
            domain_id: str = Field(..., description="The domain ID."),
            workspaces_ids: list[Any] | None = Field(
                None, description="The workspace IDs that will be unassigned from that domain."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/domains/{domain_id}/unassignWorkspaces",
                body=_drop_none({"workspacesIds": workspaces_ids}),
            )

        super().__init__(handler=_admin_unassign_domain_workspaces_by_ids, **kwargs)


class FabricAdminListDomainWorkspaces(_FabricTool):
    name: str = "fabric_admin_list_domain_workspaces"
    description: str | None = "Returns a list of the workspaces assigned to the specified domain."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_domain_workspaces(
            domain_id: str = Field(..., description="The domain ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/domains/{domain_id}/workspaces",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_domain_workspaces, **kwargs)


class FabricAdminListItems(_FabricTool):
    name: str = "fabric_admin_list_items"
    description: str | None = "Returns a list of active Fabric and PowerBI items. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_items(
            workspace_id: str | None = Field(None, description="The workspace ID."),
            capacity_id: str | None = Field(None, description="The capacity ID of the workspace."),
            state: str | None = Field(None, description="The item state. Supported states are active."),
            type: str | None = Field(None, description="The item type."),
            continuation_token: str | None = Field(
                None, description="Continuous token used to get the next page items."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/items",
                params={
                    "workspaceId": workspace_id,
                    "capacityId": capacity_id,
                    "state": state,
                    "type": type,
                    "continuationToken": continuation_token,
                },
            )

        super().__init__(handler=_admin_list_items, **kwargs)


class FabricAdminBulkRemoveItemLabels(_FabricTool):
    name: str = "fabric_admin_bulk_remove_item_labels"
    description: str | None = (
        "Remove sensitivity labels from Fabric items (such as lakehouses and reports) by item ID. The sensitivity "
        "labels of the autogenerated items linked to the items in the call, are removed and their IDs aren't "
        "returned. Items with linked autogenerated items that are supported are: Lakehouse, Warehouse, Datamart, "
        "SQLDatabase, MirroredDatabase. For a usage example, see Set or remove sensitivity labels."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_bulk_remove_item_labels(
            items: list[Any] | None = Field(None, description="A list of items."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/admin/items/bulkRemoveLabels",
                body=_drop_none({"items": items}),
            )

        super().__init__(handler=_admin_bulk_remove_item_labels, **kwargs)


class FabricAdminBulkRemoveItemSharingLinks(_FabricTool):
    name: str = "fabric_admin_bulk_remove_item_sharing_links"
    description: str | None = (
        "Deletes all organization sharing links for the specified Fabric items. This action cannot be undone. Preview "
        "API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_bulk_remove_item_sharing_links(
            items: list[Any] = Field(..., description="A list of items. The list includes item ID and type."),
            sharing_link_type: str = Field(
                ...,
                description=(
                    "Specifies the type of sharing link that is required to be deleted for each Fabric item. Additional "
                    "sharing link types may be added over time. Allowed values: OrgLink."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/admin/items/bulkRemoveSharingLinks",
                body=_drop_none({"items": items, "sharingLinkType": sharing_link_type}),
            )

        super().__init__(handler=_admin_bulk_remove_item_sharing_links, **kwargs)


class FabricAdminBulkSetItemLabels(_FabricTool):
    name: str = "fabric_admin_bulk_set_item_labels"
    description: str | None = (
        "Set sensitivity labels on Fabric items, such as lakehouses and reports, by item ID. The sensitivity labels "
        "are applied to the autogenerated items related to the requested items, and their IDs aren't returned. Items "
        "with linked autogenerated items that are supported are: Lakehouse, Warehouse, Datamart, SQLDatabase and "
        "MirroredDatabase. To set a sensitivity label using this API the admin user or the delegated user, if "
        "provided, must have the label included in their label policy. For a usage example see: Set or remove "
        "sensitivity labels."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_bulk_set_item_labels(
            items: list[Any] = Field(..., description="A list of items. The list includes item ID and type."),
            label_id: str = Field(..., description="The label ID, which must be in the user's label policy."),
            delegated_principal: dict[str, Any] | None = Field(
                None, description="Represents an identity or a Microsoft Entra group."
            ),
            assignment_method: str | None = Field(
                None,
                description=(
                    "Specifies whether the assigned label was set by an automated process or manually. Additional tenant "
                    "setting property types may be added over time. Allowed values: Standard, Priviledged."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/admin/items/bulkSetLabels",
                body=_drop_none(
                    {
                        "items": items,
                        "labelId": label_id,
                        "delegatedPrincipal": delegated_principal,
                        "assignmentMethod": assignment_method,
                    }
                ),
            )

        super().__init__(handler=_admin_bulk_set_item_labels, **kwargs)


class FabricAdminListExternalDataShares(_FabricTool):
    name: str = "fabric_admin_list_external_data_shares"
    description: str | None = "Lists the external data shares in the tenant."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_external_data_shares(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/items/externalDataShares",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_external_data_shares, **kwargs)


class FabricAdminRemoveAllItemSharingLinks(_FabricTool):
    name: str = "fabric_admin_remove_all_item_sharing_links"
    description: str | None = (
        "Deletes all organization sharing links for all Fabric items in the tenant. This action cannot be undone. "
        "Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_remove_all_item_sharing_links(
            sharing_link_type: str = Field(
                ...,
                description=(
                    "Specifies the type of sharing link that is required to be deleted for each Fabric item. Additional "
                    "sharing link types may be added over time. Allowed values: OrgLink."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/admin/items/removeAllSharingLinks",
                body=_drop_none({"sharingLinkType": sharing_link_type}),
            )

        super().__init__(handler=_admin_remove_all_item_sharing_links, **kwargs)


class FabricAdminListTags(_FabricTool):
    name: str = "fabric_admin_list_tags"
    description: str | None = "Returns a list of all the tenant's tags."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_tags(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/tags",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_tags, **kwargs)


class FabricAdminBulkCreateTags(_FabricTool):
    name: str = "fabric_admin_bulk_create_tags"
    description: str | None = "Create new tags."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_bulk_create_tags(
            create_tags_request: list[Any] = Field(..., description="An array of createTagRequest"),
            scope: dict[str, Any] | None = Field(None, description="Represents a tag scope"),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/admin/tags/bulkCreateTags",
                body=_drop_none({"scope": scope, "createTagsRequest": create_tags_request}),
            )

        super().__init__(handler=_admin_bulk_create_tags, **kwargs)


class FabricAdminDeleteTag(_FabricTool):
    name: str = "fabric_admin_delete_tag"
    description: str | None = "Delete the specified tag."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_delete_tag(
            tag_id: str = Field(..., description="The tag ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/admin/tags/{tag_id}",
            )

        super().__init__(handler=_admin_delete_tag, **kwargs)


class FabricAdminUpdateTag(_FabricTool):
    name: str = "fabric_admin_update_tag"
    description: str | None = "Updates the specified tag."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_update_tag(
            tag_id: str = Field(..., description="The tag ID."),
            display_name: str = Field(
                ..., description="The tag display name. The display name cannot contain more than 40 characters."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/admin/tags/{tag_id}",
                body=_drop_none({"displayName": display_name}),
            )

        super().__init__(handler=_admin_update_tag, **kwargs)


class FabricAdminListTenantSettings(_FabricTool):
    name: str = "fabric_admin_list_tenant_settings"
    description: str | None = "Returns a list of the tenant settings."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_tenant_settings(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/tenantsettings",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_tenant_settings, **kwargs)


class FabricAdminUpdateTenantSetting(_FabricTool):
    name: str = "fabric_admin_update_tenant_setting"
    description: str | None = "Update a given tenant setting."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_update_tenant_setting(
            tenant_setting_name: str = Field(..., description="The name of tenant setting."),
            enabled: bool = Field(
                ..., description="The status of the tenant setting. False - Disabled, True - Enabled."
            ),
            enabled_security_groups: list[Any] | None = Field(None, description="A list of enabled security groups."),
            excluded_security_groups: list[Any] | None = Field(None, description="A list of excluded security groups."),
            properties: list[Any] | None = Field(None, description="Tenant setting properties."),
            delegate_to_capacity: bool | None = Field(
                None,
                description=(
                    "Indicates whether the tenant setting can be delegated to a capacity admin. False - Capacity admin "
                    "cannot override the tenant setting. True - Capacity admin can override the tenant setting."
                ),
            ),
            delegate_to_domain: bool | None = Field(
                None,
                description=(
                    "Indicates whether the tenant setting can be delegated to a domain admin. False - Domain admin cannot "
                    "override the tenant setting. True - Domain admin can override the tenant setting."
                ),
            ),
            delegate_to_workspace: bool | None = Field(
                None,
                description=(
                    "Indicates whether the tenant setting can be delegated to a workspace admin. False - Workspace admin "
                    "cannot override the tenant setting. True - Workspace admin can override the tenant setting."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/tenantsettings/{_quote(tenant_setting_name)}/update",
                body=_drop_none(
                    {
                        "enabled": enabled,
                        "enabledSecurityGroups": enabled_security_groups,
                        "excludedSecurityGroups": excluded_security_groups,
                        "properties": properties,
                        "delegateToCapacity": delegate_to_capacity,
                        "delegateToDomain": delegate_to_domain,
                        "delegateToWorkspace": delegate_to_workspace,
                    }
                ),
            )

        super().__init__(handler=_admin_update_tenant_setting, **kwargs)


class FabricAdminListUserAccessEntities(_FabricTool):
    name: str = "fabric_admin_list_user_access_entities"
    description: str | None = (
        "Returns a list of permission details for Fabric and PowerBI items the specified user can access. Preview "
        "API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_user_access_entities(
            user_id: str = Field(..., description="The user graph ID or User Principal Name (UPN)."),
            type: str | None = Field(None, description="The item type."),
            continuation_token: str | None = Field(
                None, description="Continuous token used to get the next page items."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/users/{_quote(user_id)}/access",
                params={"type": type, "continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_user_access_entities, **kwargs)


class FabricAdminListWorkloads(_FabricTool):
    name: str = "fabric_admin_list_workloads"
    description: str | None = (
        "Returns all workloads, optionally filtered by assignment status. If AssignmentStatus is not specified or set "
        "to AssignmentStatus.Any, the result will include assigned workloads and unassigned workloads. Note: Only "
        "published workloads are visible."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_workloads(
            assignment_status: str | None = Field(
                None,
                description=(
                    "Assignment status of workloads. Additional assignment statuses may be added over time. Allowed "
                    "values: Any, Assigned, Unassigned."
                ),
            ),
            continuation_token: str | None = Field(
                None, description="Continuation token. Used to get the next items in the list."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/workloads",
                params={"assignmentStatus": assignment_status, "continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_workloads, **kwargs)


class FabricAdminListWorkloadAssignments(_FabricTool):
    name: str = "fabric_admin_list_workload_assignments"
    description: str | None = (
        "List all workload assignments. The result contains a list of assignments which can be filtered by the "
        "workload ID and assignment type. Note: Only published workloads are visible."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_workload_assignments(
            continuation_token: str | None = Field(
                None, description="Continuation token. Used to get the next items in the list."
            ),
            type: str | None = Field(
                None,
                description=(
                    "Assignment type. Additional assignment types may be added over time. Allowed values: Capacity, "
                    "Workspace, Tenant."
                ),
            ),
            workload_id: str | None = Field(None, description="WorkloadID filter."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/workloads/assignments",
                params={"continuationToken": continuation_token, "Type": type, "workloadId": workload_id},
            )

        super().__init__(handler=_admin_list_workload_assignments, **kwargs)


class FabricAdminCreateWorkloadAssignment(_FabricTool):
    name: str = "fabric_admin_create_workload_assignment"
    description: str | None = "Create workload assignment."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_create_workload_assignment(
            type: str = Field(
                ...,
                description=(
                    "The type of workload assignment. Specifies whether a workload is assigned to a capacity, workspace, "
                    "or tenant. Additional assignment types may be added over time. Allowed values: Capacity, Workspace, "
                    "Tenant."
                ),
            ),
            workload_id: str = Field(..., description="The unique identifier of the workload to assign."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/admin/workloads/assignments",
                body=_drop_none({"type": type, "workloadId": workload_id}),
            )

        super().__init__(handler=_admin_create_workload_assignment, **kwargs)


class FabricAdminDeleteWorkloadAssignment(_FabricTool):
    name: str = "fabric_admin_delete_workload_assignment"
    description: str | None = "Delete workload assignment."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_delete_workload_assignment(
            assignment_id: str = Field(..., description="Assignment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/admin/workloads/assignments/{assignment_id}",
            )

        super().__init__(handler=_admin_delete_workload_assignment, **kwargs)


class FabricAdminListWorkspaces(_FabricTool):
    name: str = "fabric_admin_list_workspaces"
    description: str | None = (
        "Returns a list of workspaces. A maximum of 10,000 records can be returned per request. With the continuation "
        "token provided in the response, you can get the next 10,000 records."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_workspaces(
            type: str | None = Field(
                None, description="The workspace type. Supported types are personal, workspace, adminworkspace."
            ),
            capacity_id: str | None = Field(None, description="The capacity ID of the workspace."),
            name: str | None = Field(None, description="The workspace name."),
            state: str | None = Field(
                None, description="The workspace state. Supported states are active and deleted."
            ),
            continuation_token: str | None = Field(
                None, description="Continuation token. Used to get the next items in the list."
            ),
            encryption_status: str | None = Field(
                None,
                description=(
                    "Indicates the CMK encryption status of the workspace and is used to filter workspaces that match the "
                    "specified status."
                ),
            ),
            include: str | None = Field(
                None,
                description=(
                    "Specifies additional data to include for each workspace in the response. Supported values: "
                    "`encryption`."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/workspaces",
                params={
                    "type": type,
                    "capacityId": capacity_id,
                    "name": name,
                    "state": state,
                    "continuationToken": continuation_token,
                    "encryptionStatus": encryption_status,
                    "include": include,
                },
            )

        super().__init__(handler=_admin_list_workspaces, **kwargs)


class FabricAdminListWorkspacesTenantSettingOverrides(_FabricTool):
    name: str = "fabric_admin_list_workspaces_tenant_setting_overrides"
    description: str | None = (
        "Returns list of workspace delegation setting overrides. A maximum of 10,000 records can be returned per "
        "request. With the continuation token provided in the response, you can get the next 10,000 records."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_workspaces_tenant_setting_overrides(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/workspaces/delegatedTenantSettingOverrides",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_workspaces_tenant_setting_overrides, **kwargs)


class FabricAdminListWorkspaceGitConnections(_FabricTool):
    name: str = "fabric_admin_list_workspace_git_connections"
    description: str | None = "Returns a list of Git connections. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_workspace_git_connections(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/workspaces/discoverGitConnections",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_admin_list_workspace_git_connections, **kwargs)


class FabricAdminListWorkspaceCommunicationPolicies(_FabricTool):
    name: str = "fabric_admin_list_workspace_communication_policies"
    description: str | None = (
        "Returns network communication policy settings for all workspaces in the tenant. Returns paginated network "
        "communication policy details for all workspaces. The response includes inbound access protection settings "
        "(public access rules and IP firewall rules), outbound access protection settings (public access rules, "
        "connection rules, gateway rules, Git policy, and managed private endpoints), and workspace metadata. With "
        "the continuation token provided in the response, you can get the next set of records."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_workspace_communication_policies(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
            filter: str | None = Field(
                None,
                description=(
                    "Filters workspaces by policy direction. Supported filter expressions: "
                    "`inbound/publicAccessRules/defaultAction eq 'deny'`, `outbound/publicAccessRules/defaultAction eq "
                    "'deny'`, or both combined with `or`."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/admin/workspaces/networking/communicationpolicies",
                params={"continuationToken": continuation_token, "filter": filter},
            )

        super().__init__(handler=_admin_list_workspace_communication_policies, **kwargs)


class FabricAdminGetWorkspace(_FabricTool):
    name: str = "fabric_admin_get_workspace"
    description: str | None = "Returns the specified workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_get_workspace(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/workspaces/{workspace_id}",
            )

        super().__init__(handler=_admin_get_workspace, **kwargs)


class FabricAdminGrantWorkspaceTemporaryAccess(_FabricTool):
    name: str = "fabric_admin_grant_workspace_temporary_access"
    description: str | None = "Grants admin temporary (24h) access to a user's 'My Workspace'."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_grant_workspace_temporary_access(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/workspaces/{workspace_id}/grantAdminTemporaryAccess",
            )

        super().__init__(handler=_admin_grant_workspace_temporary_access, **kwargs)


class FabricAdminGetItem(_FabricTool):
    name: str = "fabric_admin_get_item"
    description: str | None = "Returns the specified Fabric or PowerBI item. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_get_item(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID"),
            type: str | None = Field(
                None,
                description=(
                    "The type of the item. When querying for the following types, this parameter is required: * Report * "
                    "Dashboard * SemanticModel * App * Dataflow"
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/workspaces/{workspace_id}/items/{item_id}",
                params={"type": type},
            )

        super().__init__(handler=_admin_get_item, **kwargs)


class FabricAdminRevokeItemExternalDataShare(_FabricTool):
    name: str = "fabric_admin_revoke_item_external_data_share"
    description: str | None = "Revokes the specified external data share. This action cannot be undone."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_revoke_item_external_data_share(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            external_data_share_id: str = Field(..., description="The external data share ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/workspaces/{workspace_id}/items/{item_id}/externalDataShares/{external_data_share_id}/revoke",
            )

        super().__init__(handler=_admin_revoke_item_external_data_share, **kwargs)


class FabricAdminListItemAccessDetails(_FabricTool):
    name: str = "fabric_admin_list_item_access_details"
    description: str | None = (
        "Returns a list of users (including groups and service principals) and lists their workspace roles. Preview "
        "API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_item_access_details(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            type: str | None = Field(
                None,
                description=(
                    "The type of the item. When querying for the following types, this parameter is required: * Report * "
                    "Dashboard * SemanticModel * App * Dataflow"
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/workspaces/{workspace_id}/items/{item_id}/users",
                params={"type": type},
            )

        super().__init__(handler=_admin_list_item_access_details, **kwargs)


class FabricAdminRemoveWorkspaceTemporaryAccess(_FabricTool):
    name: str = "fabric_admin_remove_workspace_temporary_access"
    description: str | None = "Removes admin temporary access from a user's 'My Workspace'."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_remove_workspace_temporary_access(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/workspaces/{workspace_id}/removeAdminTemporaryAccess",
            )

        super().__init__(handler=_admin_remove_workspace_temporary_access, **kwargs)


class FabricAdminRestoreWorkspace(_FabricTool):
    name: str = "fabric_admin_restore_workspace"
    description: str | None = "Restores a deleted workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_restore_workspace(
            workspace_id: str = Field(..., description="The workspace ID."),
            new_workspace_name: str | None = Field(
                None, description="The name of the workspace. Mandatory if the restore request is for *My workspace*."
            ),
            new_workspace_admin_principal: dict[str, Any] | None = Field(
                None, description="Represents an identity or a Microsoft Entra group."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/admin/workspaces/{workspace_id}/restore",
                body=_drop_none(
                    {
                        "newWorkspaceName": new_workspace_name,
                        "newWorkspaceAdminPrincipal": new_workspace_admin_principal,
                    }
                ),
            )

        super().__init__(handler=_admin_restore_workspace, **kwargs)


class FabricAdminListWorkspaceAccessDetails(_FabricTool):
    name: str = "fabric_admin_list_workspace_access_details"
    description: str | None = (
        "Returns a list of users (including groups and ServicePrincipals) that have access to the specified "
        "workspace. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _admin_list_workspace_access_details(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/admin/workspaces/{workspace_id}/users",
            )

        super().__init__(handler=_admin_list_workspace_access_details, **kwargs)
