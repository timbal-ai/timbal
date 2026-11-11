"""Microsoft Fabric REST API tools: deployment pipelines, networking, OneLake, shortcuts and access control.

Auth, long running operations and request handling live in ``fabric.py``.
"""

from typing import Any

from pydantic import Field

from .fabric import _drop_none, _fabric_request, _FabricTool, _quote, _quote_path


class FabricListDeploymentPipelines(_FabricTool):
    name: str = "fabric_list_deployment_pipelines"
    description: str | None = "Returns a list of deployment pipelines the user can access."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deployment_pipelines(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/deploymentPipelines",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_deployment_pipelines, **kwargs)


class FabricCreateDeploymentPipeline(_FabricTool):
    name: str = "fabric_create_deployment_pipeline"
    description: str | None = "Creates a new deployment pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_deployment_pipeline(
            display_name: str = Field(
                ...,
                description=(
                    "The display name for the deployment pipeline.<br>The display name cannot contain more than 256 "
                    "characters."
                ),
            ),
            stages: list[Any] = Field(..., description="The collection of deployment pipeline stages."),
            description: str | None = Field(
                None,
                description=(
                    "The description for the deployment pipeline.<br>The description cannot contain more than 1024 "
                    "characters."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/deploymentPipelines",
                body=_drop_none({"displayName": display_name, "description": description, "stages": stages}),
            )

        super().__init__(handler=_create_deployment_pipeline, **kwargs)


class FabricDeleteDeploymentPipeline(_FabricTool):
    name: str = "fabric_delete_deployment_pipeline"
    description: str | None = "Deletes the specified deployment pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_deployment_pipeline(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/deploymentPipelines/{deployment_pipeline_id}",
            )

        super().__init__(handler=_delete_deployment_pipeline, **kwargs)


class FabricGetDeploymentPipeline(_FabricTool):
    name: str = "fabric_get_deployment_pipeline"
    description: str | None = "Returns the specified deployment pipeline metadata."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_deployment_pipeline(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/deploymentPipelines/{deployment_pipeline_id}",
            )

        super().__init__(handler=_get_deployment_pipeline, **kwargs)


class FabricUpdateDeploymentPipeline(_FabricTool):
    name: str = "fabric_update_deployment_pipeline"
    description: str | None = "Updates the properties of the specified deployment pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_deployment_pipeline(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            display_name: str | None = Field(
                None,
                description=(
                    "The display name for the deployment pipeline.<br>The display name cannot contain more than 256 "
                    "characters."
                ),
            ),
            description: str | None = Field(
                None,
                description=(
                    "The description for the deployment pipeline.<br>The description cannot contain more than 1024 "
                    "characters."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/deploymentPipelines/{deployment_pipeline_id}",
                body=_drop_none({"displayName": display_name, "description": description}),
            )

        super().__init__(handler=_update_deployment_pipeline, **kwargs)


class FabricDeployDeploymentPipelineStageContent(_FabricTool):
    name: str = "fabric_deploy_deployment_pipeline_stage_content"
    description: str | None = (
        "Deploys items from the specified stage of the specified deployment pipeline. To learn about items that are "
        "supported in deployment pipelines, see: Supported items."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _deploy_deployment_pipeline_stage_content(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            source_stage_id: str = Field(..., description="The ID of the source stage."),
            target_stage_id: str = Field(..., description="The ID of the target stage."),
            created_workspace_details: dict[str, Any] | None = Field(
                None,
                description=(
                    "The configuration details for creating a new workspace. Required when deploying to a stage that has "
                    "no assigned workspaces."
                ),
            ),
            note: str | None = Field(
                None, description="A note describing the deployment. The text size is limited to 1024 characters."
            ),
            items: list[Any] | None = Field(
                None, description="A list of items to be deployed. If not used, all supported stage items are deployed."
            ),
            options: dict[str, Any] | None = Field(
                None, description="Deployment configuration options for the deployment."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/deploymentPipelines/{deployment_pipeline_id}/deploy",
                body=_drop_none(
                    {
                        "sourceStageId": source_stage_id,
                        "targetStageId": target_stage_id,
                        "createdWorkspaceDetails": created_workspace_details,
                        "note": note,
                        "items": items,
                        "options": options,
                    }
                ),
            )

        super().__init__(handler=_deploy_deployment_pipeline_stage_content, **kwargs)


class FabricListDeploymentPipelineOperations(_FabricTool):
    name: str = "fabric_list_deployment_pipeline_operations"
    description: str | None = (
        "Returns a list of the up-to-20 most recent deploy operations performed on the specified deployment pipeline."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deployment_pipeline_operations(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/deploymentPipelines/{deployment_pipeline_id}/operations",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_deployment_pipeline_operations, **kwargs)


class FabricGetDeploymentPipelineOperation(_FabricTool):
    name: str = "fabric_get_deployment_pipeline_operation"
    description: str | None = (
        "Returns the details of the specified deploy operation performed on the specified deployment pipeline, "
        "including the deployment execution plan."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_deployment_pipeline_operation(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            operation_id: str = Field(..., description="The operation ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/deploymentPipelines/{deployment_pipeline_id}/operations/{operation_id}",
            )

        super().__init__(handler=_get_deployment_pipeline_operation, **kwargs)


class FabricListDeploymentPipelineRoleAssignments(_FabricTool):
    name: str = "fabric_list_deployment_pipeline_role_assignments"
    description: str | None = "Returns a list of deployment pipeline role assignments."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deployment_pipeline_role_assignments(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/deploymentPipelines/{deployment_pipeline_id}/roleAssignments",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_deployment_pipeline_role_assignments, **kwargs)


class FabricAddDeploymentPipelineRoleAssignment(_FabricTool):
    name: str = "fabric_add_deployment_pipeline_role_assignment"
    description: str | None = "Adds a deployment pipeline role assignment."

    def __init__(self, **kwargs: Any) -> None:
        async def _add_deployment_pipeline_role_assignment(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            principal: dict[str, Any] = Field(..., description="Represents an identity or a Microsoft Entra group."),
            role: str = Field(
                ...,
                description=(
                    "A Deployment Pipeline role. Additional Deployment Pipeline roles may be added over time. Allowed "
                    "values: Admin."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/deploymentPipelines/{deployment_pipeline_id}/roleAssignments",
                body=_drop_none({"principal": principal, "role": role}),
            )

        super().__init__(handler=_add_deployment_pipeline_role_assignment, **kwargs)


class FabricDeleteDeploymentPipelineRoleAssignment(_FabricTool):
    name: str = "fabric_delete_deployment_pipeline_role_assignment"
    description: str | None = "Deletes the specified deployment pipeline role assignment."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_deployment_pipeline_role_assignment(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            principal_id: str = Field(..., description="The principal ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/deploymentPipelines/{deployment_pipeline_id}/roleAssignments/{principal_id}",
            )

        super().__init__(handler=_delete_deployment_pipeline_role_assignment, **kwargs)


class FabricListDeploymentPipelineStages(_FabricTool):
    name: str = "fabric_list_deployment_pipeline_stages"
    description: str | None = "Returns the specified deployment pipeline stages."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deployment_pipeline_stages(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/deploymentPipelines/{deployment_pipeline_id}/stages",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_deployment_pipeline_stages, **kwargs)


class FabricGetDeploymentPipelineStage(_FabricTool):
    name: str = "fabric_get_deployment_pipeline_stage"
    description: str | None = "Returns the specified deployment pipeline stage metadata."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_deployment_pipeline_stage(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            stage_id: str = Field(..., description="The deployment pipeline stage ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}",
            )

        super().__init__(handler=_get_deployment_pipeline_stage, **kwargs)


class FabricUpdateDeploymentPipelineStage(_FabricTool):
    name: str = "fabric_update_deployment_pipeline_stage"
    description: str | None = "Updates the properties of the specified deployment pipeline stage."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_deployment_pipeline_stage(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            stage_id: str = Field(..., description="The deployment pipeline stage ID."),
            display_name: str = Field(
                ...,
                description=(
                    "The deployment pipeline stage display name.<br>The display name cannot contain more than 256 "
                    "characters."
                ),
            ),
            description: str | None = Field(
                None,
                description=(
                    "The deployment pipeline stage description.<br>The description cannot contain more than 1024 "
                    "characters."
                ),
            ),
            is_public: bool | None = Field(None, description="Whether the deployment pipeline stage is public."),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}",
                body=_drop_none({"displayName": display_name, "description": description, "isPublic": is_public}),
            )

        super().__init__(handler=_update_deployment_pipeline_stage, **kwargs)


class FabricAssignDeploymentPipelineWorkspaceToStage(_FabricTool):
    name: str = "fabric_assign_deployment_pipeline_workspace_to_stage"
    description: str | None = (
        "Assigns the specified workspace to the specified deployment pipeline stage. This operation will fail if "
        "there's an active deployment operation."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _assign_deployment_pipeline_workspace_to_stage(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            stage_id: str = Field(..., description="The deployment pipeline stage ID."),
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}/assignWorkspace",
                body=_drop_none({"workspaceId": workspace_id}),
            )

        super().__init__(handler=_assign_deployment_pipeline_workspace_to_stage, **kwargs)


class FabricListDeploymentPipelineStageItems(_FabricTool):
    name: str = "fabric_list_deployment_pipeline_stage_items"
    description: str | None = (
        "Returns the supported items from the workspace assigned to the specified stage of the specified deployment "
        "pipeline. To learn about items that are supported in deployment pipelines, see: Supported items."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_deployment_pipeline_stage_items(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            stage_id: str = Field(..., description="The deployment pipeline stage ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}/items",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_deployment_pipeline_stage_items, **kwargs)


class FabricUnassignDeploymentPipelineWorkspaceFromStage(_FabricTool):
    name: str = "fabric_unassign_deployment_pipeline_workspace_from_stage"
    description: str | None = (
        "Unassigns the workspace from the specified stage in the specified deployment pipeline. This operation will "
        "fail if there's an active deployment operation."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _unassign_deployment_pipeline_workspace_from_stage(
            deployment_pipeline_id: str = Field(..., description="The deployment pipeline ID."),
            stage_id: str = Field(..., description="The deployment pipeline stage ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/deploymentPipelines/{deployment_pipeline_id}/stages/{stage_id}/unassignWorkspace",
            )

        super().__init__(handler=_unassign_deployment_pipeline_workspace_from_stage, **kwargs)


class FabricGetExternalDataShareInvitationDetails(_FabricTool):
    name: str = "fabric_get_external_data_share_invitation_details"
    description: str | None = "Returns information about an external data share invitation."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_external_data_share_invitation_details(
            invitation_id: str = Field(..., description="The external data share invitation ID."),
            provider_tenant_id: str = Field(..., description="The external data share provider tenant ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/externalDataShares/invitations/{invitation_id}",
                params={"providerTenantId": provider_tenant_id},
            )

        super().__init__(handler=_get_external_data_share_invitation_details, **kwargs)


class FabricAcceptExternalDataShareInvitation(_FabricTool):
    name: str = "fabric_accept_external_data_share_invitation"
    description: str | None = "Accepts an external data share invitation into a specified data item."

    def __init__(self, **kwargs: Any) -> None:
        async def _accept_external_data_share_invitation(
            invitation_id: str = Field(..., description="The external data share invitation ID."),
            provider_tenant_id: str = Field(..., description="The provider tenant ID."),
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            payload: dict[str, Any] = Field(
                ..., description="Payload for the Accept External Data Share invitation request"
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/externalDataShares/invitations/{invitation_id}/accept",
                body=_drop_none(
                    {
                        "providerTenantId": provider_tenant_id,
                        "workspaceId": workspace_id,
                        "itemId": item_id,
                        "payload": payload,
                    }
                ),
            )

        super().__init__(handler=_accept_external_data_share_invitation, **kwargs)


class FabricDeprovisionWorkspaceIdentity(_FabricTool):
    name: str = "fabric_deprovision_workspace_identity"
    description: str | None = "Deprovision a workspace identity."

    def __init__(self, **kwargs: Any) -> None:
        async def _deprovision_workspace_identity(
            workspace_id: str = Field(..., description="The ID of the workspace."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/deprovisionIdentity",
            )

        super().__init__(handler=_deprovision_workspace_identity, **kwargs)


class FabricGetWorkspaceEncryption(_FabricTool):
    name: str = "fabric_get_workspace_encryption"
    description: str | None = (
        "Gets the workspace Customer-Managed Key (CMK) encryption settings. Returns the CMK encryption settings and "
        "status for the specified workspace."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_encryption(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/encryption",
            )

        super().__init__(handler=_get_workspace_encryption, **kwargs)


class FabricAssignWorkspaceEncryption(_FabricTool):
    name: str = "fabric_assign_workspace_encryption"
    description: str | None = (
        "Assigns a Customer-Managed Key (CMK) to a workspace. Enables CMK encryption for the workspace or rotates the "
        "encryption key for the workspace."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _assign_workspace_encryption(
            workspace_id: str = Field(..., description="The workspace ID."),
            key_identifier: str = Field(
                ..., description="The Azure Key Vault key identifier. This must be a versionless key URI."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/encryption/assign",
                body=_drop_none({"keyIdentifier": key_identifier}),
            )

        super().__init__(handler=_assign_workspace_encryption, **kwargs)


class FabricResetWorkspaceEncryption(_FabricTool):
    name: str = "fabric_reset_workspace_encryption"
    description: str | None = (
        "Resets the workspace encryption by removing the Customer-Managed Key (CMK) encryption configuration. After "
        "reset, the workspace data remains encrypted using Fabric's default Microsoft-managed keys."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _reset_workspace_encryption(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/encryption/reset",
            )

        super().__init__(handler=_reset_workspace_encryption, **kwargs)


class FabricListItemDataAccessRoles(_FabricTool):
    name: str = "fabric_list_item_data_access_roles"
    description: str | None = "Returns a list of OneLake roles."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_item_data_access_roles(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The ID of the Fabric item to put the roles."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_item_data_access_roles, **kwargs)


class FabricUpsertItemDataAccessRole(_FabricTool):
    name: str = "fabric_upsert_item_data_access_role"
    description: str | None = "Creates or updates (upserts) a single data access role."

    def __init__(self, **kwargs: Any) -> None:
        async def _upsert_item_data_access_role(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The ID of the Fabric item to create or update the role on."),
            name: str = Field(..., description="The name of the Data access role."),
            decision_rules: list[Any] = Field(
                ..., description="The array of permissions that make up the Data access role."
            ),
            data_access_role_conflict_policy: str | None = Field(
                None,
                description=(
                    "Determines the behavior when there are conflicting data access role assignments. Overwrite means new "
                    "assignments replace existing ones. Abort means the operation fails if there are conflicts. "
                    "Additional dataAccessRoleConflictPolicy types may be added over time. Allowed values: Overwrite, "
                    "Abort."
                ),
            ),
            kind: str | None = Field(
                None,
                description=(
                    "The kind of the Data access role. Currently, the only supported kind is `Policy`. Additional kind "
                    "types may be added over time. Allowed values: Policy."
                ),
            ),
            members: dict[str, Any] | None = Field(
                None,
                description=(
                    "The members object which contains the members of the role as arrays of different member types."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles",
                params={"dataAccessRoleConflictPolicy": data_access_role_conflict_policy},
                body=_drop_none({"name": name, "kind": kind, "decisionRules": decision_rules, "members": members}),
            )

        super().__init__(handler=_upsert_item_data_access_role, **kwargs)


class FabricSetItemDataAccessRoles(_FabricTool):
    name: str = "fabric_set_item_data_access_roles"
    description: str | None = "Creates or updates data access roles in OneLake."

    def __init__(self, **kwargs: Any) -> None:
        async def _set_item_data_access_roles(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The ID of the Fabric item to put the roles."),
            dry_run: bool | None = Field(
                None,
                description=(
                    "Used to trigger a dry run of the API call. True - The API call will trigger a dry run and no roles "
                    "will be changed. False - Will not trigger a dry run and roles will be updated."
                ),
            ),
            value: list[Any] | None = Field(
                None,
                description=(
                    "A list of roles that are used to manage data access security and ensure that only authorized users "
                    "can view certain data. A role represents a set of permissions and permission scopes that define what "
                    "actions its members are allowed to perform for the data in scope. Members are users or groups who "
                    "have been granted the role, and they can read the data based on the permissions assigned to the "
                    "role. For example, a member can be a Microsoft Entra ID group and permission scope can be a Read "
                    "Action applied on the given Path to File, Folder(s) or Table(s) in OneLake."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles",
                params={"dryRun": dry_run},
                body=_drop_none({"value": value}),
            )

        super().__init__(handler=_set_item_data_access_roles, **kwargs)


class FabricDeleteItemDataAccessRole(_FabricTool):
    name: str = "fabric_delete_item_data_access_role"
    description: str | None = "Deletes a single data access role."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_item_data_access_role(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The ID of the Fabric item to delete the role from."),
            role_name: str = Field(..., description="The name of the role to delete."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles/{_quote(role_name)}",
            )

        super().__init__(handler=_delete_item_data_access_role, **kwargs)


class FabricGetItemDataAccessRole(_FabricTool):
    name: str = "fabric_get_item_data_access_role"
    description: str | None = "Returns data access role details for the given role name."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item_data_access_role(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The ID of the Fabric item to get the role from."),
            role_name: str = Field(..., description="The name of the role to retrieve."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/dataAccessRoles/{_quote(role_name)}",
            )

        super().__init__(handler=_get_item_data_access_role, **kwargs)


class FabricListItemExternalDataShares(_FabricTool):
    name: str = "fabric_list_item_external_data_shares"
    description: str | None = "Returns a list of the external data shares that exist for the specified item."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_item_external_data_shares(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/externalDataShares",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_item_external_data_shares, **kwargs)


class FabricCreateItemExternalDataShare(_FabricTool):
    name: str = "fabric_create_item_external_data_share"
    description: str | None = "Creates an external data share for a given path or list of paths in the specified item."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_item_external_data_share(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            paths: list[Any] = Field(
                ...,
                description=(
                    "The path or list of paths that are to be externally shared. You can share up to 100 paths in each "
                    'share. A valid path to an external data share must start with "Files/" or "Tables/". You can\'t share '
                    'the root folder itself (Files or Tables). For example, these paths are valid: * "Files/MyFolder1" * '
                    '"Tables/MySchema" * "Tables/MyTable1"'
                ),
            ),
            recipient: dict[str, Any] = Field(
                ...,
                description=(
                    "The recipient of an external data share. Use 'ExternalDataShareUserRecipient' to share with a user "
                    "by email, or 'ExternalDataShareSPRecipient' to share with a service principal by object ID. If "
                    "'type' is not specified, it defaults to 'User'."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/externalDataShares",
                body=_drop_none({"paths": paths, "recipient": recipient}),
            )

        super().__init__(handler=_create_item_external_data_share, **kwargs)


class FabricDeleteItemExternalDataShare(_FabricTool):
    name: str = "fabric_delete_item_external_data_share"
    description: str | None = "Deletes the specified external data share."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_item_external_data_share(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            external_data_share_id: str = Field(..., description="The external data share ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/items/{item_id}/externalDataShares/{external_data_share_id}",
            )

        super().__init__(handler=_delete_item_external_data_share, **kwargs)


class FabricGetItemExternalDataShare(_FabricTool):
    name: str = "fabric_get_item_external_data_share"
    description: str | None = "Returns the details of the specified external data share."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item_external_data_share(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            external_data_share_id: str = Field(..., description="The external data share ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/externalDataShares/{external_data_share_id}",
            )

        super().__init__(handler=_get_item_external_data_share, **kwargs)


class FabricRevokeItemExternalDataShare(_FabricTool):
    name: str = "fabric_revoke_item_external_data_share"
    description: str | None = "Revokes the specified external data share. This action cannot be undone."

    def __init__(self, **kwargs: Any) -> None:
        async def _revoke_item_external_data_share(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            external_data_share_id: str = Field(..., description="The external data share ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/externalDataShares/{external_data_share_id}/revoke",
            )

        super().__init__(handler=_revoke_item_external_data_share, **kwargs)


class FabricAssignItemDefaultIdentity(_FabricTool):
    name: str = "fabric_assign_item_default_identity"
    description: str | None = "Associates the default identity with an item. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _assign_item_default_identity(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            assignment_type: str = Field(
                ...,
                description=(
                    "Specifies the type of identity for an associate identity request. Additional identity assignment "
                    "types may be added over time. Allowed values: Caller."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/identities/default/assign",
                params={"beta": "true"},
                body=_drop_none({"assignmentType": assignment_type}),
            )

        super().__init__(handler=_assign_item_default_identity, **kwargs)


class FabricListItemShortcuts(_FabricTool):
    name: str = "fabric_list_item_shortcuts"
    description: str | None = "Returns a list of shortcuts for the item, including all the subfolders exhaustively."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_item_shortcuts(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            parent_path: str | None = Field(None, description="The starting path from which to retrieve the shortcuts"),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/shortcuts",
                params={"parentPath": parent_path, "continuationToken": continuation_token},
            )

        super().__init__(handler=_list_item_shortcuts, **kwargs)


class FabricCreateItemShortcut(_FabricTool):
    name: str = "fabric_create_item_shortcut"
    description: str | None = "Creates a new shortcut or updates an existing shortcut."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_item_shortcut(
            workspace_id: str = Field(..., description="The ID of the workspace."),
            item_id: str = Field(..., description="The ID of the data item."),
            path: str = Field(
                ...,
                description=(
                    'A string representing the full path where the shortcut is created, including either "Files" or '
                    '"Tables".'
                ),
            ),
            name: str = Field(..., description="Name of the shortcut."),
            target: dict[str, Any] = Field(
                ...,
                description=(
                    "An object that contains the target datasource, and must specify exactly one of the supported "
                    "destinations as described in the table below."
                ),
            ),
            shortcut_conflict_policy: str | None = Field(
                None,
                description=(
                    "When provided, it defines the action to take when a shortcut with the same name and path already "
                    "exists. The default action is 'Abort'. Additional ShortcutConflictPolicy types may be added over "
                    "time. Allowed values: Abort, GenerateUniqueName, CreateOrOverwrite, OverwriteOnly."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/shortcuts",
                params={"shortcutConflictPolicy": shortcut_conflict_policy},
                body=_drop_none({"path": path, "name": name, "target": target}),
            )

        super().__init__(handler=_create_item_shortcut, **kwargs)


class FabricBulkCreateItemShortcuts(_FabricTool):
    name: str = "fabric_bulk_create_item_shortcuts"
    description: str | None = "Creates bulk shortcuts."

    def __init__(self, **kwargs: Any) -> None:
        async def _bulk_create_item_shortcuts(
            workspace_id: str = Field(..., description="The ID of the workspace."),
            item_id: str = Field(..., description="The ID of the data item."),
            create_shortcut_requests: list[Any] = Field(..., description="A list of shortcut creation requests."),
            shortcut_conflict_policy: str | None = Field(
                None,
                description=(
                    "When provided, it defines the action to take when a shortcut with the same name and path already "
                    "exists. The default action is 'Abort'. Additional ShortcutConflictPolicy types may be added over "
                    "time. Allowed values: Abort, GenerateUniqueName, CreateOrOverwrite, OverwriteOnly."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/items/{item_id}/shortcuts/bulkCreate",
                params={"shortcutConflictPolicy": shortcut_conflict_policy},
                body=_drop_none({"createShortcutRequests": create_shortcut_requests}),
            )

        super().__init__(handler=_bulk_create_item_shortcuts, **kwargs)


class FabricDeleteItemShortcut(_FabricTool):
    name: str = "fabric_delete_item_shortcut"
    description: str | None = "Deletes the shortcut but does not delete the destination storage folder."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_item_shortcut(
            workspace_id: str = Field(..., description="The ID of the workspace."),
            item_id: str = Field(..., description="The ID of the data item."),
            shortcut_path: str = Field(
                ...,
                description="The path of the shortcut to be deleted. For more information see: Directory and file names.",
            ),
            shortcut_name: str = Field(
                ...,
                description="The name of the shortcut to delete. For more information see: Directory and file names.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/items/{item_id}/shortcuts/{_quote_path(shortcut_path)}/{_quote(shortcut_name)}",
            )

        super().__init__(handler=_delete_item_shortcut, **kwargs)


class FabricGetItemShortcut(_FabricTool):
    name: str = "fabric_get_item_shortcut"
    description: str | None = "Returns shortcut properties."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_item_shortcut(
            workspace_id: str = Field(..., description="The ID of the workspace."),
            item_id: str = Field(..., description="The ID of the data item."),
            shortcut_path: str = Field(
                ...,
                description="The creation path of the shortcut. For more information see: Directory and file names.",
            ),
            shortcut_name: str = Field(
                ..., description="The name of the shortcut. For more information see: Directory and file names."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/shortcuts/{_quote_path(shortcut_path)}/{_quote(shortcut_name)}",
            )

        super().__init__(handler=_get_item_shortcut, **kwargs)


class FabricListWorkspaceManagedPrivateEndpoints(_FabricTool):
    name: str = "fabric_list_workspace_managed_private_endpoints"
    description: str | None = "Returns a list of managed private endpoints associated with the specified workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspace_managed_private_endpoints(
            workspace_id: str = Field(..., description="The workspace ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/managedPrivateEndpoints",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_workspace_managed_private_endpoints, **kwargs)


class FabricCreateWorkspaceManagedPrivateEndpoint(_FabricTool):
    name: str = "fabric_create_workspace_managed_private_endpoint"
    description: str | None = "Creates a managed private endpoint in the specified workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_workspace_managed_private_endpoint(
            workspace_id: str = Field(..., description="The workspace ID."),
            name: str = Field(..., description="The private endpoint name. Should not be more than 64 characters."),
            target_private_link_resource_id: str = Field(
                ..., description="Resource Id of data source for which private endpoint needs to be created."
            ),
            target_subresource_type: str | None = Field(
                None, description="Sub-resource pointing to Private-link resoure."
            ),
            request_message: str | None = Field(
                None, description="Message to approve private endpoint request. Should not be more than 140 characters."
            ),
            target_fqd_ns: list[Any] | None = Field(
                None,
                description=(
                    "Fully qualified domain names (FQDNs) to be associated with the private endpoint. Should not be more "
                    "than 20 FQDNs."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/managedPrivateEndpoints",
                body=_drop_none(
                    {
                        "name": name,
                        "targetPrivateLinkResourceId": target_private_link_resource_id,
                        "targetSubresourceType": target_subresource_type,
                        "requestMessage": request_message,
                        "targetFQDNs": target_fqd_ns,
                    }
                ),
            )

        super().__init__(handler=_create_workspace_managed_private_endpoint, **kwargs)


class FabricDeleteWorkspaceManagedPrivateEndpoint(_FabricTool):
    name: str = "fabric_delete_workspace_managed_private_endpoint"
    description: str | None = "Deletes the specified managed private endpoint."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_workspace_managed_private_endpoint(
            workspace_id: str = Field(..., description="The workspace ID."),
            managed_private_endpoint_id: str = Field(..., description="The managed private endpoint ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/managedPrivateEndpoints/{managed_private_endpoint_id}",
            )

        super().__init__(handler=_delete_workspace_managed_private_endpoint, **kwargs)


class FabricGetWorkspaceManagedPrivateEndpoint(_FabricTool):
    name: str = "fabric_get_workspace_managed_private_endpoint"
    description: str | None = "Gets the specified managed private endpoint."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_managed_private_endpoint(
            workspace_id: str = Field(..., description="The workspace ID."),
            managed_private_endpoint_id: str = Field(..., description="The managed private endpoint ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/managedPrivateEndpoints/{managed_private_endpoint_id}",
            )

        super().__init__(handler=_get_workspace_managed_private_endpoint, **kwargs)


class FabricGetWorkspaceCommunicationPolicy(_FabricTool):
    name: str = "fabric_get_workspace_communication_policy"
    description: str | None = "Returns the networking communication policy for the specified workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_communication_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/networking/communicationPolicy",
            )

        super().__init__(handler=_get_workspace_communication_policy, **kwargs)


class FabricSetWorkspaceCommunicationPolicy(_FabricTool):
    name: str = "fabric_set_workspace_communication_policy"
    description: str | None = "Sets the networking communication policy for the specified workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _set_workspace_communication_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
            inbound: dict[str, Any] | None = Field(
                None, description="The policy for all inbound communications to a workspace."
            ),
            outbound: dict[str, Any] | None = Field(
                None, description="The policy for all outbound communications from a workspace."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/networking/communicationPolicy",
                body=_drop_none({"inbound": inbound, "outbound": outbound}),
            )

        super().__init__(handler=_set_workspace_communication_policy, **kwargs)


class FabricGetWorkspaceInboundAzureResourceRules(_FabricTool):
    name: str = "fabric_get_workspace_inbound_azure_resource_rules"
    description: str | None = (
        "Returns the inbound Azure resource instance rules for a workspace. This API is designed to help workspace "
        "administrators view the effective inbound Azure resource instance rules in their workspace settings. This "
        "feature is currently in preview. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_inbound_azure_resource_rules(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/inbound/azureResources",
            )

        super().__init__(handler=_get_workspace_inbound_azure_resource_rules, **kwargs)


class FabricSetWorkspaceInboundAzureResourceRules(_FabricTool):
    name: str = "fabric_set_workspace_inbound_azure_resource_rules"
    description: str | None = (
        "Sets the inbound Azure resource instance rules for a workspace. This API enables workspace administrators to "
        "set inbound Azure resource instance rules that control which Azure resource instances are in the allowed "
        "list for a workspace. This feature is currently in preview. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _set_workspace_inbound_azure_resource_rules(
            workspace_id: str = Field(..., description="The workspace ID."),
            rules: list[Any] | None = Field(
                None, description="An array of inbound Azure resource instance rules associated with the workspace."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/inbound/azureResources",
                body=_drop_none({"rules": rules}),
            )

        super().__init__(handler=_set_workspace_inbound_azure_resource_rules, **kwargs)


class FabricGetWorkspaceInboundExternalDataSharesPolicy(_FabricTool):
    name: str = "fabric_get_workspace_inbound_external_data_shares_policy"
    description: str | None = (
        "Returns the inbound External Data Shares bypass policy for a workspace. This API is designed to help "
        "workspace administrators view whether External Data Shares traffic is allowed to bypass inbound networking "
        "restrictions."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_inbound_external_data_shares_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/inbound/externalDataShares",
            )

        super().__init__(handler=_get_workspace_inbound_external_data_shares_policy, **kwargs)


class FabricSetWorkspaceInboundExternalDataSharesPolicy(_FabricTool):
    name: str = "fabric_set_workspace_inbound_external_data_shares_policy"
    description: str | None = (
        "Sets the inbound External Data Shares bypass policy for a workspace. This API enables workspace "
        "administrators to allow or deny External Data Shares traffic to bypass inbound networking restrictions. This "
        "feature is currently in preview. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _set_workspace_inbound_external_data_shares_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
            default_action: str | None = Field(
                None,
                description=(
                    "The default option for a network communications policy. Additional network connection defaults may "
                    "be added over time. Allowed values: Allow, Deny."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/inbound/externalDataShares",
                body=_drop_none({"defaultAction": default_action}),
            )

        super().__init__(handler=_set_workspace_inbound_external_data_shares_policy, **kwargs)


class FabricGetWorkspaceFirewallRules(_FabricTool):
    name: str = "fabric_get_workspace_firewall_rules"
    description: str | None = (
        "Returns the IP firewall rules for the workspace. This API is designed to help workspace administrators view "
        "the effective IP firewall rules. This feature is currently in preview. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_firewall_rules(
            workspace_id: str = Field(
                ..., description="Unique identifier of the workspace whose firewall rules are being queried."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/inbound/firewall",
            )

        super().__init__(handler=_get_workspace_firewall_rules, **kwargs)


class FabricSetWorkspaceFirewallRules(_FabricTool):
    name: str = "fabric_set_workspace_firewall_rules"
    description: str | None = (
        "Sets the IP firewall rules for the workspace. This API enables workspace administrators to set IP firewall "
        "rules that control which IP addresses are to be allowed to connect to the workspace. This feature is "
        "currently in preview. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _set_workspace_firewall_rules(
            workspace_id: str = Field(..., description="Unique identifier of the workspace to update."),
            rules: list[Any] | None = Field(
                None,
                description=(
                    "A list of rules that define IP addresses permitted for inbound access. Each rule may include a name "
                    "and a single IP address, an IP address range, or a CIDR IP address. A maximum of 256 rules can be "
                    "specified per workspace."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/inbound/firewall",
                body=_drop_none({"rules": rules}),
            )

        super().__init__(handler=_set_workspace_firewall_rules, **kwargs)


class FabricGetWorkspaceOutboundCloudConnectionRules(_FabricTool):
    name: str = "fabric_get_workspace_outbound_cloud_connection_rules"
    description: str | None = (
        "Returns the cloud connection rules for the workspace enabled with Outbound Access Protection (OAP). This API "
        "helps workspace administrators view the effective outbound network communication policies enforced for cloud "
        "connections. Cloud connection rules are returned and applied only if the workspace’s network communication "
        "policy has `outbound.publicAccessRules.defaultAction` set to `Deny`. If OAP is not enabled for the "
        "workspace, the API fails because outbound connections are not being restricted."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_outbound_cloud_connection_rules(
            workspace_id: str = Field(
                ..., description="Unique identifier of the workspace whose outbound rules are being queried."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/outbound/connections",
            )

        super().__init__(handler=_get_workspace_outbound_cloud_connection_rules, **kwargs)


class FabricSetWorkspaceOutboundCloudConnectionRules(_FabricTool):
    name: str = "fabric_set_workspace_outbound_cloud_connection_rules"
    description: str | None = "Sets the outbound access protection cloud connection rules for the workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _set_workspace_outbound_cloud_connection_rules(
            workspace_id: str = Field(..., description="Unique identifier of the workspace to update."),
            default_action: str | None = Field(
                None,
                description=(
                    "Defines the access control behavior for outbound connections. This enum is used for the field "
                    "defaultAction to specify whether outbound communication should be allowed or denied by default. This "
                    "type enables both global and connection-specific control over outbound access, helping enforce "
                    "secure and predictable network communication policies. Additional connection access action types may "
                    "be added over time. Allowed values: Allow, Deny."
                ),
            ),
            rules: list[Any] | None = Field(
                None,
                description=(
                    "A list of rules that define outbound access behavior for specific cloud connection types. Each rule "
                    "may include endpoint-based or workspace-based restrictions depending on supported connection types."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/outbound/connections",
                body=_drop_none({"defaultAction": default_action, "rules": rules}),
            )

        super().__init__(handler=_set_workspace_outbound_cloud_connection_rules, **kwargs)


class FabricGetWorkspaceOutboundGatewayRules(_FabricTool):
    name: str = "fabric_get_workspace_outbound_gateway_rules"
    description: str | None = (
        "Returns the gateway rules for the workspace enabled with Outbound Access Protection (OAP). This API helps "
        "workspace administrators view the effective outbound network communication policies enforced for on-premises "
        "and VNet data gateways. Gateway rules are returned and applied only if the workspace’s network communication "
        "policy has `outbound.publicAccessRules.defaultAction` set to `Deny`. If OAP is not enabled for the "
        "workspace, the API fails because outbound connections are not being restricted."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_outbound_gateway_rules(
            workspace_id: str = Field(
                ..., description="Unique identifier of the workspace whose outbound rules are being queried."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/outbound/gateways",
            )

        super().__init__(handler=_get_workspace_outbound_gateway_rules, **kwargs)


class FabricSetWorkspaceOutboundGatewayRules(_FabricTool):
    name: str = "fabric_set_workspace_outbound_gateway_rules"
    description: str | None = "Sets the gateway rules for the workspace enabled with Outbound Access Protection (OAP)."

    def __init__(self, **kwargs: Any) -> None:
        async def _set_workspace_outbound_gateway_rules(
            workspace_id: str = Field(..., description="Unique identifier of the workspace to update."),
            default_action: str | None = Field(
                None,
                description=(
                    "Defines the access control behavior for outbound gateways. This enum is used for the field "
                    "defaultAction to specify whether outbound communication should be allowed or denied by default. This "
                    "type enables both global and gateway-specific control over outbound access, helping enforce secure "
                    "and predictable network communication policies. Additional gateway access action types may be added "
                    "over time. Allowed values: Allow, Deny."
                ),
            ),
            allowed_gateways: list[Any] | None = Field(
                None, description="A list of rules that define outbound access behavior for gateways."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/outbound/gateways",
                body=_drop_none({"defaultAction": default_action, "allowedGateways": allowed_gateways}),
            )

        super().__init__(handler=_set_workspace_outbound_gateway_rules, **kwargs)


class FabricGetWorkspaceOutboundGitPolicy(_FabricTool):
    name: str = "fabric_get_workspace_outbound_git_policy"
    description: str | None = (
        "Returns Git Outbound policy for the specified workspace. In cases the workspace restricts outbound policy, a "
        "workspace admin needs to allow the use of Git integration on the specified workspace."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_outbound_git_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/outbound/git",
            )

        super().__init__(handler=_get_workspace_outbound_git_policy, **kwargs)


class FabricSetWorkspaceOutboundGitPolicy(_FabricTool):
    name: str = "fabric_set_workspace_outbound_git_policy"
    description: str | None = (
        "Sets Git Outbound policy for the specified workspace, when Outbound policy is set to 'Deny'."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _set_workspace_outbound_git_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
            default_action: str | None = Field(
                None,
                description=(
                    "The default option for a network communications policy. Additional network connection defaults may "
                    "be added over time. Allowed values: Allow, Deny."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/networking/communicationPolicy/outbound/git",
                body=_drop_none({"defaultAction": default_action}),
            )

        super().__init__(handler=_set_workspace_outbound_git_policy, **kwargs)


class FabricExportOnelakeLifecyclePolicy(_FabricTool):
    name: str = "fabric_export_onelake_lifecycle_policy"
    description: str | None = (
        "Exports the OneLake lifecycle management policy for a workspace. Returns the lifecycle management policy for "
        "a workspace. A lifecycle policy is made up of rules, which are made up of filters, actions, and conditions. "
        'A prefixMatch of "*/diagnosticLogs" applies the rule to all diagnostic events routed to that workspace. '
        "OneLake supports the same lifecycle policy structure as Azure Storage lifecycle management, with the "
        "following exceptions: - OneLake does not support the TierToArchive action. - OneLake does not support the "
        "Delete action. - OneLake does not support the blobIndexMatch filter. For more information, see OneLake "
        "lifecycle management."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _export_onelake_lifecycle_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/onelake/lifecycle/exportPolicy",
            )

        super().__init__(handler=_export_onelake_lifecycle_policy, **kwargs)


class FabricImportOnelakeLifecyclePolicy(_FabricTool):
    name: str = "fabric_import_onelake_lifecycle_policy"
    description: str | None = (
        "Imports a OneLake lifecycle management policy for a workspace. Creates or replaces the OneLake lifecycle "
        "management policy for a workspace. The policy must be sent in full as a complete replacement. To delete an "
        "existing policy, send a request with an empty rules array. In OneLake, lifecycle rules are scoped to the "
        "workspace. The prefixMatch filter starts at the item level, so prefixMatch values must begin with a valid "
        'item name or item ID (for example, "MyLakehouse.Lakehouse/Files/data"). Setting the prefixMatch to '
        '"*/diagnosticLogs" applies the rule to all diagnostic events routed to that workspace. OneLake supports the '
        "same lifecycle policy structure as Azure Storage lifecycle management, with the following exceptions: - "
        "OneLake does not support the TierToArchive action. - OneLake does not support the Delete action. - OneLake "
        "does not support the blobIndexMatch filter. For more information, see OneLake lifecycle management."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _import_onelake_lifecycle_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
            properties: dict[str, Any] = Field(
                ...,
                description="The lifecycle policy properties. The structure follows Azure Storage lifecycle management.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/onelake/lifecycle/importPolicy",
                body=_drop_none({"properties": properties}),
            )

        super().__init__(handler=_import_onelake_lifecycle_policy, **kwargs)


class FabricResetOnelakeShortcutCache(_FabricTool):
    name: str = "fabric_reset_onelake_shortcut_cache"
    description: str | None = "Deletes any cached files that were stored while reading from shortcuts."

    def __init__(self, **kwargs: Any) -> None:
        async def _reset_onelake_shortcut_cache(
            workspace_id: str = Field(..., description="The ID of the workspace."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/onelake/resetShortcutCache",
            )

        super().__init__(handler=_reset_onelake_shortcut_cache, **kwargs)


class FabricGetOnelakeSettings(_FabricTool):
    name: str = "fabric_get_onelake_settings"
    description: str | None = "Get workspace OneLake settings."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_onelake_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/onelake/settings",
            )

        super().__init__(handler=_get_onelake_settings, **kwargs)


class FabricModifyOnelakeSettingsAccessTimeTracking(_FabricTool):
    name: str = "fabric_modify_onelake_settings_access_time_tracking"
    description: str | None = (
        "Enables or disables OneLake access time tracking for a workspace. Enables or disables Azure Storage "
        "`LastAccessTime` tracking for OneLake files in the workspace. When enabled, file reads and writes update the "
        "file's last access timestamp; metadata-only operations, such as checking properties, metadata, or tags, do "
        "not. Changes take effect immediately."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _modify_onelake_settings_access_time_tracking(
            workspace_id: str = Field(..., description="The workspace ID."),
            status: str = Field(
                ...,
                description=(
                    "The status of OneLake access time tracking. When enabled, Azure Storage updates the LastAccessTime "
                    "property on file reads and writes. Additional status values may be added over time. Allowed values: "
                    "Enabled, Disabled."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/onelake/settings/modifyAccessTimeTracking",
                body=_drop_none({"status": status}),
            )

        super().__init__(handler=_modify_onelake_settings_access_time_tracking, **kwargs)


class FabricModifyOnelakeSettingsDefaultTier(_FabricTool):
    name: str = "fabric_modify_onelake_settings_default_tier"
    description: str | None = (
        "Modifies the default OneLake storage tier for a workspace. All files without an explicitly set tier will be "
        "moved to the new default tier. You will be billed for any transactions resulting from movement between "
        "tiers."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _modify_onelake_settings_default_tier(
            workspace_id: str = Field(..., description="The workspace ID."),
            default_tier: str = Field(
                ...,
                description=(
                    "The new default access tier for the workspace. Additional access tier values may be added over time. "
                    "Allowed values: Hot, Cool, Cold."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/onelake/settings/modifyDefaultTier",
                params={"defaultTier": default_tier},
            )

        super().__init__(handler=_modify_onelake_settings_default_tier, **kwargs)


class FabricModifyOnelakeSettingsDiagnostics(_FabricTool):
    name: str = "fabric_modify_onelake_settings_diagnostics"
    description: str | None = "Enables or disables workspace OneLake diagnostic settings."

    def __init__(self, **kwargs: Any) -> None:
        async def _modify_onelake_settings_diagnostics(
            workspace_id: str = Field(..., description="The workspace ID."),
            status: str = Field(..., description="The status of the diagnostics settings."),
            destination: dict[str, Any] | None = Field(
                None, description="The destination where OneLake diagnostic logs are stored."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/onelake/settings/modifyDiagnostics",
                body=_drop_none({"status": status, "destination": destination}),
            )

        super().__init__(handler=_modify_onelake_settings_diagnostics, **kwargs)


class FabricModifyOnelakeSettingsImmutabilityPolicy(_FabricTool):
    name: str = "fabric_modify_onelake_settings_immutability_policy"
    description: str | None = (
        "Create or update OneLake immutability settings. Set immutability policy for data stored in OneLake. "
        "Currently, this feature supports configuring a retention period specifically for diagnostic logs within a "
        "workspace, ensuring they remain unaltered once written."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _modify_onelake_settings_immutability_policy(
            workspace_id: str = Field(..., description="The workspace ID."),
            scope: str = Field(
                ...,
                description=(
                    "The scope of immutability policy. Additional Immutability Scope types may be added over time. "
                    "Allowed values: DiagnosticLogs."
                ),
            ),
            retention_days: int = Field(..., description="Retention Days for the action."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/onelake/settings/modifyImmutabilityPolicy",
                body=_drop_none({"scope": scope, "retentionDays": retention_days}),
            )

        super().__init__(handler=_modify_onelake_settings_immutability_policy, **kwargs)


class FabricProvisionWorkspaceIdentity(_FabricTool):
    name: str = "fabric_provision_workspace_identity"
    description: str | None = "Provision a workspace identity for a workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _provision_workspace_identity(
            workspace_id: str = Field(..., description="The ID of the workspace."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/provisionIdentity",
            )

        super().__init__(handler=_provision_workspace_identity, **kwargs)


class FabricListWorkspaceRoleAssignments(_FabricTool):
    name: str = "fabric_list_workspace_role_assignments"
    description: str | None = "Returns a list of workspace role assignments."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspace_role_assignments(
            workspace_id: str = Field(..., description="The workspace ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/roleAssignments",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_workspace_role_assignments, **kwargs)


class FabricAddWorkspaceRoleAssignment(_FabricTool):
    name: str = "fabric_add_workspace_role_assignment"
    description: str | None = (
        "Adds a workspace role assignment. To get the principal user object ID required for request body, see Find "
        "the user object ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _add_workspace_role_assignment(
            workspace_id: str = Field(..., description="The workspace ID."),
            principal: dict[str, Any] = Field(..., description="Represents an identity or a Microsoft Entra group."),
            role: str = Field(
                ...,
                description=(
                    "A Workspace role. Additional workspace roles may be added over time. Allowed values: Admin, Member, "
                    "Contributor, Viewer."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/roleAssignments",
                body=_drop_none({"principal": principal, "role": role}),
            )

        super().__init__(handler=_add_workspace_role_assignment, **kwargs)


class FabricDeleteWorkspaceRoleAssignment(_FabricTool):
    name: str = "fabric_delete_workspace_role_assignment"
    description: str | None = (
        "Deletes the specified workspace role assignment. The role assignment of the last admin can't be deleted."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_workspace_role_assignment(
            workspace_id: str = Field(..., description="The workspace ID."),
            workspace_role_assignment_id: str = Field(..., description="The workspace role assignment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/roleAssignments/{workspace_role_assignment_id}",
            )

        super().__init__(handler=_delete_workspace_role_assignment, **kwargs)


class FabricGetWorkspaceRoleAssignment(_FabricTool):
    name: str = "fabric_get_workspace_role_assignment"
    description: str | None = "Returns information of a workspace role assignment."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_role_assignment(
            workspace_id: str = Field(..., description="The workspace ID."),
            workspace_role_assignment_id: str = Field(..., description="The workspace role assignment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/roleAssignments/{workspace_role_assignment_id}",
            )

        super().__init__(handler=_get_workspace_role_assignment, **kwargs)


class FabricUpdateWorkspaceRoleAssignment(_FabricTool):
    name: str = "fabric_update_workspace_role_assignment"
    description: str | None = (
        "Updates a workspace role assignment. The role assignment of the last admin can't be changed."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_workspace_role_assignment(
            workspace_id: str = Field(..., description="The workspace ID."),
            workspace_role_assignment_id: str = Field(..., description="The workspace role assignment ID."),
            role: str = Field(
                ...,
                description=(
                    "A Workspace role. Additional workspace roles may be added over time. Allowed values: Admin, Member, "
                    "Contributor, Viewer."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/roleAssignments/{workspace_role_assignment_id}",
                body=_drop_none({"role": role}),
            )

        super().__init__(handler=_update_workspace_role_assignment, **kwargs)
