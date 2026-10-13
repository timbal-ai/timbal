"""Microsoft Fabric REST API tools: connections and gateways.

Auth, long running operations and request handling live in ``fabric.py``.
"""

from typing import Any

from pydantic import Field

from .fabric import _drop_none, _fabric_request, _FabricTool


class FabricListConnections(_FabricTool):
    name: str = "fabric_list_connections"
    description: str | None = (
        "Returns a list of on-premises, virtual network and cloud connections the user has permission for."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_connections(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/connections",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_connections, **kwargs)


class FabricCreateConnection(_FabricTool):
    name: str = "fabric_create_connection"
    description: str | None = (
        "Creates a connection. To encrypt credentials, see Configure credentials programmatically."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_connection(
            connectivity_type: str = Field(
                ...,
                description=(
                    "The connectivity type of the connection. Additional connectivity types may be added over time. "
                    "Allowed values: ShareableCloud, PersonalCloud, OnPremisesGateway, OnPremisesGatewayPersonal, "
                    "VirtualNetworkGateway, StreamingVirtualNetworkGateway, Automatic, None."
                ),
            ),
            display_name: str = Field(
                ..., description="The display name of the connection. Maximum length is 200 characters."
            ),
            connection_details: dict[str, Any] = Field(
                ..., description="The connection details input for create operations."
            ),
            privacy_level: str | None = Field(
                None,
                description=(
                    "The privacy level setting of the connection. Additional privacy levels may be added over time. "
                    "Allowed values: None, Private, Organizational, Public."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/connections",
                body=_drop_none(
                    {
                        "connectivityType": connectivity_type,
                        "displayName": display_name,
                        "connectionDetails": connection_details,
                        "privacyLevel": privacy_level,
                    }
                ),
            )

        super().__init__(handler=_create_connection, **kwargs)


class FabricListSupportedConnectionTypes(_FabricTool):
    name: str = "fabric_list_supported_connection_types"
    description: str | None = "Lists supported connection types."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_supported_connection_types(
            gateway_id: str | None = Field(
                None,
                description=(
                    "The gateway to list supported connection types. If omitted, the API lists supported connection types "
                    "in the cloud."
                ),
            ),
            show_all_creation_methods: bool | None = Field(
                None,
                description=(
                    "Setting that controls whether to show all creation methods. True - Show all creation methods, False "
                    "- Show only recommended creation methods."
                ),
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/connections/supportedConnectionTypes",
                params={
                    "gatewayId": gateway_id,
                    "showAllCreationMethods": show_all_creation_methods,
                    "continuationToken": continuation_token,
                },
            )

        super().__init__(handler=_list_supported_connection_types, **kwargs)


class FabricDeleteConnection(_FabricTool):
    name: str = "fabric_delete_connection"
    description: str | None = "Delete connection by ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_connection(
            connection_id: str = Field(..., description="The ID of the connection."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/connections/{connection_id}",
            )

        super().__init__(handler=_delete_connection, **kwargs)


class FabricGetConnection(_FabricTool):
    name: str = "fabric_get_connection"
    description: str | None = "Get connection by ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_connection(
            connection_id: str = Field(..., description="The ID of the connection."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/connections/{connection_id}",
            )

        super().__init__(handler=_get_connection, **kwargs)


class FabricUpdateConnection(_FabricTool):
    name: str = "fabric_update_connection"
    description: str | None = (
        "Updates connection by ID. To encrypt credentials, see Configure credentials programmatically."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_connection(
            connection_id: str = Field(..., description="The ID of the connection."),
            connectivity_type: str = Field(
                ...,
                description=(
                    "The connectivity type of the connection. Additional connectivity types may be added over time. "
                    "Allowed values: ShareableCloud, PersonalCloud, OnPremisesGateway, OnPremisesGatewayPersonal, "
                    "VirtualNetworkGateway, StreamingVirtualNetworkGateway, Automatic, None."
                ),
            ),
            privacy_level: str | None = Field(
                None,
                description=(
                    "The privacy level setting of the connection. Additional privacy levels may be added over time. "
                    "Allowed values: None, Private, Organizational, Public."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/connections/{connection_id}",
                body=_drop_none({"connectivityType": connectivity_type, "privacyLevel": privacy_level}),
            )

        super().__init__(handler=_update_connection, **kwargs)


class FabricListConnectionRoleAssignments(_FabricTool):
    name: str = "fabric_list_connection_role_assignments"
    description: str | None = "Returns a list of connection role assignments."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_connection_role_assignments(
            connection_id: str = Field(..., description="The ID of the connection."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/connections/{connection_id}/roleAssignments",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_connection_role_assignments, **kwargs)


class FabricAddConnectionRoleAssignment(_FabricTool):
    name: str = "fabric_add_connection_role_assignment"
    description: str | None = (
        "Adds a connection role assignment. To get the principal user ID required for request body, see Find the user "
        "ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _add_connection_role_assignment(
            connection_id: str = Field(..., description="The ID of the connection."),
            principal: dict[str, Any] = Field(..., description="Represents an identity or a Microsoft Entra group."),
            role: str = Field(
                ...,
                description=(
                    "A Connection role. Additional connection roles may be added over time. Allowed values: User, "
                    "UserWithReshare, Owner."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/connections/{connection_id}/roleAssignments",
                body=_drop_none({"principal": principal, "role": role}),
            )

        super().__init__(handler=_add_connection_role_assignment, **kwargs)


class FabricDeleteConnectionRoleAssignment(_FabricTool):
    name: str = "fabric_delete_connection_role_assignment"
    description: str | None = (
        "Delete the specified role assignment for the connection. To get the principal user ID required for request "
        "body, see Find the user ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_connection_role_assignment(
            connection_id: str = Field(..., description="The ID of the connection"),
            connection_role_assignment_id: str = Field(..., description="The ID of the role assignment"),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/connections/{connection_id}/roleAssignments/{connection_role_assignment_id}",
            )

        super().__init__(handler=_delete_connection_role_assignment, **kwargs)


class FabricGetConnectionRoleAssignment(_FabricTool):
    name: str = "fabric_get_connection_role_assignment"
    description: str | None = (
        "Returns the principal's role assignment for the connection. To get the principal user ID required for "
        "request body, see Find the user ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_connection_role_assignment(
            connection_id: str = Field(..., description="The ID of the connection"),
            connection_role_assignment_id: str = Field(..., description="The ID of the connection role assignment"),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/connections/{connection_id}/roleAssignments/{connection_role_assignment_id}",
            )

        super().__init__(handler=_get_connection_role_assignment, **kwargs)


class FabricUpdateConnectionRoleAssignment(_FabricTool):
    name: str = "fabric_update_connection_role_assignment"
    description: str | None = (
        "Updates the principal's role assignment for the connection. To get the principal user ID required for "
        "request body, see Find the user ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_connection_role_assignment(
            connection_id: str = Field(..., description="The ID of the connection"),
            connection_role_assignment_id: str = Field(..., description="The ID of the role assignment"),
            role: str = Field(
                ...,
                description=(
                    "A Connection role. Additional connection roles may be added over time. Allowed values: User, "
                    "UserWithReshare, Owner."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/connections/{connection_id}/roleAssignments/{connection_role_assignment_id}",
                body=_drop_none({"role": role}),
            )

        super().__init__(handler=_update_connection_role_assignment, **kwargs)


class FabricTestConnection(_FabricTool):
    name: str = "fabric_test_connection"
    description: str | None = "Tests the connection."

    def __init__(self, **kwargs: Any) -> None:
        async def _test_connection(
            connection_id: str = Field(..., description="The ID of the connection."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/connections/{connection_id}/testConnection",
            )

        super().__init__(handler=_test_connection, **kwargs)


class FabricListGateways(_FabricTool):
    name: str = "fabric_list_gateways"
    description: str | None = (
        "Returns a list of all gateways the user has permission for, including on-premises, on-premises (personal "
        "mode), and virtual network gateways."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_gateways(
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                "/gateways",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_gateways, **kwargs)


class FabricCreateGateway(_FabricTool):
    name: str = "fabric_create_gateway"
    description: str | None = "Creates a gateway."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_gateway(
            type: str = Field(
                ...,
                description=(
                    "The type of the gateway. Additional gateway types may be added over time. Allowed values: "
                    "OnPremises, OnPremisesPersonal, VirtualNetwork, StreamingVirtualNetwork."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                "/gateways",
                body=_drop_none({"type": type}),
            )

        super().__init__(handler=_create_gateway, **kwargs)


class FabricDeleteGateway(_FabricTool):
    name: str = "fabric_delete_gateway"
    description: str | None = "Delete gateway by ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_gateway(
            gateway_id: str = Field(..., description="The ID of the gateway."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/gateways/{gateway_id}",
            )

        super().__init__(handler=_delete_gateway, **kwargs)


class FabricGetGateway(_FabricTool):
    name: str = "fabric_get_gateway"
    description: str | None = "Get gateway by ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_gateway(
            gateway_id: str = Field(..., description="The ID of the gateway."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/gateways/{gateway_id}",
            )

        super().__init__(handler=_get_gateway, **kwargs)


class FabricUpdateGateway(_FabricTool):
    name: str = "fabric_update_gateway"
    description: str | None = "Updates gateway by ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_gateway(
            gateway_id: str = Field(..., description="The ID of the gateway."),
            type: str = Field(
                ...,
                description=(
                    "The type of the gateway. Additional gateway types may be added over time. Allowed values: "
                    "OnPremises, OnPremisesPersonal, VirtualNetwork, StreamingVirtualNetwork."
                ),
            ),
            display_name: str | None = Field(
                None, description="The name of the gateway. Maximum length is 200 characters."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/gateways/{gateway_id}",
                body=_drop_none({"type": type, "displayName": display_name}),
            )

        super().__init__(handler=_update_gateway, **kwargs)


class FabricCheckGatewayStatus(_FabricTool):
    name: str = "fabric_check_gateway_status"
    description: str | None = "Checks the status of the specified gateway."

    def __init__(self, **kwargs: Any) -> None:
        async def _check_gateway_status(
            gateway_id: str = Field(..., description="The ID of the gateway."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/gateways/{gateway_id}/checkStatus",
            )

        super().__init__(handler=_check_gateway_status, **kwargs)


class FabricListGatewayMembers(_FabricTool):
    name: str = "fabric_list_gateway_members"
    description: str | None = "Lists gateway members of an OnPremisesGateway by ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_gateway_members(
            gateway_id: str = Field(..., description="The ID of the gateway."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/gateways/{gateway_id}/members",
            )

        super().__init__(handler=_list_gateway_members, **kwargs)


class FabricDeleteGatewayMember(_FabricTool):
    name: str = "fabric_delete_gateway_member"
    description: str | None = "Delete gateway member of an OnPremisesGateway by ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_gateway_member(
            gateway_id: str = Field(..., description="The ID of the gateway."),
            gateway_member_id: str = Field(..., description="The ID of the gateway member."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/gateways/{gateway_id}/members/{gateway_member_id}",
            )

        super().__init__(handler=_delete_gateway_member, **kwargs)


class FabricUpdateGatewayMember(_FabricTool):
    name: str = "fabric_update_gateway_member"
    description: str | None = "Updates gateway member of an OnPremisesGateway by ID."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_gateway_member(
            gateway_id: str = Field(..., description="The ID of the gateway."),
            gateway_member_id: str = Field(..., description="The ID of the gateway member."),
            display_name: str | None = Field(
                None, description="The display name of the gateway member. Maximum length is 200 characters."
            ),
            enabled: bool | None = Field(
                None, description="Whether the gateway member is enabled. True - Enabled, False - Not enabled."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/gateways/{gateway_id}/members/{gateway_member_id}",
                body=_drop_none({"displayName": display_name, "enabled": enabled}),
            )

        super().__init__(handler=_update_gateway_member, **kwargs)


class FabricCheckGatewayMemberStatus(_FabricTool):
    name: str = "fabric_check_gateway_member_status"
    description: str | None = "Checks the status of the specified gateway member."

    def __init__(self, **kwargs: Any) -> None:
        async def _check_gateway_member_status(
            gateway_id: str = Field(..., description="The ID of the gateway."),
            gateway_member_id: str = Field(..., description="The ID of the gateway member."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/gateways/{gateway_id}/members/{gateway_member_id}/checkStatus",
            )

        super().__init__(handler=_check_gateway_member_status, **kwargs)


class FabricRestartGateway(_FabricTool):
    name: str = "fabric_restart_gateway"
    description: str | None = "Restarts the specified gateway."

    def __init__(self, **kwargs: Any) -> None:
        async def _restart_gateway(
            gateway_id: str = Field(..., description="The ID of the gateway."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/gateways/{gateway_id}/restart",
            )

        super().__init__(handler=_restart_gateway, **kwargs)


class FabricListGatewayRoleAssignments(_FabricTool):
    name: str = "fabric_list_gateway_role_assignments"
    description: str | None = "Returns a list of gateway role assignments."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_gateway_role_assignments(
            gateway_id: str = Field(..., description="The ID of the gateway."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/gateways/{gateway_id}/roleAssignments",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_gateway_role_assignments, **kwargs)


class FabricAddGatewayRoleAssignment(_FabricTool):
    name: str = "fabric_add_gateway_role_assignment"
    description: str | None = (
        "Adds a gateway role assignment. To get the principal user ID required for request body, see Find the user ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _add_gateway_role_assignment(
            gateway_id: str = Field(..., description="The ID of the gateway."),
            principal: dict[str, Any] = Field(..., description="Represents an identity or a Microsoft Entra group."),
            role: str = Field(
                ...,
                description=(
                    "A Gateway role. Additional gateway roles may be added over time. Allowed values: Admin, "
                    "ConnectionCreatorWithResharing, ConnectionCreator."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/gateways/{gateway_id}/roleAssignments",
                body=_drop_none({"principal": principal, "role": role}),
            )

        super().__init__(handler=_add_gateway_role_assignment, **kwargs)


class FabricDeleteGatewayRoleAssignment(_FabricTool):
    name: str = "fabric_delete_gateway_role_assignment"
    description: str | None = (
        "Delete the specified role assignment for the gateway. To get the principal user ID required for request "
        "body, see Find the user ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_gateway_role_assignment(
            gateway_id: str = Field(..., description="The ID of the gateway"),
            gateway_role_assignment_id: str = Field(..., description="The ID of the role assignment"),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/gateways/{gateway_id}/roleAssignments/{gateway_role_assignment_id}",
            )

        super().__init__(handler=_delete_gateway_role_assignment, **kwargs)


class FabricGetGatewayRoleAssignment(_FabricTool):
    name: str = "fabric_get_gateway_role_assignment"
    description: str | None = (
        "Returns the principal's role assignment for the gateway. To get the principal user ID required for request "
        "body, see Find the user ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_gateway_role_assignment(
            gateway_id: str = Field(..., description="The ID of the gateway"),
            gateway_role_assignment_id: str = Field(..., description="The ID of the gateway role assignment."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/gateways/{gateway_id}/roleAssignments/{gateway_role_assignment_id}",
            )

        super().__init__(handler=_get_gateway_role_assignment, **kwargs)


class FabricUpdateGatewayRoleAssignment(_FabricTool):
    name: str = "fabric_update_gateway_role_assignment"
    description: str | None = (
        "Updates the principal's role assignment for the gateway. To get the principal user ID required for request "
        "body, see Find the user ID."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_gateway_role_assignment(
            gateway_id: str = Field(..., description="The ID of the gateway"),
            gateway_role_assignment_id: str = Field(..., description="The ID of the role assignment"),
            role: str = Field(
                ...,
                description=(
                    "A Gateway role. Additional gateway roles may be added over time. Allowed values: Admin, "
                    "ConnectionCreatorWithResharing, ConnectionCreator."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/gateways/{gateway_id}/roleAssignments/{gateway_role_assignment_id}",
                body=_drop_none({"role": role}),
            )

        super().__init__(handler=_update_gateway_role_assignment, **kwargs)


class FabricShutdownGateway(_FabricTool):
    name: str = "fabric_shutdown_gateway"
    description: str | None = "Shuts down the specified gateway."

    def __init__(self, **kwargs: Any) -> None:
        async def _shutdown_gateway(
            gateway_id: str = Field(..., description="The ID of the gateway."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/gateways/{gateway_id}/shutdown",
            )

        super().__init__(handler=_shutdown_gateway, **kwargs)


class FabricListItemConnections(_FabricTool):
    name: str = "fabric_list_item_connections"
    description: str | None = "Returns the list of connections that the specified item is connected to."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_item_connections(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/items/{item_id}/connections",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_item_connections, **kwargs)
