"""Microsoft Fabric REST API tools: warehouses, SQL databases and endpoints, eventstreams, KQL, ML models and data agents.

Auth, long running operations and request handling live in ``fabric.py``.
"""

from typing import Any

from pydantic import Field

from .fabric import _drop_none, _fabric_request, _FabricTool, _quote


class FabricListDataAgentDatasources(_FabricTool):
    name: str = "fabric_list_data_agent_datasources"
    description: str | None = (
        "Returns a list of published datasources for the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_agent_datasources(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_data_agent_datasources, **kwargs)


class FabricGetDataAgentDatasource(_FabricTool):
    name: str = "fabric_get_data_agent_datasource"
    description: str | None = (
        "Returns metadata for a specific published datasource of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_agent_datasource(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources/{datasource_id}",
            )

        super().__init__(handler=_get_data_agent_datasource, **kwargs)


class FabricListDataAgentDatasourceElements(_FabricTool):
    name: str = "fabric_list_data_agent_datasource_elements"
    description: str | None = (
        "Returns schema elements at a level of the schema tree for a published datasource. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_agent_datasource_elements(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            root_id: str | None = Field(
                None, description="The identifier of a parent element. Omit to get the root level."
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources/{datasource_id}/elements",
                params={"rootId": root_id, "continuationToken": continuation_token},
            )

        super().__init__(handler=_list_data_agent_datasource_elements, **kwargs)


class FabricListDataAgentDatasourceFewshots(_FabricTool):
    name: str = "fabric_list_data_agent_datasource_fewshots"
    description: str | None = (
        "Returns a list of published fewshots for a datasource of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_agent_datasource_fewshots(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources/{datasource_id}/fewshots",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_data_agent_datasource_fewshots, **kwargs)


class FabricGetDataAgentDatasourceFewshot(_FabricTool):
    name: str = "fabric_get_data_agent_datasource_fewshot"
    description: str | None = "Returns a specific published fewshot by ID. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_agent_datasource_fewshot(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            few_shot_id: str = Field(..., description="The fewshot ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/datasources/{datasource_id}/fewshots/{few_shot_id}",
            )

        super().__init__(handler=_get_data_agent_datasource_fewshot, **kwargs)


class FabricGetDataAgentSettings(_FabricTool):
    name: str = "fabric_get_data_agent_settings"
    description: str | None = "Returns the published settings for the specified DataAgent. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_agent_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/settings",
            )

        super().__init__(handler=_get_data_agent_settings, **kwargs)


class FabricListDataAgentStagingDatasources(_FabricTool):
    name: str = "fabric_list_data_agent_staging_datasources"
    description: str | None = (
        "Returns a list of staging datasources for the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_agent_staging_datasources(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_data_agent_staging_datasources, **kwargs)


class FabricCreateDataAgentStagingDatasource(_FabricTool):
    name: str = "fabric_create_data_agent_staging_datasource"
    description: str | None = (
        "Creates a datasource in the staging environment of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_data_agent_staging_datasource(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            type: str = Field(
                ...,
                description=(
                    "The datasource type. Additional DatasourceType types may be added over time. Allowed values: "
                    "FabricItem, LakehouseTables."
                ),
            ),
            item_reference: dict[str, Any] | None = Field(None, description="An item reference by ID object."),
            lakehouse_reference: dict[str, Any] | None = Field(None, description="An item reference by ID object."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources",
                body=_drop_none(
                    {"type": type, "itemReference": item_reference, "lakehouseReference": lakehouse_reference}
                ),
            )

        super().__init__(handler=_create_data_agent_staging_datasource, **kwargs)


class FabricDeleteDataAgentStagingDatasource(_FabricTool):
    name: str = "fabric_delete_data_agent_staging_datasource"
    description: str | None = (
        "Deletes a datasource from the staging environment of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_data_agent_staging_datasource(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}",
            )

        super().__init__(handler=_delete_data_agent_staging_datasource, **kwargs)


class FabricGetDataAgentStagingDatasource(_FabricTool):
    name: str = "fabric_get_data_agent_staging_datasource"
    description: str | None = (
        "Returns metadata for a specific staging datasource of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_agent_staging_datasource(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}",
            )

        super().__init__(handler=_get_data_agent_staging_datasource, **kwargs)


class FabricUpdateDataAgentStagingDatasource(_FabricTool):
    name: str = "fabric_update_data_agent_staging_datasource"
    description: str | None = (
        "Updates metadata for a specific staging datasource of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_data_agent_staging_datasource(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            instructions: str | None = Field(None, description="Custom AI instructions specific to this datasource."),
            description: str | None = Field(None, description="The description of the datasource."),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}",
                body=_drop_none({"instructions": instructions, "description": description}),
            )

        super().__init__(handler=_update_data_agent_staging_datasource, **kwargs)


class FabricDeleteDataAgentStagingDatasourceElement(_FabricTool):
    name: str = "fabric_delete_data_agent_staging_datasource_element"
    description: str | None = (
        "Deletes a schema element that is no longer present in the live schema from a staging datasource. Preview "
        "API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_data_agent_staging_datasource_element(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            id: str = Field(..., description="The element identifier."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/elements",
                params={"id": id},
            )

        super().__init__(handler=_delete_data_agent_staging_datasource_element, **kwargs)


class FabricListDataAgentStagingDatasourceElements(_FabricTool):
    name: str = "fabric_list_data_agent_staging_datasource_elements"
    description: str | None = (
        "Returns schema elements at a level of the schema tree for a staging datasource. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_agent_staging_datasource_elements(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            root_id: str | None = Field(
                None, description="The identifier of a parent element. Omit to get the root level."
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/elements",
                params={"rootId": root_id, "continuationToken": continuation_token},
            )

        super().__init__(handler=_list_data_agent_staging_datasource_elements, **kwargs)


class FabricUpdateDataAgentStagingDatasourceElement(_FabricTool):
    name: str = "fabric_update_data_agent_staging_datasource_element"
    description: str | None = "Updates a specific schema element for a staging datasource. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_data_agent_staging_datasource_element(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            id: str = Field(..., description="The element identifier."),
            is_selected: bool | None = Field(
                None,
                description=(
                    "Select or deselect the element. When deselected, the element configuration is preserved and restored "
                    "when re-selected."
                ),
            ),
            description: str | None = Field(None, description="Set a custom description for the element."),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/elements",
                params={"id": id},
                body=_drop_none({"isSelected": is_selected, "description": description}),
            )

        super().__init__(handler=_update_data_agent_staging_datasource_element, **kwargs)


class FabricListDataAgentStagingDatasourceFewshots(_FabricTool):
    name: str = "fabric_list_data_agent_staging_datasource_fewshots"
    description: str | None = (
        "Returns a list of staging fewshots for a datasource of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_agent_staging_datasource_fewshots(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_data_agent_staging_datasource_fewshots, **kwargs)


class FabricCreateDataAgentStagingDatasourceFewshot(_FabricTool):
    name: str = "fabric_create_data_agent_staging_datasource_fewshot"
    description: str | None = (
        "Creates a new fewshot for a staging datasource of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_data_agent_staging_datasource_fewshot(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            question: str = Field(..., description="Natural language question."),
            query: str | None = Field(None, description="The SQL/KQL query answer."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots",
                body=_drop_none({"question": question, "query": query}),
            )

        super().__init__(handler=_create_data_agent_staging_datasource_fewshot, **kwargs)


class FabricDeleteDataAgentStagingAllDatasourceFewshots(_FabricTool):
    name: str = "fabric_delete_data_agent_staging_all_datasource_fewshots"
    description: str | None = (
        "Deletes all fewshots for a staging datasource of the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_data_agent_staging_all_datasource_fewshots(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots/deleteAll",
            )

        super().__init__(handler=_delete_data_agent_staging_all_datasource_fewshots, **kwargs)


class FabricDeleteDataAgentStagingDatasourceFewshot(_FabricTool):
    name: str = "fabric_delete_data_agent_staging_datasource_fewshot"
    description: str | None = "Deletes a specific staging fewshot. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_data_agent_staging_datasource_fewshot(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            few_shot_id: str = Field(..., description="The fewshot ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots/{few_shot_id}",
            )

        super().__init__(handler=_delete_data_agent_staging_datasource_fewshot, **kwargs)


class FabricGetDataAgentStagingDatasourceFewshot(_FabricTool):
    name: str = "fabric_get_data_agent_staging_datasource_fewshot"
    description: str | None = "Returns a specific staging fewshot by ID. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_agent_staging_datasource_fewshot(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            few_shot_id: str = Field(..., description="The fewshot ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots/{few_shot_id}",
            )

        super().__init__(handler=_get_data_agent_staging_datasource_fewshot, **kwargs)


class FabricUpdateDataAgentStagingDatasourceFewshot(_FabricTool):
    name: str = "fabric_update_data_agent_staging_datasource_fewshot"
    description: str | None = "Updates a specific staging fewshot. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_data_agent_staging_datasource_fewshot(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            datasource_id: str = Field(..., description="The datasource ID."),
            few_shot_id: str = Field(..., description="The fewshot ID."),
            question: str | None = Field(None, description="Natural language question."),
            query: str | None = Field(None, description="The SQL/KQL query answer."),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/datasources/{datasource_id}/fewshots/{few_shot_id}",
                body=_drop_none({"question": question, "query": query}),
            )

        super().__init__(handler=_update_data_agent_staging_datasource_fewshot, **kwargs)


class FabricPublishDataAgent(_FabricTool):
    name: str = "fabric_publish_data_agent"
    description: str | None = "Publishes the staging configuration of the specified DataAgent. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _publish_data_agent(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            published_description: str | None = Field(
                None, description="A description shown when the agent appears in other experiences."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/publish",
                body=_drop_none({"publishedDescription": published_description}),
            )

        super().__init__(handler=_publish_data_agent, **kwargs)


class FabricResetDataAgentStagingChanges(_FabricTool):
    name: str = "fabric_reset_data_agent_staging_changes"
    description: str | None = (
        "Reverts the staging environment of the specified DataAgent to the published state. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _reset_data_agent_staging_changes(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/reset",
            )

        super().__init__(handler=_reset_data_agent_staging_changes, **kwargs)


class FabricGetDataAgentStagingSettings(_FabricTool):
    name: str = "fabric_get_data_agent_staging_settings"
    description: str | None = (
        "Returns the staging (draft) settings for the specified DataAgent. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_agent_staging_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/settings",
            )

        super().__init__(handler=_get_data_agent_staging_settings, **kwargs)


class FabricUpdateDataAgentStagingSettings(_FabricTool):
    name: str = "fabric_update_data_agent_staging_settings"
    description: str | None = "Updates the staging settings for the specified DataAgent. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_data_agent_staging_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_agent_id: str = Field(..., description="The DataAgent ID."),
            ai_instructions: str | None = Field(None, description="Custom AI system prompt for the DataAgent."),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/dataAgents/{data_agent_id}/staging/settings",
                body=_drop_none({"aiInstructions": ai_instructions}),
            )

        super().__init__(handler=_update_data_agent_staging_settings, **kwargs)


class FabricGetEventstreamDestination(_FabricTool):
    name: str = "fabric_get_eventstream_destination"
    description: str | None = "Returns the specified destination of the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_eventstream_destination(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            destination_id: str = Field(..., description="The destination ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/destinations/{destination_id}",
            )

        super().__init__(handler=_get_eventstream_destination, **kwargs)


class FabricGetEventstreamDestinationConnection(_FabricTool):
    name: str = "fabric_get_eventstream_destination_connection"
    description: str | None = "Returns the connection information of a specified destination of the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_eventstream_destination_connection(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            destination_id: str = Field(..., description="The destination ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/destinations/{destination_id}/connection",
            )

        super().__init__(handler=_get_eventstream_destination_connection, **kwargs)


class FabricPauseEventstreamDestination(_FabricTool):
    name: str = "fabric_pause_eventstream_destination"
    description: str | None = "Pause running the specified destination in the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _pause_eventstream_destination(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            destination_id: str = Field(..., description="The eventstream destination ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/destinations/{destination_id}/pause",
            )

        super().__init__(handler=_pause_eventstream_destination, **kwargs)


class FabricResumeEventstreamDestination(_FabricTool):
    name: str = "fabric_resume_eventstream_destination"
    description: str | None = "Resume running the specified destination in the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _resume_eventstream_destination(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            destination_id: str = Field(..., description="The eventstream destination ID."),
            start_type: str = Field(
                ...,
                description=(
                    "Represents the start type of the data source. Additional start types may be added over time. Allowed "
                    "values: Now, WhenLastStopped, CustomTime."
                ),
            ),
            custom_start_date_time: str | None = Field(
                None,
                description="The custom start time of the data source in UTC, using the YYYY-MM-DDTHH:mm:ssZ format.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/destinations/{destination_id}/resume",
                body=_drop_none({"startType": start_type, "customStartDateTime": custom_start_date_time}),
            )

        super().__init__(handler=_resume_eventstream_destination, **kwargs)


class FabricPauseEventstream(_FabricTool):
    name: str = "fabric_pause_eventstream"
    description: str | None = "Pause running all supported sources and destinations of the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _pause_eventstream(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/pause",
            )

        super().__init__(handler=_pause_eventstream, **kwargs)


class FabricResumeEventstream(_FabricTool):
    name: str = "fabric_resume_eventstream"
    description: str | None = "Resume running all supported sources and destinations of the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _resume_eventstream(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            start_type: str = Field(
                ...,
                description=(
                    "Represents the start type of the data source. Additional start types may be added over time. Allowed "
                    "values: Now, WhenLastStopped, CustomTime."
                ),
            ),
            custom_start_date_time: str | None = Field(
                None,
                description="The custom start time of the data source in UTC, using the YYYY-MM-DDTHH:mm:ssZ format.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/resume",
                body=_drop_none({"startType": start_type, "customStartDateTime": custom_start_date_time}),
            )

        super().__init__(handler=_resume_eventstream, **kwargs)


class FabricGetEventstreamSource(_FabricTool):
    name: str = "fabric_get_eventstream_source"
    description: str | None = "Returns the specified source of the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_eventstream_source(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            source_id: str = Field(..., description="The source ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/sources/{source_id}",
            )

        super().__init__(handler=_get_eventstream_source, **kwargs)


class FabricGetEventstreamSourceConnection(_FabricTool):
    name: str = "fabric_get_eventstream_source_connection"
    description: str | None = "Returns the connection information of specified source of the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_eventstream_source_connection(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            source_id: str = Field(..., description="The source ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/sources/{source_id}/connection",
            )

        super().__init__(handler=_get_eventstream_source_connection, **kwargs)


class FabricPauseEventstreamSource(_FabricTool):
    name: str = "fabric_pause_eventstream_source"
    description: str | None = "Pause running the specified source in the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _pause_eventstream_source(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            source_id: str = Field(..., description="The eventstream source ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/sources/{source_id}/pause",
            )

        super().__init__(handler=_pause_eventstream_source, **kwargs)


class FabricResumeEventstreamSource(_FabricTool):
    name: str = "fabric_resume_eventstream_source"
    description: str | None = "Resume running the specified source in the eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _resume_eventstream_source(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
            source_id: str = Field(..., description="The eventstream source ID."),
            start_type: str = Field(
                ...,
                description=(
                    "Represents the start type of the data source. Additional start types may be added over time. Allowed "
                    "values: Now, WhenLastStopped, CustomTime."
                ),
            ),
            custom_start_date_time: str | None = Field(
                None,
                description="The custom start time of the data source in UTC, using the YYYY-MM-DDTHH:mm:ssZ format.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/sources/{source_id}/resume",
                body=_drop_none({"startType": start_type, "customStartDateTime": custom_start_date_time}),
            )

        super().__init__(handler=_resume_eventstream_source, **kwargs)


class FabricGetEventstreamTopology(_FabricTool):
    name: str = "fabric_get_eventstream_topology"
    description: str | None = "Returns the topology of the specified eventstream."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_eventstream_topology(
            workspace_id: str = Field(..., description="The workspace ID."),
            eventstream_id: str = Field(..., description="The eventstream ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/eventstreams/{eventstream_id}/topology",
            )

        super().__init__(handler=_get_eventstream_topology, **kwargs)


class FabricListKQLDatabaseShortcuts(_FabricTool):
    name: str = "fabric_list_kql_database_shortcuts"
    description: str | None = "Returns a list of all table shortcuts under a database."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_kql_database_shortcuts(
            workspace_id: str = Field(..., description="The workspace ID."),
            kql_database_id: str = Field(..., description="The KQL database ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/kqlDatabases/{kql_database_id}/shortcuts",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_kql_database_shortcuts, **kwargs)


class FabricCreateKQLDatabaseShortcut(_FabricTool):
    name: str = "fabric_create_kql_database_shortcut"
    description: str | None = "Creates a new table shortcut under a database."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_kql_database_shortcut(
            workspace_id: str = Field(..., description="The ID of the workspace."),
            kql_database_id: str = Field(..., description="The KQL database ID."),
            name: str = Field(
                ...,
                description=(
                    'The table shortcut name. The table shortcut name cannot contain the following characters: ? : / \\ " '
                    "] [ + # %"
                ),
            ),
            enable_query_acceleration: bool = Field(
                ..., description="A boolean flag indicating whether the shortcut has query acceleration enabled."
            ),
            target: dict[str, Any] = Field(
                ...,
                description=(
                    "An object that contains the target datasource, and must specify exactly one of the supported "
                    "destinations as described in the table below."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/kqlDatabases/{kql_database_id}/shortcuts",
                body=_drop_none({"name": name, "enableQueryAcceleration": enable_query_acceleration, "target": target}),
            )

        super().__init__(handler=_create_kql_database_shortcut, **kwargs)


class FabricDeleteKQLDatabaseShortcut(_FabricTool):
    name: str = "fabric_delete_kql_database_shortcut"
    description: str | None = "Deletes a table shortcut under a database."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_kql_database_shortcut(
            workspace_id: str = Field(..., description="The workspace ID."),
            kql_database_id: str = Field(..., description="The KQL database ID."),
            shortcut_name: str = Field(..., description="The name of the shortcut"),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/kqlDatabases/{kql_database_id}/shortcuts/{_quote(shortcut_name)}",
            )

        super().__init__(handler=_delete_kql_database_shortcut, **kwargs)


class FabricGetKQLDatabaseShortcut(_FabricTool):
    name: str = "fabric_get_kql_database_shortcut"
    description: str | None = "Returns the properties of a table shortcut under a database."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_kql_database_shortcut(
            workspace_id: str = Field(..., description="The workspace ID."),
            kql_database_id: str = Field(..., description="The KQL database ID."),
            shortcut_name: str = Field(..., description="The name of the shortcut"),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/kqlDatabases/{kql_database_id}/shortcuts/{_quote(shortcut_name)}",
            )

        super().__init__(handler=_get_kql_database_shortcut, **kwargs)


class FabricScoreMLModelEndpoint(_FabricTool):
    name: str = "fabric_score_ml_model_endpoint"
    description: str | None = "Scores input data using the default version of the endpoint and returns results."

    def __init__(self, **kwargs: Any) -> None:
        async def _score_ml_model_endpoint(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
            inputs: list[Any] = Field(
                ...,
                description=(
                    "Machine learning inputs to score in the form of Pandas dataset arrays that can include strings, "
                    "numbers, integers and booleans."
                ),
            ),
            format_type: str | None = Field(
                None,
                description=(
                    "Format type of data. Additional Format types may be added over time. Allowed values: dataframe."
                ),
            ),
            orientation: str | None = Field(
                None,
                description=(
                    "Orientation of data. Additional Orientation types may be added over time. Allowed values: split, "
                    "values, record, index, table."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mlModels/{model_id}/endpoint/score",
                body=_drop_none({"formatType": format_type, "orientation": orientation, "inputs": inputs}),
            )

        super().__init__(handler=_score_ml_model_endpoint, **kwargs)


class FabricGetMLModelEndpoint(_FabricTool):
    name: str = "fabric_get_ml_model_endpoint"
    description: str | None = "Returns properties of the specified machine learning model's endpoint."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_ml_model_endpoint(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint",
            )

        super().__init__(handler=_get_ml_model_endpoint, **kwargs)


class FabricUpdateMLModelEndpoint(_FabricTool):
    name: str = "fabric_update_ml_model_endpoint"
    description: str | None = "Updates the default version of the specified model."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_ml_model_endpoint(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
            default_version_name: str | None = Field(
                None, description="Default machine learning model endpoint version name."
            ),
            default_version_assignment_behavior: str | None = Field(
                None,
                description=(
                    "The default version assignment behavior of a given machine learning model endpoint. Additional "
                    "EndpointDefaultVersionConfigurationPolicy types may be added over time. Allowed values: "
                    "StaticallyConfigured, NotConfigured."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint",
                body=_drop_none(
                    {
                        "defaultVersionName": default_version_name,
                        "defaultVersionAssignmentBehavior": default_version_assignment_behavior,
                    }
                ),
            )

        super().__init__(handler=_update_ml_model_endpoint, **kwargs)


class FabricListMLModelEndpointVersions(_FabricTool):
    name: str = "fabric_list_ml_model_endpoint_versions"
    description: str | None = "Returns a list of all machine learning model endpoint versions."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_ml_model_endpoint_versions(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_ml_model_endpoint_versions, **kwargs)


class FabricDeactivateAllMLModelEndpointVersions(_FabricTool):
    name: str = "fabric_deactivate_all_ml_model_endpoint_versions"
    description: str | None = "Deactivates the specified machine learning model and its version's endpoints."

    def __init__(self, **kwargs: Any) -> None:
        async def _deactivate_all_ml_model_endpoint_versions(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/deactivateAll",
            )

        super().__init__(handler=_deactivate_all_ml_model_endpoint_versions, **kwargs)


class FabricGetMLModelEndpointVersion(_FabricTool):
    name: str = "fabric_get_ml_model_endpoint_version"
    description: str | None = (
        "Returns information about the specified MLModel endpoint version. For machine learning model versions where "
        "endpoints have never been enabled, or have been disabled after enabling, the status would be 'deactivated'."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_ml_model_endpoint_version(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
            name: str = Field(..., description="The MLModel version name."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{_quote(name)}",
            )

        super().__init__(handler=_get_ml_model_endpoint_version, **kwargs)


class FabricUpdateMLModelEndpointVersion(_FabricTool):
    name: str = "fabric_update_ml_model_endpoint_version"
    description: str | None = "Update machine learning model endpoint version configuration."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_ml_model_endpoint_version(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
            name: str = Field(..., description="The MLModel version name."),
            scale_rule: str | None = Field(
                None,
                description=(
                    "Machine learning model endpoint scale rule. Additional ScaleRule types may be added over time. "
                    "Allowed values: AlwaysOn, AllowScaleToZero."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{_quote(name)}",
                body=_drop_none({"scaleRule": scale_rule}),
            )

        super().__init__(handler=_update_ml_model_endpoint_version, **kwargs)


class FabricActivateMLModelEndpointVersion(_FabricTool):
    name: str = "fabric_activate_ml_model_endpoint_version"
    description: str | None = "Activates the specified model version endpoint."

    def __init__(self, **kwargs: Any) -> None:
        async def _activate_ml_model_endpoint_version(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
            name: str = Field(..., description="The MLModel version name."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{_quote(name)}/activate",
            )

        super().__init__(handler=_activate_ml_model_endpoint_version, **kwargs)


class FabricDeactivateMLModelEndpointVersion(_FabricTool):
    name: str = "fabric_deactivate_ml_model_endpoint_version"
    description: str | None = "Deactivates the specified model version version."

    def __init__(self, **kwargs: Any) -> None:
        async def _deactivate_ml_model_endpoint_version(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
            name: str = Field(..., description="The MLModel version name."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{_quote(name)}/deactivate",
            )

        super().__init__(handler=_deactivate_ml_model_endpoint_version, **kwargs)


class FabricScoreMLModelEndpointVersion(_FabricTool):
    name: str = "fabric_score_ml_model_endpoint_version"
    description: str | None = "Scores input data for the specific version of the endpoint and returns results."

    def __init__(self, **kwargs: Any) -> None:
        async def _score_ml_model_endpoint_version(
            workspace_id: str = Field(..., description="The workspace ID."),
            model_id: str = Field(..., description="The machine learning model ID."),
            name: str = Field(..., description="The MLModel version name."),
            inputs: list[Any] = Field(
                ...,
                description=(
                    "Machine learning inputs to score in the form of Pandas dataset arrays that can include strings, "
                    "numbers, integers and booleans."
                ),
            ),
            format_type: str | None = Field(
                None,
                description=(
                    "Format type of data. Additional Format types may be added over time. Allowed values: dataframe."
                ),
            ),
            orientation: str | None = Field(
                None,
                description=(
                    "Orientation of data. Additional Orientation types may be added over time. Allowed values: split, "
                    "values, record, index, table."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mlmodels/{model_id}/endpoint/versions/{_quote(name)}/score",
                body=_drop_none({"formatType": format_type, "orientation": orientation, "inputs": inputs}),
            )

        super().__init__(handler=_score_ml_model_endpoint_version, **kwargs)


class FabricNlToKQL(_FabricTool):
    name: str = "fabric_nl_to_kql"
    description: str | None = "Returns a KQL query generated from natural language. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _nl_to_kql(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id_for_billing: str = Field(
                ...,
                description=(
                    "The ID of the item for the request. This can be a KQLQueryset, KQLDashboard, or Eventhouse item. "
                    "This item is used for billing the request."
                ),
            ),
            cluster_url: str = Field(..., description="The cluster URL for the request"),
            natural_language: str = Field(..., description="The natural language to generate the KQL query from."),
            database_name: str = Field(..., description="The name of the database for the request."),
            user_shots: list[Any] | None = Field(
                None,
                description=(
                    "The user shots for the request. This consists of user provided pairs of natural language and KQL "
                    "queries in order to help in generating the current requested KQL query."
                ),
            ),
            chat_messages: list[Any] | None = Field(
                None,
                description=(
                    "The chat messages for the request. The chat messages provide additional context for generating the "
                    "KQL query if necessary."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/realTimeIntelligence/nltokql",
                params={"beta": "true"},
                body=_drop_none(
                    {
                        "itemIdForBilling": item_id_for_billing,
                        "clusterUrl": cluster_url,
                        "naturalLanguage": natural_language,
                        "databaseName": database_name,
                        "userShots": user_shots,
                        "chatMessages": chat_messages,
                    }
                ),
            )

        super().__init__(handler=_nl_to_kql, **kwargs)


class FabricBindSemanticModelConnection(_FabricTool):
    name: str = "fabric_bind_semantic_model_connection"
    description: str | None = (
        "Binds a semantic model data source reference to a data connection. This API can also be used to unbind data "
        "source references."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _bind_semantic_model_connection(
            workspace_id: str = Field(..., description="The workspace ID."),
            semantic_model_id: str = Field(..., description="The semantic model ID."),
            connection_binding: dict[str, Any] = Field(..., description="The details of the connection binding."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/semanticModels/{semantic_model_id}/bindConnection",
                body=_drop_none({"connectionBinding": connection_binding}),
            )

        super().__init__(handler=_bind_semantic_model_connection, **kwargs)


class FabricListSQLDatabaseRestorableDeletedDatabases(_FabricTool):
    name: str = "fabric_list_sql_database_restorable_deleted_databases"
    description: str | None = "Gets the list of restorable deleted databases in the workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_sql_database_restorable_deleted_databases(
            workspace_id: str = Field(..., description="The workspace identifier."),
            recursive: bool | None = Field(
                None,
                description=(
                    "Lists items in a folder and its nested folders, or just a folder only. True - All items in the "
                    "folder and its nested folders are listed, False - Only items in the folder are listed. The default "
                    "value is true."
                ),
            ),
            root_folder_id: str | None = Field(
                None,
                description=(
                    "This parameter allows users to filter items based on a specific root folder. If not provided, the "
                    "workspace is used as the root folder."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/sqlDatabases/restorableDeletedDatabases",
                params={"recursive": recursive, "rootFolderId": root_folder_id},
            )

        super().__init__(handler=_list_sql_database_restorable_deleted_databases, **kwargs)


class FabricRevalidateSQLDatabaseCMK(_FabricTool):
    name: str = "fabric_revalidate_sql_database_cmk"
    description: str | None = (
        "Revalidate Customer‑Managed Key (CMK) of the specified SQL database. Revalidating Customer‑Managed Key (CMK) "
        "of the specified SQL database, which checks that the currently configured Azure Key Vault (AKV) key is still "
        "accessible, valid, and authorized for encryption operations."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _revalidate_sql_database_cmk(
            workspace_id: str = Field(..., description="The workspace identifier."),
            sql_database_id: str = Field(..., description="The SQL database ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/revalidateCMK",
            )

        super().__init__(handler=_revalidate_sql_database_cmk, **kwargs)


class FabricGetSQLDatabaseAuditSettings(_FabricTool):
    name: str = "fabric_get_sql_database_audit_settings"
    description: str | None = "Gets the auditing settings on the specified SQL database."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sql_database_audit_settings(
            workspace_id: str = Field(..., description="The workspace identifier."),
            sql_database_id: str = Field(..., description="The SQL database ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/settings/sqlAudit",
            )

        super().__init__(handler=_get_sql_database_audit_settings, **kwargs)


class FabricUpdateSQLDatabaseAuditSettings(_FabricTool):
    name: str = "fabric_update_sql_database_audit_settings"
    description: str | None = "Updates the auditing settings on the specified SQL database."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_sql_database_audit_settings(
            workspace_id: str = Field(..., description="The workspace identifier."),
            sql_database_id: str = Field(..., description="The SQL database ID."),
            state: str | None = Field(
                None,
                description=(
                    "Sql Audit settings state. When enabling the audit policy for the first time after database creation "
                    "(by setting state to 'Enabled' without other properties), default values are applied. For all "
                    "subsequent enable/disable operations, the previous policy settings are preserved. Additional "
                    "SqlAuditSettingsState may be added over time. Allowed values: Enabled, Disabled."
                ),
            ),
            retention_days: int | None = Field(
                None,
                description=(
                    "Retention days. For the first time, when state is set to Enabled and this property is not provided, "
                    "retentionDays will be set to 0 (indefinite retention period) by default."
                ),
            ),
            audit_actions_and_groups: list[Any] | None = Field(
                None,
                description=(
                    "Audit actions and groups. For the first time, when state is set to Enabled and this property is not "
                    "provided, default audit actions and groups will be applied."
                ),
            ),
            predicate_expression: str | None = Field(
                None,
                description=(
                    "The predicate expression used to filter audit logs. For the first time, when state is set to Enabled "
                    "and this property is not provided, no predicate expression will be applied by default."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/settings/sqlAudit",
                body=_drop_none(
                    {
                        "state": state,
                        "retentionDays": retention_days,
                        "auditActionsAndGroups": audit_actions_and_groups,
                        "predicateExpression": predicate_expression,
                    }
                ),
            )

        super().__init__(handler=_update_sql_database_audit_settings, **kwargs)


class FabricStartSQLDatabaseMirroring(_FabricTool):
    name: str = "fabric_start_sql_database_mirroring"
    description: str | None = "Starts the mirroring of the specified SQL database."

    def __init__(self, **kwargs: Any) -> None:
        async def _start_sql_database_mirroring(
            workspace_id: str = Field(..., description="The workspace identifier."),
            sql_database_id: str = Field(..., description="The SQL database ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/startMirroring",
            )

        super().__init__(handler=_start_sql_database_mirroring, **kwargs)


class FabricStopSQLDatabaseMirroring(_FabricTool):
    name: str = "fabric_stop_sql_database_mirroring"
    description: str | None = "Stops the mirroring of the specified SQL database."

    def __init__(self, **kwargs: Any) -> None:
        async def _stop_sql_database_mirroring(
            workspace_id: str = Field(..., description="The workspace identifier."),
            sql_database_id: str = Field(..., description="The SQL database ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/sqlDatabases/{sql_database_id}/stopMirroring",
            )

        super().__init__(handler=_stop_sql_database_mirroring, **kwargs)


class FabricGetSQLEndpointAuditSettings(_FabricTool):
    name: str = "fabric_get_sql_endpoint_audit_settings"
    description: str | None = "Returns the settings associated with the SQL endpoint."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sql_endpoint_audit_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/sqlEndpoints/{item_id}/settings/sqlAudit",
            )

        super().__init__(handler=_get_sql_endpoint_audit_settings, **kwargs)


class FabricUpdateSQLEndpointAuditSettings(_FabricTool):
    name: str = "fabric_update_sql_endpoint_audit_settings"
    description: str | None = "Update settings associated with the SQL endpoint."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_sql_endpoint_audit_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            state: str | None = Field(
                None,
                description=(
                    "Audit settings state. Additional AuditSettingsState may be added over time. Allowed values: Enabled, "
                    "Disabled."
                ),
            ),
            retention_days: int | None = Field(None, description="Retention days."),
            predicate_expression: str | None = Field(
                None,
                description=(
                    "The predicate expression that determines whether an audit event is processed. Specify only the "
                    "`<predicate_expression>` value; don't include the `WHERE` keyword. The expression uses the following "
                    "syntax from CREATE SERVER AUDIT (Transact-SQL): ```syntaxsql <predicate_expression> ::= { [ NOT ] "
                    "<predicate_factor> [ { AND | OR } [ NOT ] { <predicate_factor> } ] [ ,... n ] } <predicate_factor> "
                    "::= event_field_name { = | < > | != | > | >= | < | <= | LIKE } { number | 'string' } ``` Event field "
                    "names correspond to the audit record columns documented in sys.fn_get_audit_file (Transact-SQL). All "
                    "documented fields can be used except `file_name`, `audit_file_offset`, and `event_time`. Although "
                    "`action_id` and `class_type` are returned as strings, they can only be compared with numeric values "
                    "in a predicate. String comparisons don't perform implicit type conversion. The maximum expression "
                    "length is 3,000 characters. When SQL auditing is enabled for the first time, omitting this property "
                    "applies no predicate. On subsequent updates, omitting this property leaves the existing predicate "
                    "unchanged. Specify an empty string to remove the existing predicate."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/sqlEndpoints/{item_id}/settings/sqlAudit",
                body=_drop_none(
                    {"state": state, "retentionDays": retention_days, "predicateExpression": predicate_expression}
                ),
            )

        super().__init__(handler=_update_sql_endpoint_audit_settings, **kwargs)


class FabricSetSQLEndpointAuditActionsAndGroups(_FabricTool):
    name: str = "fabric_set_sql_endpoint_audit_actions_and_groups"
    description: str | None = "Update the audit actions and groups for this SQL endpoint."

    def __init__(self, **kwargs: Any) -> None:
        async def _set_sql_endpoint_audit_actions_and_groups(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            audit_actions_and_groups: list[str] = Field(
                ..., description="Audit action groups and actions to enable, e.g. BATCH_COMPLETED_GROUP."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/sqlEndpoints/{item_id}/settings/sqlAudit/setAuditActionsAndGroups",
                body=audit_actions_and_groups,
            )

        super().__init__(handler=_set_sql_endpoint_audit_actions_and_groups, **kwargs)


class FabricGetSQLEndpointConnectionString(_FabricTool):
    name: str = "fabric_get_sql_endpoint_connection_string"
    description: str | None = "Returns the SQL connection string of the specified warehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_sql_endpoint_connection_string(
            workspace_id: str = Field(..., description="The workspace ID."),
            sql_endpoint_id: str = Field(..., description="The SQL endpoint ID."),
            guest_tenant_id: str | None = Field(
                None,
                description="The guest tenant ID if the end user's tenant is different from the SQL endpoint's tenant.",
            ),
            private_link_type: str | None = Field(
                None,
                description=(
                    "Indicates the type of private link this connection string uses. Additional `privateLinkType` types "
                    "may be added over time. Allowed values: None, Workspace."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/sqlEndpoints/{sql_endpoint_id}/connectionString",
                params={"guestTenantId": guest_tenant_id, "privateLinkType": private_link_type},
            )

        super().__init__(handler=_get_sql_endpoint_connection_string, **kwargs)


class FabricRefreshSQLEndpointMetadata(_FabricTool):
    name: str = "fabric_refresh_sql_endpoint_metadata"
    description: str | None = (
        "Refreshes tables within a SQL analytics endpoint. When `tables` is provided in the request body, only the "
        "specified tables are refreshed. When omitted or empty, all tables are refreshed."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _refresh_sql_endpoint_metadata(
            workspace_id: str = Field(..., description="The workspace ID."),
            sql_endpoint_id: str = Field(..., description="The SQL analytics endpoint ID."),
            timeout: dict[str, Any] | None = Field(None, description="A duration."),
            recreate_tables: bool | None = Field(
                None,
                description=(
                    "When set to true, this property instructs the system to drop and recreate all tables on the SQL "
                    "analytics endpoint during the refresh process. Use this option if you need to fully rebuild tables "
                    "from their source definitions, for example, to resolve inconsistencies or ensure a clean refresh. "
                    "When combined with `tables`, the sync state reset is scoped only to the specified tables. The "
                    "default value is false."
                ),
            ),
            tables: list[Any] | None = Field(
                None,
                description=(
                    "When provided, scopes the refresh to only the listed tables. When omitted or empty, all tables are "
                    "refreshed. Each entry specifies a schema and one or more table names to refresh under that schema. "
                    "The maximum number of tables that can be synchronized in a single request is 25. Table resolution "
                    "depends on whether the SQL endpoint's parent item is schema-enabled. For schema-enabled items, "
                    "tables are resolved using the caller-provided schema. For non-schema-enabled items, all tables "
                    "resolve under the default schema regardless of the caller-provided schema value; tables under a "
                    "non-default schema cannot be resolved and will be reported with a `DeltaTableNotFound` error."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/sqlEndpoints/{sql_endpoint_id}/refreshMetadata",
                body=_drop_none({"timeout": timeout, "recreateTables": recreate_tables, "tables": tables}),
            )

        super().__init__(handler=_refresh_sql_endpoint_metadata, **kwargs)


class FabricGetWarehouseSQLPoolsConfiguration(_FabricTool):
    name: str = "fabric_get_warehouse_sql_pools_configuration"
    description: str | None = (
        "Gets the SQL Pools configuration in the specified workspace (beta). Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_warehouse_sql_pools_configuration(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/warehouses/sqlPoolsConfiguration",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_warehouse_sql_pools_configuration, **kwargs)


class FabricUpdateWarehouseSQLPoolsConfiguration(_FabricTool):
    name: str = "fabric_update_warehouse_sql_pools_configuration"
    description: str | None = (
        "Updates the SQL Pools configuration in the specified workspace (beta). Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_warehouse_sql_pools_configuration(
            workspace_id: str = Field(..., description="The workspace ID."),
            custom_sql_pools_enabled: bool | None = Field(
                None,
                description=(
                    "Indicates whether the SQL pools configuration is enabled. When set to false, the configuration is "
                    "disabled but preserved. Re-enabling it restores the previously saved configuration."
                ),
            ),
            custom_sql_pools: list[Any] | None = Field(
                None,
                description=(
                    "A list of SQL pool elements. In update requests, you must explicitly specify the name of the pools "
                    "to retain. Any SQL pool not included in the update request will be deleted."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/warehouses/sqlPoolsConfiguration",
                params={"beta": "true"},
                body=_drop_none(
                    {"customSQLPoolsEnabled": custom_sql_pools_enabled, "customSQLPools": custom_sql_pools}
                ),
            )

        super().__init__(handler=_update_warehouse_sql_pools_configuration, **kwargs)


class FabricGetWarehouseAuditSettings(_FabricTool):
    name: str = "fabric_get_warehouse_audit_settings"
    description: str | None = "Returns the settings associated with the warehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_warehouse_audit_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/warehouses/{item_id}/settings/sqlAudit",
            )

        super().__init__(handler=_get_warehouse_audit_settings, **kwargs)


class FabricUpdateWarehouseAuditSettings(_FabricTool):
    name: str = "fabric_update_warehouse_audit_settings"
    description: str | None = "Update settings associated with the warehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_warehouse_audit_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            state: str | None = Field(
                None,
                description=(
                    "Audit settings state. Additional AuditSettingsState may be added over time. Allowed values: Enabled, "
                    "Disabled."
                ),
            ),
            retention_days: int | None = Field(None, description="Retention days."),
            predicate_expression: str | None = Field(
                None,
                description=(
                    "The predicate expression that determines whether an audit event is processed. Specify only the "
                    "`<predicate_expression>` value; don't include the `WHERE` keyword. The expression uses the following "
                    "syntax from CREATE SERVER AUDIT (Transact-SQL): ```syntaxsql <predicate_expression> ::= { [ NOT ] "
                    "<predicate_factor> [ { AND | OR } [ NOT ] { <predicate_factor> } ] [ ,... n ] } <predicate_factor> "
                    "::= event_field_name { = | < > | != | > | >= | < | <= | LIKE } { number | 'string' } ``` Event field "
                    "names correspond to the audit record columns documented in sys.fn_get_audit_file (Transact-SQL). All "
                    "documented fields can be used except `file_name`, `audit_file_offset`, and `event_time`. Although "
                    "`action_id` and `class_type` are returned as strings, they can only be compared with numeric values "
                    "in a predicate. String comparisons don't perform implicit type conversion. The maximum expression "
                    "length is 3,000 characters. When SQL auditing is enabled for the first time, omitting this property "
                    "applies no predicate. On subsequent updates, omitting this property leaves the existing predicate "
                    "unchanged. Specify an empty string to remove the existing predicate."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/warehouses/{item_id}/settings/sqlAudit",
                body=_drop_none(
                    {"state": state, "retentionDays": retention_days, "predicateExpression": predicate_expression}
                ),
            )

        super().__init__(handler=_update_warehouse_audit_settings, **kwargs)


class FabricSetWarehouseAuditActionsAndGroups(_FabricTool):
    name: str = "fabric_set_warehouse_audit_actions_and_groups"
    description: str | None = "Update the audit actions and groups for this warehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _set_warehouse_audit_actions_and_groups(
            workspace_id: str = Field(..., description="The workspace ID."),
            item_id: str = Field(..., description="The item ID."),
            audit_actions_and_groups: list[str] = Field(
                ..., description="Audit action groups and actions to enable, e.g. BATCH_COMPLETED_GROUP."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/warehouses/{item_id}/settings/sqlAudit/setAuditActionsAndGroups",
                body=audit_actions_and_groups,
            )

        super().__init__(handler=_set_warehouse_audit_actions_and_groups, **kwargs)


class FabricGetWarehouseConnectionString(_FabricTool):
    name: str = "fabric_get_warehouse_connection_string"
    description: str | None = "Returns the SQL connection string of the specified warehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_warehouse_connection_string(
            workspace_id: str = Field(..., description="The workspace ID."),
            warehouse_id: str = Field(..., description="The warehouse ID."),
            guest_tenant_id: str | None = Field(
                None, description="The guest tenant ID if the end user's tenant is different from the warehouse tenant."
            ),
            private_link_type: str | None = Field(
                None,
                description=(
                    "Indicates the type of private link this connection string uses. Additional `privateLinkType` types "
                    "may be added over time. Allowed values: None, Workspace."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/warehouses/{warehouse_id}/connectionString",
                params={"guestTenantId": guest_tenant_id, "privateLinkType": private_link_type},
            )

        super().__init__(handler=_get_warehouse_connection_string, **kwargs)


class FabricListWarehouseRestorePoints(_FabricTool):
    name: str = "fabric_list_warehouse_restore_points"
    description: str | None = "Returns all restore points for a warehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_warehouse_restore_points(
            workspace_id: str = Field(..., description="The workspace ID."),
            warehouse_id: str = Field(..., description="The warehouse ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_warehouse_restore_points, **kwargs)


class FabricCreateWarehouseRestorePoint(_FabricTool):
    name: str = "fabric_create_warehouse_restore_point"
    description: str | None = "Creates a restore point for a warehouse at the current timestamp."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_warehouse_restore_point(
            workspace_id: str = Field(..., description="The workspace ID."),
            warehouse_id: str = Field(..., description="The warehouse ID."),
            display_name: str | None = Field(
                None, description="The restore point name. Maximum length is 128 characters."
            ),
            description: str | None = Field(
                None, description="The restore point description. Maximum length is 512 characters."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints",
                body=_drop_none({"displayName": display_name, "description": description}),
            )

        super().__init__(handler=_create_warehouse_restore_point, **kwargs)


class FabricDeleteWarehouseRestorePoint(_FabricTool):
    name: str = "fabric_delete_warehouse_restore_point"
    description: str | None = "Deletes a restore point specified for a warehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_warehouse_restore_point(
            workspace_id: str = Field(..., description="The workspace ID."),
            warehouse_id: str = Field(..., description="The warehouse ID."),
            restore_point_id: str = Field(..., description="The restore point ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints/{_quote(restore_point_id)}",
            )

        super().__init__(handler=_delete_warehouse_restore_point, **kwargs)


class FabricGetWarehouseRestorePoint(_FabricTool):
    name: str = "fabric_get_warehouse_restore_point"
    description: str | None = "Returns the properties of a restore point specified for a warehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_warehouse_restore_point(
            workspace_id: str = Field(..., description="The workspace ID."),
            warehouse_id: str = Field(..., description="The warehouse ID."),
            restore_point_id: str = Field(..., description="The restore point ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints/{_quote(restore_point_id)}",
            )

        super().__init__(handler=_get_warehouse_restore_point, **kwargs)


class FabricUpdateWarehouseRestorePoint(_FabricTool):
    name: str = "fabric_update_warehouse_restore_point"
    description: str | None = "Updates an existing restore point by renaming the name or description of it."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_warehouse_restore_point(
            workspace_id: str = Field(..., description="The workspace ID."),
            warehouse_id: str = Field(..., description="The warehouse ID."),
            restore_point_id: str = Field(..., description="The restore point ID."),
            display_name: str | None = Field(
                None, description="The restore point name. Maximum length is 128 characters."
            ),
            description: str | None = Field(
                None, description="The restore point description. Maximum length is 512 characters."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints/{_quote(restore_point_id)}",
                body=_drop_none({"displayName": display_name, "description": description}),
            )

        super().__init__(handler=_update_warehouse_restore_point, **kwargs)


class FabricRestoreWarehouseToRestorePoint(_FabricTool):
    name: str = "fabric_restore_warehouse_to_restore_point"
    description: str | None = "Restores a warehouse in-place to the restore point specified."

    def __init__(self, **kwargs: Any) -> None:
        async def _restore_warehouse_to_restore_point(
            workspace_id: str = Field(..., description="The workspace ID."),
            warehouse_id: str = Field(..., description="The warehouse ID."),
            restore_point_id: str = Field(..., description="The restore point ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/warehouses/{warehouse_id}/restorePoints/{_quote(restore_point_id)}/restore",
            )

        super().__init__(handler=_restore_warehouse_to_restore_point, **kwargs)
