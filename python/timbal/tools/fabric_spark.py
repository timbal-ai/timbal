"""Microsoft Fabric REST API tools: lakehouses, notebooks, Spark, Spark job definitions and environments.

Auth, long running operations and request handling live in ``fabric.py``.
"""

from typing import Any

from pydantic import Field

from .fabric import _drop_none, _fabric_request, _FabricTool, _quote, _upload_bytes


class FabricListEnvironmentPublishedLibraries(_FabricTool):
    name: str = "fabric_list_environment_published_libraries"
    description: str | None = "Get environment published libraries. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_environment_published_libraries(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
            continuation_token: str | None = Field(
                None, description="Token to retrieve the next page of results, if available."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/environments/{environment_id}/libraries",
                params={"continuationToken": continuation_token, "beta": "false"},
            )

        super().__init__(handler=_list_environment_published_libraries, **kwargs)


class FabricExportEnvironmentPublishedExternalLibraries(_FabricTool):
    name: str = "fabric_export_environment_published_external_libraries"
    description: str | None = "Export a set of external libraries published in the environment in `YML` format."

    def __init__(self, **kwargs: Any) -> None:
        async def _export_environment_published_external_libraries(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/environments/{environment_id}/libraries/exportExternalLibraries",
            )

        super().__init__(handler=_export_environment_published_external_libraries, **kwargs)


class FabricGetEnvironmentPublishedSparkCompute(_FabricTool):
    name: str = "fabric_get_environment_published_spark_compute"
    description: str | None = "Get environment spark compute. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_environment_published_spark_compute(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/environments/{environment_id}/sparkcompute",
                params={"beta": "false"},
            )

        super().__init__(handler=_get_environment_published_spark_compute, **kwargs)


class FabricCancelEnvironmentPublish(_FabricTool):
    name: str = "fabric_cancel_environment_publish"
    description: str | None = "Trigger an environment publish cancellation."

    def __init__(self, **kwargs: Any) -> None:
        async def _cancel_environment_publish(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/cancelPublish",
            )

        super().__init__(handler=_cancel_environment_publish, **kwargs)


class FabricListEnvironmentStagingLibraries(_FabricTool):
    name: str = "fabric_list_environment_staging_libraries"
    description: str | None = "Get a list of libraries staged into environment. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_environment_staging_libraries(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
            continuation_token: str | None = Field(
                None, description="Token to retrieve the next page of results, if available."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries",
                params={"continuationToken": continuation_token, "beta": "false"},
            )

        super().__init__(handler=_list_environment_staging_libraries, **kwargs)


class FabricExportEnvironmentStagingExternalLibraries(_FabricTool):
    name: str = "fabric_export_environment_staging_external_libraries"
    description: str | None = "Export a set of external libraries saved in the environment in YML format."

    def __init__(self, **kwargs: Any) -> None:
        async def _export_environment_staging_external_libraries(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/exportExternalLibraries",
            )

        super().__init__(handler=_export_environment_staging_external_libraries, **kwargs)


class FabricImportEnvironmentStagingExternalLibraries(_FabricTool):
    name: str = "fabric_import_environment_staging_external_libraries"
    description: str | None = (
        "Upload spark external libraries as an `environment.yml` file into environment. It overrides the list of "
        "existing external libraries in environment. This API allows file upload at a time."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _import_environment_staging_external_libraries(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
            content: str | None = Field(None, description="UTF-8 text content. Provide this or content_base64."),
            content_base64: str | None = Field(
                None, description="Base64-encoded binary content. Provide this or content."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/importExternalLibraries",
                content=_upload_bytes(content, content_base64),
                content_type="application/octet-stream",
            )

        super().__init__(handler=_import_environment_staging_external_libraries, **kwargs)


class FabricRemoveEnvironmentStagingExternalLibrary(_FabricTool):
    name: str = "fabric_remove_environment_staging_external_library"
    description: str | None = (
        "Delete a spark external library from an environment. This API allows one library deletion at a time."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _remove_environment_staging_external_library(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
            name: str = Field(..., description="The name of library."),
            version: str = Field(..., description="The version of external library."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/removeExternalLibrary",
                body=_drop_none({"name": name, "version": version}),
            )

        super().__init__(handler=_remove_environment_staging_external_library, **kwargs)


class FabricDeleteEnvironmentStagingCustomLibrary(_FabricTool):
    name: str = "fabric_delete_environment_staging_custom_library"
    description: str | None = (
        "Delete a custom library from environment. It supports deleting one file at a time. The supported file "
        "formats are .jar, .py, .whl, and .tar.gz."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_environment_staging_custom_library(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
            library_name: str = Field(
                ...,
                description=(
                    "The library name to be deleted. The library name needs to include its extension if it is a custom "
                    "library, for example `samplefile.jar`."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/{_quote(library_name)}",
            )

        super().__init__(handler=_delete_environment_staging_custom_library, **kwargs)


class FabricUploadEnvironmentStagingCustomLibrary(_FabricTool):
    name: str = "fabric_upload_environment_staging_custom_library"
    description: str | None = (
        "Upload spark library into environment. This API allows one file upload at a time. The supported file formats "
        "are .jar, .py, .whl, and .tar.gz."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _upload_environment_staging_custom_library(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
            library_name: str = Field(
                ...,
                description=(
                    "The library name to be uploaded. The library name needs to include its extension, for example "
                    "`samplefile.jar`."
                ),
            ),
            content: str | None = Field(None, description="UTF-8 text content. Provide this or content_base64."),
            content_base64: str | None = Field(
                None, description="Base64-encoded binary content. Provide this or content."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/libraries/{_quote(library_name)}",
                content=_upload_bytes(content, content_base64),
                content_type="application/octet-stream",
            )

        super().__init__(handler=_upload_environment_staging_custom_library, **kwargs)


class FabricPublishEnvironment(_FabricTool):
    name: str = "fabric_publish_environment"
    description: str | None = "Trigger an environment publish operation. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _publish_environment(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/publish",
                params={"beta": "false"},
            )

        super().__init__(handler=_publish_environment, **kwargs)


class FabricGetEnvironmentStagingSparkCompute(_FabricTool):
    name: str = "fabric_get_environment_staging_spark_compute"
    description: str | None = "Get environment staging spark compute. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_environment_staging_spark_compute(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/sparkcompute",
                params={"beta": "false"},
            )

        super().__init__(handler=_get_environment_staging_spark_compute, **kwargs)


class FabricUpdateEnvironmentStagingSparkCompute(_FabricTool):
    name: str = "fabric_update_environment_staging_spark_compute"
    description: str | None = (
        "Update environment staging spark compute. If you want to delete a spark property, set its value as null. "
        "Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_environment_staging_spark_compute(
            workspace_id: str = Field(..., description="The workspace ID."),
            environment_id: str = Field(..., description="The environment ID."),
            instance_pool: dict[str, Any] | None = Field(None, description="The instance pool."),
            driver_cores: int | None = Field(
                None, description="Spark driver core. Must be one of the following values: 4, 8, 16, 32, 64."
            ),
            driver_memory: str | None = Field(
                None,
                description=(
                    "Custom pool memory for Spark driver or Spark executor. Additional `CustomPoolMemory` types may be "
                    "added over time. Allowed values: 28g, 56g, 112g, 224g, 400g."
                ),
            ),
            executor_cores: int | None = Field(
                None, description="Spark executor core. Must be one of the following values: 4, 8, 16, 32, 64."
            ),
            executor_memory: str | None = Field(
                None,
                description=(
                    "Custom pool memory for Spark driver or Spark executor. Additional `CustomPoolMemory` types may be "
                    "added over time. Allowed values: 28g, 56g, 112g, 224g, 400g."
                ),
            ),
            dynamic_executor_allocation: dict[str, Any] | None = Field(
                None, description="Dynamic executor allocation proerties."
            ),
            spark_properties: list[Any] | None = Field(None, description="Spark properties."),
            runtime_version: str | None = Field(
                None, description="Runtime version, find the supported fabric runtimes. For example: 1.3"
            ),
            custom_live_pool_support: str | None = Field(
                None,
                description=(
                    "Flag controlling whether live pool support is active. Additional `CustomLivePoolSupport` values may "
                    "be added over time. Allowed values: Enabled, Disabled."
                ),
            ),
            custom_live_pool_settings: dict[str, Any] | None = Field(
                None, description="Live pool hydration settings on the environment's spark compute."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/environments/{environment_id}/staging/sparkcompute",
                params={"beta": "false"},
                body=_drop_none(
                    {
                        "instancePool": instance_pool,
                        "driverCores": driver_cores,
                        "driverMemory": driver_memory,
                        "executorCores": executor_cores,
                        "executorMemory": executor_memory,
                        "dynamicExecutorAllocation": dynamic_executor_allocation,
                        "sparkProperties": spark_properties,
                        "runtimeVersion": runtime_version,
                        "customLivePoolSupport": custom_live_pool_support,
                        "customLivePoolSettings": custom_live_pool_settings,
                    }
                ),
            )

        super().__init__(handler=_update_environment_staging_spark_compute, **kwargs)


class FabricRunLakehouseRefreshMaterializedLakeViews(_FabricTool):
    name: str = "fabric_run_lakehouse_refresh_materialized_lake_views"
    description: str | None = (
        "Run on-demand [Refresh "
        "MaterializedLakeViews](/fabric/data-engineering/materialized-lake-views/refresh-materialized-lake-view) job "
        "instance. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _run_lakehouse_refresh_materialized_lake_views(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/refreshMaterializedLakeViews/instances",
            )

        super().__init__(handler=_run_lakehouse_refresh_materialized_lake_views, **kwargs)


class FabricCreateLakehouseRefreshMaterializedLakeViewsSchedule(_FabricTool):
    name: str = "fabric_create_lakehouse_refresh_materialized_lake_views_schedule"
    description: str | None = (
        "Create a new Refresh MaterializedLakeViews schedule for a lakehouse. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_lakehouse_refresh_materialized_lake_views_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
            execution_data: dict[str, Any] | None = Field(
                None, description="The execution data for the refresh materialized lake views."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/refreshMaterializedLakeViews/schedules",
                body=_drop_none({"enabled": enabled, "configuration": configuration, "executionData": execution_data}),
            )

        super().__init__(handler=_create_lakehouse_refresh_materialized_lake_views_schedule, **kwargs)


class FabricDeleteLakehouseRefreshMaterializedLakeViewsSchedule(_FabricTool):
    name: str = "fabric_delete_lakehouse_refresh_materialized_lake_views_schedule"
    description: str | None = (
        "Delete an existing Refresh MaterializedLakeViews schedule for a lakehouse. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_lakehouse_refresh_materialized_lake_views_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            schedule_id: str = Field(..., description="The lakehouse schedule ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/refreshMaterializedLakeViews/schedules/{schedule_id}",
            )

        super().__init__(handler=_delete_lakehouse_refresh_materialized_lake_views_schedule, **kwargs)


class FabricUpdateLakehouseRefreshMaterializedLakeViewsSchedule(_FabricTool):
    name: str = "fabric_update_lakehouse_refresh_materialized_lake_views_schedule"
    description: str | None = (
        "Update an existing Refresh MaterializedLakeViews schedule for a lakehouse. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_lakehouse_refresh_materialized_lake_views_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            schedule_id: str = Field(..., description="The lakehouse schedule ID."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
            execution_data: dict[str, Any] | None = Field(
                None, description="The execution data for the refresh materialized lake views."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/refreshMaterializedLakeViews/schedules/{schedule_id}",
                body=_drop_none({"enabled": enabled, "configuration": configuration, "executionData": execution_data}),
            )

        super().__init__(handler=_update_lakehouse_refresh_materialized_lake_views_schedule, **kwargs)


class FabricRunLakehouseTableMaintenance(_FabricTool):
    name: str = "fabric_run_lakehouse_table_maintenance"
    description: str | None = (
        "Run on-demand [table maintenance](/fabric/data-engineering/lakehouse-table-maintenance) job instance. "
        "Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _run_lakehouse_table_maintenance(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The Lakehouse item ID."),
            execution_data: dict[str, Any] = Field(
                ..., description="Run on demand lakehouse table maintenance instance payload"
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/jobs/tableMaintenance/instances",
                body=_drop_none({"executionData": execution_data}),
            )

        super().__init__(handler=_run_lakehouse_table_maintenance, **kwargs)


class FabricListLakehouseLivySessions(_FabricTool):
    name: str = "fabric_list_lakehouse_livy_sessions"
    description: str | None = "Returns a list of livy sessions from the specified item identifier."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_lakehouse_livy_sessions(
            workspace_id: str = Field(..., description="The workspace identifier."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            continuation_token: str | None = Field(
                None, description="Token to retrieve the next page of results, if available."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/livySessions",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_lakehouse_livy_sessions, **kwargs)


class FabricGetLakehouseLivySession(_FabricTool):
    name: str = "fabric_get_lakehouse_livy_session"
    description: str | None = "Returns properties of the specified livy session."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_lakehouse_livy_session(
            workspace_id: str = Field(..., description="The workspace identifier."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            livy_id: str = Field(..., description="The session identifier."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/livySessions/{_quote(livy_id)}",
            )

        super().__init__(handler=_get_lakehouse_livy_session, **kwargs)


class FabricListLakehouseMLVExecutionDefinitions(_FabricTool):
    name: str = "fabric_list_lakehouse_mlv_execution_definitions"
    description: str | None = "Returns a list of materialized lake views execution definitions for a lakehouse."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_lakehouse_mlv_execution_definitions(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_lakehouse_mlv_execution_definitions, **kwargs)


class FabricCreateLakehouseMLVExecutionDefinition(_FabricTool):
    name: str = "fabric_create_lakehouse_mlv_execution_definition"
    description: str | None = "Creates a materialized lake views execution definition."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_lakehouse_mlv_execution_definition(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            display_name: str = Field(
                ..., description="The materialized lake views execution definition display name."
            ),
            current_lakehouse_execution_context: dict[str, Any] = Field(
                ...,
                description="The current lakehouse execution context for a materialized lake views execution definition.",
            ),
            description: str | None = Field(
                None, description="A short description of the materialized lake views execution definition."
            ),
            settings: dict[str, Any] | None = Field(
                None, description="The materialized lake views execution definition settings."
            ),
            extended_lineage_execution_context: dict[str, Any] | None = Field(
                None,
                description="The extended lineage execution context for a materialized lake views execution definition.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions",
                body=_drop_none(
                    {
                        "displayName": display_name,
                        "description": description,
                        "settings": settings,
                        "currentLakehouseExecutionContext": current_lakehouse_execution_context,
                        "extendedLineageExecutionContext": extended_lineage_execution_context,
                    }
                ),
            )

        super().__init__(handler=_create_lakehouse_mlv_execution_definition, **kwargs)


class FabricDeleteLakehouseMLVExecutionDefinition(_FabricTool):
    name: str = "fabric_delete_lakehouse_mlv_execution_definition"
    description: str | None = "Deletes the specified materialized lake views execution definition."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_lakehouse_mlv_execution_definition(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            mlv_execution_definition_id: str = Field(
                ..., description="The materialized lake views execution definition ID."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions/{mlv_execution_definition_id}",
            )

        super().__init__(handler=_delete_lakehouse_mlv_execution_definition, **kwargs)


class FabricGetLakehouseMLVExecutionDefinition(_FabricTool):
    name: str = "fabric_get_lakehouse_mlv_execution_definition"
    description: str | None = "Returns the specified materialized lake views execution definition."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_lakehouse_mlv_execution_definition(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            mlv_execution_definition_id: str = Field(
                ..., description="The materialized lake views execution definition ID."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions/{mlv_execution_definition_id}",
            )

        super().__init__(handler=_get_lakehouse_mlv_execution_definition, **kwargs)


class FabricUpdateLakehouseMLVExecutionDefinition(_FabricTool):
    name: str = "fabric_update_lakehouse_mlv_execution_definition"
    description: str | None = (
        "Updates the specified materialized lake views execution definition. Only the fields provided in the request "
        "body are updated; omitted fields retain their existing values."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_lakehouse_mlv_execution_definition(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            mlv_execution_definition_id: str = Field(
                ..., description="The materialized lake views execution definition ID."
            ),
            display_name: str | None = Field(
                None, description="The materialized lake views execution definition display name."
            ),
            description: str | None = Field(
                None, description="A short description of the materialized lake views execution definition."
            ),
            settings: dict[str, Any] | None = Field(
                None, description="The materialized lake views execution definition settings."
            ),
            current_lakehouse_execution_context: dict[str, Any] | None = Field(
                None,
                description="The current lakehouse execution context for a materialized lake views execution definition.",
            ),
            extended_lineage_execution_context: dict[str, Any] | None = Field(
                None,
                description="The extended lineage execution context for a materialized lake views execution definition.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/mlvexecutiondefinitions/{mlv_execution_definition_id}",
                body=_drop_none(
                    {
                        "displayName": display_name,
                        "description": description,
                        "settings": settings,
                        "currentLakehouseExecutionContext": current_lakehouse_execution_context,
                        "extendedLineageExecutionContext": extended_lineage_execution_context,
                    }
                ),
            )

        super().__init__(handler=_update_lakehouse_mlv_execution_definition, **kwargs)


class FabricLoadLakehouseSchemaTable(_FabricTool):
    name: str = "fabric_load_lakehouse_schema_table"
    description: str | None = (
        "Starts a load table operation for a table within a schema enabled lakehouse and returns the operation status "
        "URL in the response location header. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _load_lakehouse_schema_table(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse item ID."),
            schema_name: str = Field(..., description="The schema name."),
            table_name: str = Field(..., description="The table name."),
            relative_path: str = Field(..., description="The relative path of the data file or folder."),
            path_type: str = Field(
                ...,
                description=(
                    "The type of `relativePath`, either file or folder. Additional `PathType` types may be added over "
                    "time. Allowed values: File, Folder."
                ),
            ),
            file_extension: str | None = Field(None, description="The file extension of the data file."),
            mode: str | None = Field(
                None,
                description=(
                    "The load table operation mode, overwrite or append. Additional mode types may be added over time. "
                    "Allowed values: Overwrite, Append."
                ),
            ),
            recursive: bool | None = Field(
                None,
                description=(
                    "Indicates whether to search data files recursively or not, when loading a table from a folder."
                ),
            ),
            format_options: dict[str, Any] | None = Field(
                None, description="Abstract type of data file format options."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/schemas/{_quote(schema_name)}/tables/{_quote(table_name)}/load",
                params={"beta": "true"},
                body=_drop_none(
                    {
                        "relativePath": relative_path,
                        "pathType": path_type,
                        "fileExtension": file_extension,
                        "mode": mode,
                        "recursive": recursive,
                        "formatOptions": format_options,
                    }
                ),
            )

        super().__init__(handler=_load_lakehouse_schema_table, **kwargs)


class FabricListLakehouseTables(_FabricTool):
    name: str = "fabric_list_lakehouse_tables"
    description: str | None = "Returns a list of lakehouse Tables. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_lakehouse_tables(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse ID."),
            max_results: int | None = Field(None, description="The maximum number of results per page to return."),
            continuation_token: str | None = Field(
                None, description="Token to retrieve the next page of results, if available."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/tables",
                params={"maxResults": max_results, "continuationToken": continuation_token},
            )

        super().__init__(handler=_list_lakehouse_tables, **kwargs)


class FabricLoadLakehouseTable(_FabricTool):
    name: str = "fabric_load_lakehouse_table"
    description: str | None = (
        "Starts a load table operation and returns the operation status URL in the response location header. Preview "
        "API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _load_lakehouse_table(
            workspace_id: str = Field(..., description="The workspace ID."),
            lakehouse_id: str = Field(..., description="The lakehouse item ID."),
            table_name: str = Field(..., description="The table name."),
            relative_path: str = Field(..., description="The relative path of the data file or folder."),
            path_type: str = Field(
                ...,
                description=(
                    "The type of `relativePath`, either file or folder. Additional `PathType` types may be added over "
                    "time. Allowed values: File, Folder."
                ),
            ),
            file_extension: str | None = Field(None, description="The file extension of the data file."),
            mode: str | None = Field(
                None,
                description=(
                    "The load table operation mode, overwrite or append. Additional mode types may be added over time. "
                    "Allowed values: Overwrite, Append."
                ),
            ),
            recursive: bool | None = Field(
                None,
                description=(
                    "Indicates whether to search data files recursively or not, when loading a table from a folder."
                ),
            ),
            format_options: dict[str, Any] | None = Field(
                None, description="Abstract type of data file format options."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/lakehouses/{lakehouse_id}/tables/{_quote(table_name)}/load",
                body=_drop_none(
                    {
                        "relativePath": relative_path,
                        "pathType": path_type,
                        "fileExtension": file_extension,
                        "mode": mode,
                        "recursive": recursive,
                        "formatOptions": format_options,
                    }
                ),
            )

        super().__init__(handler=_load_lakehouse_table, **kwargs)


class FabricRunNotebook(_FabricTool):
    name: str = "fabric_run_notebook"
    description: str | None = "Run on-demand notebook job instance. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _run_notebook(
            workspace_id: str = Field(..., description="The workspace ID."),
            notebook_id: str = Field(..., description="The notebook item ID."),
            execution_data: dict[str, Any] | None = Field(None, description="Execution data for the notebook run."),
            parameters: list[Any] | None = Field(
                None,
                description=(
                    "The parameter list for run on-demand job request. Per-run, user-defined inputs to tailor this "
                    "invocation. Note: parameter names are case-insensitive, but the casing must match the parameter name "
                    "used in the code cell."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/notebooks/{notebook_id}/jobs/execute/instances",
                params={"beta": "false"},
                body=_drop_none({"executionData": execution_data, "parameters": parameters}),
            )

        super().__init__(handler=_run_notebook, **kwargs)


class FabricGetNotebookJobInstance(_FabricTool):
    name: str = "fabric_get_notebook_job_instance"
    description: str | None = "Get one notebook's job instance. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_notebook_job_instance(
            workspace_id: str = Field(..., description="The workspace ID."),
            notebook_id: str = Field(..., description="The notebook ID."),
            job_instance_id: str = Field(..., description="The job instance ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/notebooks/{notebook_id}/jobs/execute/instances/{job_instance_id}",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_notebook_job_instance, **kwargs)


class FabricListNotebookLivySessions(_FabricTool):
    name: str = "fabric_list_notebook_livy_sessions"
    description: str | None = "Returns a list of livy sessions from the specified item identifier."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_notebook_livy_sessions(
            workspace_id: str = Field(..., description="The workspace identifier."),
            notebook_id: str = Field(..., description="The notebook ID."),
            continuation_token: str | None = Field(
                None, description="Token to retrieve the next page of results, if available."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/notebooks/{notebook_id}/livySessions",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_notebook_livy_sessions, **kwargs)


class FabricGetNotebookLivySession(_FabricTool):
    name: str = "fabric_get_notebook_livy_session"
    description: str | None = "Returns properties of the specified livy session."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_notebook_livy_session(
            workspace_id: str = Field(..., description="The workspace identifier."),
            notebook_id: str = Field(..., description="The notebook ID."),
            livy_id: str = Field(..., description="The session identifier."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/notebooks/{notebook_id}/livySessions/{_quote(livy_id)}",
            )

        super().__init__(handler=_get_notebook_livy_session, **kwargs)


class FabricListCapacitySparkCustomPools(_FabricTool):
    name: str = "fabric_list_capacity_spark_custom_pools"
    description: str | None = "List custom pools. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_capacity_spark_custom_pools(
            capacity_id: str = Field(..., description="The capacity ID."),
            continuation_token: str | None = Field(
                None, description="Continuation token. Used to get the next items in the list."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/capacities/{capacity_id}/spark/pools",
                params={"continuationToken": continuation_token, "beta": "true"},
            )

        super().__init__(handler=_list_capacity_spark_custom_pools, **kwargs)


class FabricCreateCapacitySparkCustomPool(_FabricTool):
    name: str = "fabric_create_capacity_spark_custom_pool"
    description: str | None = "Create custom pool. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_capacity_spark_custom_pool(
            capacity_id: str = Field(..., description="The capacity ID."),
            name: str = Field(
                ...,
                description=(
                    "Custom pool name.<br>The name must be between 1 and 64 characters long and must contain only "
                    "letters, numbers, dashes, underscores and spaces.<br>Custom pool names must be unique within the "
                    'workspace.<br>"Starter Pool" is a reserved custom pool name.'
                ),
            ),
            node_family: str = Field(
                ...,
                description=(
                    "Node family. Additional `NodeFamily` types may be added over time. Allowed values: MemoryOptimized."
                ),
            ),
            node_size: str = Field(
                ...,
                description=(
                    "Node size. Additional `NodeSize` types may be added over time. Allowed values: Small, Medium, Large, "
                    "XLarge, XXLarge."
                ),
            ),
            auto_scale: dict[str, Any] = Field(..., description="Autoscale properties."),
            dynamic_executor_allocation: dict[str, Any] = Field(
                ..., description="Dynamic executor allocation proerties."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/capacities/{capacity_id}/spark/pools",
                params={"beta": "true"},
                body=_drop_none(
                    {
                        "name": name,
                        "nodeFamily": node_family,
                        "nodeSize": node_size,
                        "autoScale": auto_scale,
                        "dynamicExecutorAllocation": dynamic_executor_allocation,
                    }
                ),
            )

        super().__init__(handler=_create_capacity_spark_custom_pool, **kwargs)


class FabricDeleteCapacitySparkCustomPool(_FabricTool):
    name: str = "fabric_delete_capacity_spark_custom_pool"
    description: str | None = "Delete custom pool. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_capacity_spark_custom_pool(
            capacity_id: str = Field(..., description="The capacity ID."),
            pool_id: str = Field(..., description="The custom pool ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/capacities/{capacity_id}/spark/pools/{pool_id}",
                params={"beta": "true"},
            )

        super().__init__(handler=_delete_capacity_spark_custom_pool, **kwargs)


class FabricGetCapacitySparkCustomPool(_FabricTool):
    name: str = "fabric_get_capacity_spark_custom_pool"
    description: str | None = "Get custom pool. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_capacity_spark_custom_pool(
            capacity_id: str = Field(..., description="The capacity ID."),
            pool_id: str = Field(..., description="The custom pool ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/capacities/{capacity_id}/spark/pools/{pool_id}",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_capacity_spark_custom_pool, **kwargs)


class FabricUpdateCapacitySparkCustomPool(_FabricTool):
    name: str = "fabric_update_capacity_spark_custom_pool"
    description: str | None = "Update custom pool. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_capacity_spark_custom_pool(
            capacity_id: str = Field(..., description="The capacity ID."),
            pool_id: str = Field(..., description="The custom pool ID."),
            name: str | None = Field(
                None,
                description=(
                    "Custom pool name.<br>The name must be between 1 and 64 characters long and must contain only "
                    "letters, numbers, dashes, underscores and spaces.<br>Custom pool names must be unique within the "
                    'workspace.<br>"Starter Pool" is a reserved custom pool name.'
                ),
            ),
            node_family: str | None = Field(
                None,
                description=(
                    "Node family. Additional `NodeFamily` types may be added over time. Allowed values: MemoryOptimized."
                ),
            ),
            node_size: str | None = Field(
                None,
                description=(
                    "Node size. Additional `NodeSize` types may be added over time. Allowed values: Small, Medium, Large, "
                    "XLarge, XXLarge."
                ),
            ),
            auto_scale: dict[str, Any] | None = Field(None, description="Autoscale properties."),
            dynamic_executor_allocation: dict[str, Any] | None = Field(
                None, description="Dynamic executor allocation proerties."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/capacities/{capacity_id}/spark/pools/{pool_id}",
                params={"beta": "true"},
                body=_drop_none(
                    {
                        "name": name,
                        "nodeFamily": node_family,
                        "nodeSize": node_size,
                        "autoScale": auto_scale,
                        "dynamicExecutorAllocation": dynamic_executor_allocation,
                    }
                ),
            )

        super().__init__(handler=_update_capacity_spark_custom_pool, **kwargs)


class FabricGetCapacitySparkSettings(_FabricTool):
    name: str = "fabric_get_capacity_spark_settings"
    description: str | None = "Get capacity spark settings. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_capacity_spark_settings(
            capacity_id: str = Field(..., description="The capacity ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/capacities/{capacity_id}/spark/settings",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_capacity_spark_settings, **kwargs)


class FabricUpdateCapacitySparkSettings(_FabricTool):
    name: str = "fabric_update_capacity_spark_settings"
    description: str | None = "Update capacity spark settings. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_capacity_spark_settings(
            capacity_id: str = Field(..., description="The capacity ID."),
            workspace_custom_pools_support: str | None = Field(
                None,
                description=(
                    "Feature enablement. Additional `Enablement` types may be added over time. Allowed values: Enabled, "
                    "Disabled."
                ),
            ),
            workspace_starter_pool_status: str | None = Field(
                None,
                description=(
                    "Feature enablement. Additional `Enablement` types may be added over time. Allowed values: Enabled, "
                    "Disabled."
                ),
            ),
            job_burst_support: str | None = Field(
                None,
                description=(
                    "Feature enablement. Additional `Enablement` types may be added over time. Allowed values: Enabled, "
                    "Disabled."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/capacities/{capacity_id}/spark/settings",
                params={"beta": "true"},
                body=_drop_none(
                    {
                        "workspaceCustomPoolsSupport": workspace_custom_pools_support,
                        "workspaceStarterPoolStatus": workspace_starter_pool_status,
                        "jobBurstSupport": job_burst_support,
                    }
                ),
            )

        super().__init__(handler=_update_capacity_spark_settings, **kwargs)


class FabricListWorkspaceLivySessions(_FabricTool):
    name: str = "fabric_list_workspace_livy_sessions"
    description: str | None = "Returns a list of livy sessions from the specified workspace."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspace_livy_sessions(
            workspace_id: str = Field(..., description="The workspace identifier."),
            continuation_token: str | None = Field(
                None, description="Token to retrieve the next page of results, if available."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/spark/livySessions",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_workspace_livy_sessions, **kwargs)


class FabricListWorkspaceSparkCustomPools(_FabricTool):
    name: str = "fabric_list_workspace_spark_custom_pools"
    description: str | None = "List custom pools."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_workspace_spark_custom_pools(
            workspace_id: str = Field(..., description="The workspace ID."),
            continuation_token: str | None = Field(
                None, description="Continuation token. Used to get the next items in the list."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/spark/pools",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_workspace_spark_custom_pools, **kwargs)


class FabricCreateWorkspaceSparkCustomPool(_FabricTool):
    name: str = "fabric_create_workspace_spark_custom_pool"
    description: str | None = "Create custom pool."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_workspace_spark_custom_pool(
            workspace_id: str = Field(..., description="The workspace ID."),
            name: str = Field(
                ...,
                description=(
                    "Custom pool name.<br>The name must be between 1 and 64 characters long and must contain only "
                    "letters, numbers, dashes, underscores and spaces.<br>Custom pool names must be unique within the "
                    'workspace.<br>"Starter Pool" is a reserved custom pool name.'
                ),
            ),
            node_family: str = Field(
                ...,
                description=(
                    "Node family. Additional `NodeFamily` types may be added over time. Allowed values: MemoryOptimized."
                ),
            ),
            node_size: str = Field(
                ...,
                description=(
                    "Node size. Additional `NodeSize` types may be added over time. Allowed values: Small, Medium, Large, "
                    "XLarge, XXLarge."
                ),
            ),
            auto_scale: dict[str, Any] = Field(..., description="Autoscale properties."),
            dynamic_executor_allocation: dict[str, Any] = Field(
                ..., description="Dynamic executor allocation proerties."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/spark/pools",
                body=_drop_none(
                    {
                        "name": name,
                        "nodeFamily": node_family,
                        "nodeSize": node_size,
                        "autoScale": auto_scale,
                        "dynamicExecutorAllocation": dynamic_executor_allocation,
                    }
                ),
            )

        super().__init__(handler=_create_workspace_spark_custom_pool, **kwargs)


class FabricDeleteWorkspaceSparkCustomPool(_FabricTool):
    name: str = "fabric_delete_workspace_spark_custom_pool"
    description: str | None = "Delete custom pool."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_workspace_spark_custom_pool(
            workspace_id: str = Field(..., description="The workspace ID."),
            pool_id: str = Field(..., description="The custom pool ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/spark/pools/{pool_id}",
            )

        super().__init__(handler=_delete_workspace_spark_custom_pool, **kwargs)


class FabricGetWorkspaceSparkCustomPool(_FabricTool):
    name: str = "fabric_get_workspace_spark_custom_pool"
    description: str | None = "Get custom pool."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_spark_custom_pool(
            workspace_id: str = Field(..., description="The workspace ID."),
            pool_id: str = Field(..., description="The custom pool ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/spark/pools/{pool_id}",
            )

        super().__init__(handler=_get_workspace_spark_custom_pool, **kwargs)


class FabricUpdateWorkspaceSparkCustomPool(_FabricTool):
    name: str = "fabric_update_workspace_spark_custom_pool"
    description: str | None = "Update custom pool."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_workspace_spark_custom_pool(
            workspace_id: str = Field(..., description="The workspace ID."),
            pool_id: str = Field(..., description="The custom pool ID."),
            name: str | None = Field(
                None,
                description=(
                    "Custom pool name.<br>The name must be between 1 and 64 characters long and must contain only "
                    "letters, numbers, dashes, underscores and spaces.<br>Custom pool names must be unique within the "
                    'workspace.<br>"Starter Pool" is a reserved custom pool name.'
                ),
            ),
            node_family: str | None = Field(
                None,
                description=(
                    "Node family. Additional `NodeFamily` types may be added over time. Allowed values: MemoryOptimized."
                ),
            ),
            node_size: str | None = Field(
                None,
                description=(
                    "Node size. Additional `NodeSize` types may be added over time. Allowed values: Small, Medium, Large, "
                    "XLarge, XXLarge."
                ),
            ),
            auto_scale: dict[str, Any] | None = Field(None, description="Autoscale properties."),
            dynamic_executor_allocation: dict[str, Any] | None = Field(
                None, description="Dynamic executor allocation proerties."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/spark/pools/{pool_id}",
                body=_drop_none(
                    {
                        "name": name,
                        "nodeFamily": node_family,
                        "nodeSize": node_size,
                        "autoScale": auto_scale,
                        "dynamicExecutorAllocation": dynamic_executor_allocation,
                    }
                ),
            )

        super().__init__(handler=_update_workspace_spark_custom_pool, **kwargs)


class FabricGetWorkspaceSparkSettings(_FabricTool):
    name: str = "fabric_get_workspace_spark_settings"
    description: str | None = "Get workspace spark settings."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_workspace_spark_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/spark/settings",
            )

        super().__init__(handler=_get_workspace_spark_settings, **kwargs)


class FabricUpdateWorkspaceSparkSettings(_FabricTool):
    name: str = "fabric_update_workspace_spark_settings"
    description: str | None = "Update workspace spark settings."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_workspace_spark_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            automatic_log: dict[str, Any] | None = Field(None, description="Automatic Log Properties."),
            high_concurrency: dict[str, Any] | None = Field(None, description="High Concurrency Properties."),
            pool: dict[str, Any] | None = Field(None, description="Properties of a pool"),
            environment: dict[str, Any] | None = Field(None, description="Properties of an environment."),
            job: dict[str, Any] | None = Field(None, description="Properties of a Spark job."),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/spark/settings",
                body=_drop_none(
                    {
                        "automaticLog": automatic_log,
                        "highConcurrency": high_concurrency,
                        "pool": pool,
                        "environment": environment,
                        "job": job,
                    }
                ),
            )

        super().__init__(handler=_update_workspace_spark_settings, **kwargs)


class FabricRunSparkJobDefinition(_FabricTool):
    name: str = "fabric_run_spark_job_definition"
    description: str | None = (
        "Run on-demand [Spark job definition](/fabric/data-engineering/run-spark-job-definition) job instance."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _run_spark_job_definition(
            workspace_id: str = Field(..., description="The workspace ID."),
            spark_job_definition_id: str = Field(..., description="The Spark job definition item ID."),
            execution_data: dict[str, Any] | None = Field(
                None,
                description="ExecutionData for spark job definition run if customer wants to override default values.",
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/sparkJobDefinitions/{spark_job_definition_id}/jobs/sparkjob/instances",
                body=_drop_none({"executionData": execution_data}),
            )

        super().__init__(handler=_run_spark_job_definition, **kwargs)


class FabricListSparkJobDefinitionLivySessions(_FabricTool):
    name: str = "fabric_list_spark_job_definition_livy_sessions"
    description: str | None = "Returns a list of livy sessions from the specified item identifier."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_spark_job_definition_livy_sessions(
            workspace_id: str = Field(..., description="The workspace identifier."),
            spark_job_definition_id: str = Field(..., description="The sparkJobDefinition ID."),
            continuation_token: str | None = Field(
                None, description="Token to retrieve the next page of results, if available."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/sparkJobDefinitions/{spark_job_definition_id}/livySessions",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_spark_job_definition_livy_sessions, **kwargs)


class FabricGetSparkJobDefinitionLivySession(_FabricTool):
    name: str = "fabric_get_spark_job_definition_livy_session"
    description: str | None = "Returns properties of the specified livy session."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_spark_job_definition_livy_session(
            workspace_id: str = Field(..., description="The workspace identifier."),
            spark_job_definition_id: str = Field(..., description="The sparkJobDefinition ID."),
            livy_id: str = Field(..., description="The session identifier."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/sparkJobDefinitions/{spark_job_definition_id}/livySessions/{_quote(livy_id)}",
            )

        super().__init__(handler=_get_spark_job_definition_livy_session, **kwargs)
