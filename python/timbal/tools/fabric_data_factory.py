"""Microsoft Fabric REST API tools: pipelines, dataflows, copy jobs, Airflow jobs, graph models and mirroring.

Auth, long running operations and request handling live in ``fabric.py``.
"""

from typing import Any

from pydantic import Field

from .fabric import _drop_none, _fabric_request, _FabricTool, _quote, _quote_path, _upload_bytes


class FabricListAirflowPoolTemplates(_FabricTool):
    name: str = "fabric_list_airflow_pool_templates"
    description: str | None = "List Apache Airflow pool templates. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_airflow_pool_templates(
            workspace_id: str = Field(..., description="The workspace ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/poolTemplates",
                params={"continuationToken": continuation_token, "beta": "true"},
            )

        super().__init__(handler=_list_airflow_pool_templates, **kwargs)


class FabricCreateAirflowPoolTemplate(_FabricTool):
    name: str = "fabric_create_airflow_pool_template"
    description: str | None = "Create an Apache Airflow pool template. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_airflow_pool_template(
            workspace_id: str = Field(..., description="The workspace ID."),
            name: str = Field(..., description="The pool template name."),
            node_size: str = Field(
                ...,
                description="Node size. Additional `NodeSize` types may be added over time. Allowed values: Small, Large.",
            ),
            compute_scalability: dict[str, Any] = Field(..., description="Compute scalability properties."),
            apache_airflow_job_version: str = Field(..., description="The Apache Airflow job version (e.g., '1.0.0')."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/poolTemplates",
                params={"beta": "true"},
                body=_drop_none(
                    {
                        "name": name,
                        "nodeSize": node_size,
                        "computeScalability": compute_scalability,
                        "apacheAirflowJobVersion": apache_airflow_job_version,
                    }
                ),
            )

        super().__init__(handler=_create_airflow_pool_template, **kwargs)


class FabricDeleteAirflowPoolTemplate(_FabricTool):
    name: str = "fabric_delete_airflow_pool_template"
    description: str | None = "Delete an Apache Airflow pool template. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_airflow_pool_template(
            workspace_id: str = Field(..., description="The workspace ID."),
            pool_template_id: str = Field(..., description="The Apache Airflow pool template ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/poolTemplates/{pool_template_id}",
                params={"beta": "true"},
            )

        super().__init__(handler=_delete_airflow_pool_template, **kwargs)


class FabricGetAirflowPoolTemplate(_FabricTool):
    name: str = "fabric_get_airflow_pool_template"
    description: str | None = "Get an Apache Airflow pool template. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_airflow_pool_template(
            workspace_id: str = Field(..., description="The workspace ID."),
            pool_template_id: str = Field(..., description="The Apache Airflow pool template ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/poolTemplates/{pool_template_id}",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_airflow_pool_template, **kwargs)


class FabricGetAirflowWorkspaceSettings(_FabricTool):
    name: str = "fabric_get_airflow_workspace_settings"
    description: str | None = "Get Apache Airflow workspace settings. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_airflow_workspace_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/settings",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_airflow_workspace_settings, **kwargs)


class FabricUpdateAirflowWorkspaceSettings(_FabricTool):
    name: str = "fabric_update_airflow_workspace_settings"
    description: str | None = "Update Apache Airflow workspace settings. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_airflow_workspace_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            default_pool_template_id: str | None = Field(
                None, description="The default pool template ID for the workspace."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/settings",
                params={"beta": "true"},
                body=_drop_none({"defaultPoolTemplateId": default_pool_template_id}),
            )

        super().__init__(handler=_update_airflow_workspace_settings, **kwargs)


class FabricGetApacheAirflowJobEnvironment(_FabricTool):
    name: str = "fabric_get_apache_airflow_job_environment"
    description: str | None = (
        "Gets the Apache Airflow environment for the specified Apache Airflow job. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_apache_airflow_job_environment(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_apache_airflow_job_environment, **kwargs)


class FabricGetApacheAirflowJobCompute(_FabricTool):
    name: str = "fabric_get_apache_airflow_job_compute"
    description: str | None = (
        "Returns the compute configuration for the specified Apache Airflow job environment. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_apache_airflow_job_compute(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/compute",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_apache_airflow_job_compute, **kwargs)


class FabricDeployApacheAirflowJobRequirements(_FabricTool):
    name: str = "fabric_deploy_apache_airflow_job_requirements"
    description: str | None = "Deploys requirements for an Apache Airflow job environment. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _deploy_apache_airflow_job_requirements(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
            content: str = Field(..., description="Text content to upload (UTF-8)."),
            file_path: str | None = Field(
                None,
                description=(
                    "The path to an existing requirements file to deploy. If not provided, the request body should "
                    "contain the requirements file content."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/deployRequirements",
                params={"filePath": file_path, "beta": "true"},
                content=_upload_bytes(content, None),
                content_type="text/plain",
            )

        super().__init__(handler=_deploy_apache_airflow_job_requirements, **kwargs)


class FabricListApacheAirflowJobLibraries(_FabricTool):
    name: str = "fabric_list_apache_airflow_job_libraries"
    description: str | None = (
        "Returns a list of installed libraries for the specified Apache Airflow job environment. Preview API: may "
        "change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_apache_airflow_job_libraries(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/libraries",
                params={"continuationToken": continuation_token, "beta": "true"},
            )

        super().__init__(handler=_list_apache_airflow_job_libraries, **kwargs)


class FabricGetApacheAirflowJobSettings(_FabricTool):
    name: str = "fabric_get_apache_airflow_job_settings"
    description: str | None = (
        "Returns the environment settings for the specified Apache Airflow job. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _get_apache_airflow_job_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/settings",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_apache_airflow_job_settings, **kwargs)


class FabricStartApacheAirflowJobEnvironment(_FabricTool):
    name: str = "fabric_start_apache_airflow_job_environment"
    description: str | None = "Starts an Apache Airflow job environment. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _start_apache_airflow_job_environment(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/start",
                params={"beta": "true"},
            )

        super().__init__(handler=_start_apache_airflow_job_environment, **kwargs)


class FabricStopApacheAirflowJobEnvironment(_FabricTool):
    name: str = "fabric_stop_apache_airflow_job_environment"
    description: str | None = "Stops an Apache Airflow job environment. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _stop_apache_airflow_job_environment(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/stop",
                params={"beta": "true"},
            )

        super().__init__(handler=_stop_apache_airflow_job_environment, **kwargs)


class FabricUpdateApacheAirflowJobCompute(_FabricTool):
    name: str = "fabric_update_apache_airflow_job_compute"
    description: str | None = (
        "Updates the compute configuration for an Apache Airflow job environment. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _update_apache_airflow_job_compute(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
            pool_template_id: str = Field(..., description="The pool template ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/updateCompute",
                params={"beta": "true"},
                body=_drop_none({"poolTemplateId": pool_template_id}),
            )

        super().__init__(handler=_update_apache_airflow_job_compute, **kwargs)


class FabricUpdateApacheAirflowJobSettings(_FabricTool):
    name: str = "fabric_update_apache_airflow_job_settings"
    description: str | None = "Updates the settings for an Apache Airflow job environment. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_apache_airflow_job_settings(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
            environment_variables: list[Any] | None = Field(
                None,
                description=(
                    "Environment variables to configure for the Airflow environment. When updating, users must submit the "
                    "complete set of desired values; existing values will be replaced."
                ),
            ),
            airflow_configuration_overrides: list[Any] | None = Field(
                None,
                description=(
                    "Airflow configuration overrides. When updating, users must submit the complete set of desired "
                    "values; existing values will be replaced."
                ),
            ),
            triggerers: str | None = Field(
                None,
                description=(
                    "Triggerers status. Additional `TriggerersStatus` values may be added over time. Allowed values: "
                    "Enabled, Disabled."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/environment/updateSettings",
                params={"beta": "true"},
                body=_drop_none(
                    {
                        "environmentVariables": environment_variables,
                        "airflowConfigurationOverrides": airflow_configuration_overrides,
                        "triggerers": triggerers,
                    }
                ),
            )

        super().__init__(handler=_update_apache_airflow_job_settings, **kwargs)


class FabricListApacheAirflowJobFiles(_FabricTool):
    name: str = "fabric_list_apache_airflow_job_files"
    description: str | None = (
        "Returns a list of Apache Airflow job files from the specified Apache Airflow job. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_apache_airflow_job_files(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
            root_path: str | None = Field(
                None, description="The folder path to list. If not provided, the root directory is used."
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/files",
                params={"rootPath": root_path, "continuationToken": continuation_token, "beta": "true"},
            )

        super().__init__(handler=_list_apache_airflow_job_files, **kwargs)


class FabricDeleteApacheAirflowJobFile(_FabricTool):
    name: str = "fabric_delete_apache_airflow_job_file"
    description: str | None = "Deletes the specified Apache Airflow job file. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_apache_airflow_job_file(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
            file_path: str = Field(
                ...,
                description=(
                    "The file path relative to the Apache Airflow job root. It must begin with either 'dags/' or "
                    "'plugins/' (for example, `dags/example_dag.py`)."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/files/{_quote_path(file_path)}",
                params={"beta": "true"},
            )

        super().__init__(handler=_delete_apache_airflow_job_file, **kwargs)


class FabricGetApacheAirflowJobFile(_FabricTool):
    name: str = "fabric_get_apache_airflow_job_file"
    description: str | None = "Returns the specified Apache Airflow job file. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_apache_airflow_job_file(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
            file_path: str = Field(
                ...,
                description=(
                    "The file path relative to the Apache Airflow job root. It must begin with either 'dags/' or "
                    "'plugins/' (for example, `dags/example_dag.py`)."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/files/{_quote_path(file_path)}",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_apache_airflow_job_file, **kwargs)


class FabricCreateOrUpdateApacheAirflowJobFile(_FabricTool):
    name: str = "fabric_create_or_update_apache_airflow_job_file"
    description: str | None = "Creates or updates an Apache Airflow job file. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_or_update_apache_airflow_job_file(
            workspace_id: str = Field(..., description="The workspace ID."),
            apache_airflow_job_id: str = Field(..., description="The Apache Airflow job ID."),
            file_path: str = Field(
                ...,
                description=(
                    "The file path relative to the Apache Airflow job root. It must begin with either 'dags/' or "
                    "'plugins/' (for example, `dags/example_dag.py`)."
                ),
            ),
            content: str | None = Field(None, description="UTF-8 text content. Provide this or content_base64."),
            content_base64: str | None = Field(
                None, description="Base64-encoded binary content. Provide this or content."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "PUT",
                f"/workspaces/{workspace_id}/apacheAirflowJobs/{apache_airflow_job_id}/files/{_quote_path(file_path)}",
                params={"beta": "true"},
                content=_upload_bytes(content, content_base64),
                content_type="application/octet-stream",
            )

        super().__init__(handler=_create_or_update_apache_airflow_job_file, **kwargs)


class FabricResetCopyJob(_FabricTool):
    name: str = "fabric_reset_copy_job"
    description: str | None = (
        "Resets the specified CopyJob. If `resetAllCopyJobEntities` is true, all entities are reset. Otherwise, only "
        "the specified entities are reset."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _reset_copy_job(
            workspace_id: str = Field(..., description="The workspace ID."),
            copy_job_id: str = Field(..., description="The CopyJob ID."),
            reset_all_copy_job_entities: bool | None = Field(
                None, description="Whether to reset all CopyJob entities."
            ),
            copy_job_entities_to_reset: list[Any] | None = Field(
                None, description="A list of CopyJob entities to reset."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/copyJobs/{copy_job_id}/resetCopyJob",
                body=_drop_none(
                    {
                        "resetAllCopyJobEntities": reset_all_copy_job_entities,
                        "copyJobEntitiesToReset": copy_job_entities_to_reset,
                    }
                ),
            )

        super().__init__(handler=_reset_copy_job, **kwargs)


class FabricRunDataBuildToolJob(_FabricTool):
    name: str = "fabric_run_data_build_tool_job"
    description: str | None = "Run on-demand DataBuildToolJob execute job instance. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _run_data_build_tool_job(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_build_tool_job_id: str = Field(..., description="The DataBuildToolJob item ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataBuildToolJobs/{data_build_tool_job_id}/jobs/execute/instances",
            )

        super().__init__(handler=_run_data_build_tool_job, **kwargs)


class FabricCreateDataBuildToolJobSchedule(_FabricTool):
    name: str = "fabric_create_data_build_tool_job_schedule"
    description: str | None = "Create a new execute schedule for a DataBuildToolJob. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_data_build_tool_job_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_build_tool_job_id: str = Field(..., description="The DataBuildToolJob ID."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataBuildToolJobs/{data_build_tool_job_id}/jobs/execute/schedules",
                body=_drop_none({"enabled": enabled, "configuration": configuration}),
            )

        super().__init__(handler=_create_data_build_tool_job_schedule, **kwargs)


class FabricListDataPipelineJobInstances(_FabricTool):
    name: str = "fabric_list_data_pipeline_job_instances"
    description: str | None = "Returns a list of execute job instances for the specified data pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_pipeline_job_instances(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_pipeline_id: str = Field(..., description="The data pipeline ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/instances",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_data_pipeline_job_instances, **kwargs)


class FabricRunDataPipeline(_FabricTool):
    name: str = "fabric_run_data_pipeline"
    description: str | None = "Runs an on-demand execute job instance."

    def __init__(self, **kwargs: Any) -> None:
        async def _run_data_pipeline(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_pipeline_id: str = Field(..., description="The data pipeline ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/instances",
            )

        super().__init__(handler=_run_data_pipeline, **kwargs)


class FabricGetDataPipelineJobInstance(_FabricTool):
    name: str = "fabric_get_data_pipeline_job_instance"
    description: str | None = "Returns the specified execute job instance for a data pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_pipeline_job_instance(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_pipeline_id: str = Field(..., description="The data pipeline ID."),
            job_instance_id: str = Field(..., description="The job instance ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/instances/{job_instance_id}",
            )

        super().__init__(handler=_get_data_pipeline_job_instance, **kwargs)


class FabricListDataPipelineSchedules(_FabricTool):
    name: str = "fabric_list_data_pipeline_schedules"
    description: str | None = "Returns a list of execute schedules for the specified data pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _list_data_pipeline_schedules(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_pipeline_id: str = Field(..., description="The data pipeline ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_list_data_pipeline_schedules, **kwargs)


class FabricCreateDataPipelineSchedule(_FabricTool):
    name: str = "fabric_create_data_pipeline_schedule"
    description: str | None = "Create a new execute schedule for a data pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _create_data_pipeline_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_pipeline_id: str = Field(..., description="The data pipeline ID."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules",
                body=_drop_none({"enabled": enabled, "configuration": configuration}),
            )

        super().__init__(handler=_create_data_pipeline_schedule, **kwargs)


class FabricDeleteDataPipelineSchedule(_FabricTool):
    name: str = "fabric_delete_data_pipeline_schedule"
    description: str | None = "Deletes the specified execute schedule for a data pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _delete_data_pipeline_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_pipeline_id: str = Field(..., description="The data pipeline ID."),
            schedule_id: str = Field(..., description="The data pipeline schedule ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "DELETE",
                f"/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules/{schedule_id}",
            )

        super().__init__(handler=_delete_data_pipeline_schedule, **kwargs)


class FabricGetDataPipelineSchedule(_FabricTool):
    name: str = "fabric_get_data_pipeline_schedule"
    description: str | None = "Returns the specified execute schedule for a data pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_data_pipeline_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_pipeline_id: str = Field(..., description="The data pipeline ID."),
            schedule_id: str = Field(..., description="The data pipeline schedule ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules/{schedule_id}",
            )

        super().__init__(handler=_get_data_pipeline_schedule, **kwargs)


class FabricUpdateDataPipelineSchedule(_FabricTool):
    name: str = "fabric_update_data_pipeline_schedule"
    description: str | None = "Updates the specified execute schedule for a data pipeline."

    def __init__(self, **kwargs: Any) -> None:
        async def _update_data_pipeline_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            data_pipeline_id: str = Field(..., description="The data pipeline ID."),
            schedule_id: str = Field(..., description="The data pipeline schedule ID."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
        ) -> Any:
            return await _fabric_request(
                self,
                "PATCH",
                f"/workspaces/{workspace_id}/dataPipelines/{data_pipeline_id}/jobs/execute/schedules/{schedule_id}",
                body=_drop_none({"enabled": enabled, "configuration": configuration}),
            )

        super().__init__(handler=_update_data_pipeline_schedule, **kwargs)


class FabricUpgradeDataflowsGen1(_FabricTool):
    name: str = "fabric_upgrade_dataflows_gen1"
    description: str | None = "Upgrades Gen1 dataflows in the specified workspace. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _upgrade_dataflows_gen1(
            workspace_id: str = Field(..., description="The workspace ID."),
            dataflows: list[Any] = Field(..., description="The Gen1 dataflows to upgrade in the specified workspace."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataflows/gen1Upgrade",
                body=_drop_none({"dataflows": dataflows}),
            )

        super().__init__(handler=_upgrade_dataflows_gen1, **kwargs)


class FabricListDataflowGen1UpgradeReadiness(_FabricTool):
    name: str = "fabric_list_dataflow_gen1_upgrade_readiness"
    description: str | None = (
        "Lists Gen1 dataflow upgrade readiness results in the specified workspace. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_dataflow_gen1_upgrade_readiness(
            workspace_id: str = Field(..., description="The workspace ID."),
            page_size: int | None = Field(
                None,
                description=(
                    "The maximum number of results per page. The value must be between 1 and 30. When omitted, the server "
                    "selects the page size."
                ),
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataflows/gen1UpgradeReadinessResults",
                params={"pageSize": page_size, "continuationToken": continuation_token},
            )

        super().__init__(handler=_list_dataflow_gen1_upgrade_readiness, **kwargs)


class FabricExecuteDataflowQuery(_FabricTool):
    name: str = "fabric_execute_dataflow_query"
    description: str | None = (
        "Executes a query against a dataflow and returns the result. Executes a specified query against a dataflow "
        "and streams the result back to the caller. Supports using custom mashup documents for advanced scenarios."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _execute_dataflow_query(
            workspace_id: str = Field(..., description="The workspace ID."),
            dataflow_id: str = Field(..., description="The Dataflow ID."),
            query_name: str = Field(
                ...,
                description=(
                    "The name of the query to execute from the dataflow (or from the custom mashup document if provided)."
                ),
            ),
            custom_mashup_document: str | None = Field(
                None, description="Optional custom mashup document to override the dataflow's default mashup."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataflows/{dataflow_id}/executeQuery",
                body=_drop_none({"queryName": query_name, "customMashupDocument": custom_mashup_document}),
            )

        super().__init__(handler=_execute_dataflow_query, **kwargs)


class FabricRunDataflowApplyChanges(_FabricTool):
    name: str = "fabric_run_dataflow_apply_changes"
    description: str | None = "Run on-demand apply changes job instance. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _run_dataflow_apply_changes(
            workspace_id: str = Field(..., description="The workspace ID."),
            dataflow_id: str = Field(..., description="The dataflow ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataflows/{dataflow_id}/jobs/applyChanges/instances",
            )

        super().__init__(handler=_run_dataflow_apply_changes, **kwargs)


class FabricCreateDataflowApplyChangesSchedule(_FabricTool):
    name: str = "fabric_create_dataflow_apply_changes_schedule"
    description: str | None = (
        "Create a new apply changes schedule for a dataflow. A dataflow can create maximum 20 schedulers. Preview "
        "API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_dataflow_apply_changes_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            dataflow_id: str = Field(..., description="The item ID."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
            execution_data: dict[str, Any] | None = Field(
                None,
                description=(
                    "The execution data for a scheduled job. This is fixed static data, defined by the specific item job "
                    "type."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataflows/{dataflow_id}/jobs/applyChanges/schedules",
                body=_drop_none({"enabled": enabled, "configuration": configuration, "executionData": execution_data}),
            )

        super().__init__(handler=_create_dataflow_apply_changes_schedule, **kwargs)


class FabricRunDataflow(_FabricTool):
    name: str = "fabric_run_dataflow"
    description: str | None = "Run on-demand execute job instance. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _run_dataflow(
            workspace_id: str = Field(..., description="The workspace ID."),
            dataflow_id: str = Field(..., description="The dataflow ID."),
            execution_data: dict[str, Any] | None = Field(None, description="The execution data payload for Dataflow"),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataflows/{dataflow_id}/jobs/execute/instances",
                body=_drop_none({"executionData": execution_data}),
            )

        super().__init__(handler=_run_dataflow, **kwargs)


class FabricCreateDataflowSchedule(_FabricTool):
    name: str = "fabric_create_dataflow_schedule"
    description: str | None = (
        "Create a new execute schedule for a dataflow. A dataflow can create maximum 20 schedulers. Preview API: may "
        "change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _create_dataflow_schedule(
            workspace_id: str = Field(..., description="The workspace ID."),
            dataflow_id: str = Field(..., description="The item ID."),
            enabled: bool = Field(
                ..., description="Whether this schedule is enabled. True - Enabled, False - Disabled."
            ),
            configuration: dict[str, Any] = Field(..., description="Item schedule plan detail settings."),
            execution_data: dict[str, Any] | None = Field(
                None,
                description=(
                    "The execution data for a scheduled job. This is fixed static data, defined by the specific item job "
                    "type."
                ),
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/dataflows/{dataflow_id}/jobs/execute/schedules",
                body=_drop_none({"enabled": enabled, "configuration": configuration, "executionData": execution_data}),
            )

        super().__init__(handler=_create_dataflow_schedule, **kwargs)


class FabricDiscoverDataflowParameters(_FabricTool):
    name: str = "fabric_discover_dataflow_parameters"
    description: str | None = "Retrieves all parameters defined in the specified Dataflow."

    def __init__(self, **kwargs: Any) -> None:
        async def _discover_dataflow_parameters(
            workspace_id: str = Field(..., description="The workspace ID."),
            dataflow_id: str = Field(..., description="The Dataflow ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/dataflows/{dataflow_id}/parameters",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_discover_dataflow_parameters, **kwargs)


class FabricExecuteGraphModelQuery(_FabricTool):
    name: str = "fabric_execute_graph_model_query"
    description: str | None = "Executes a query on the specified graph model. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _execute_graph_model_query(
            workspace_id: str = Field(..., description="The workspace ID."),
            graph_model_id: str = Field(..., description="The GraphModel ID."),
            query: str = Field(..., description="The query string."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/graphModels/{graph_model_id}/executeQuery",
                params={"beta": "true"},
                body=_drop_none({"query": query}),
            )

        super().__init__(handler=_execute_graph_model_query, **kwargs)


class FabricGetGraphModelQueryableGraphType(_FabricTool):
    name: str = "fabric_get_graph_model_queryable_graph_type"
    description: str | None = "Get the current queryable graph type. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_graph_model_queryable_graph_type(
            workspace_id: str = Field(..., description="The workspace ID."),
            graph_model_id: str = Field(..., description="The GraphModel ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/graphModels/{graph_model_id}/getQueryableGraphType",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_graph_model_queryable_graph_type, **kwargs)


class FabricRefreshGraphModel(_FabricTool):
    name: str = "fabric_refresh_graph_model"
    description: str | None = "Run on-demand RefreshGraph job model. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _refresh_graph_model(
            workspace_id: str = Field(..., description="The workspace ID."),
            graph_model_id: str = Field(..., description="The GraphModel item ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/graphModels/{graph_model_id}/jobs/refreshGraph/instances",
            )

        super().__init__(handler=_refresh_graph_model, **kwargs)


class FabricDiscoverAzureDatabricksCatalogs(_FabricTool):
    name: str = "fabric_discover_azure_databricks_catalogs"
    description: str | None = "Returns a list of catalogs from Unity Catalog. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _discover_azure_databricks_catalogs(
            workspace_id: str = Field(..., description="The workspace ID."),
            databricks_workspace_connection_id: str = Field(..., description="The Databricks workspace connection ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
            max_results: int | None = Field(None, description="The maximum number of results to return."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/azureDatabricks/catalogs",
                params={
                    "databricksWorkspaceConnectionId": databricks_workspace_connection_id,
                    "continuationToken": continuation_token,
                    "maxResults": max_results,
                },
            )

        super().__init__(handler=_discover_azure_databricks_catalogs, **kwargs)


class FabricDiscoverAzureDatabricksCatalogSchemas(_FabricTool):
    name: str = "fabric_discover_azure_databricks_catalog_schemas"
    description: str | None = (
        "Returns a list of schemas in the given catalog from Unity Catalog. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _discover_azure_databricks_catalog_schemas(
            workspace_id: str = Field(..., description="The workspace ID."),
            catalog_name: str = Field(..., description="The catalog name."),
            databricks_workspace_connection_id: str = Field(..., description="The Databricks workspace connection ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
            max_results: int | None = Field(None, description="The maximum number of results to return."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/azureDatabricks/catalogs/{_quote(catalog_name)}/schemas",
                params={
                    "databricksWorkspaceConnectionId": databricks_workspace_connection_id,
                    "continuationToken": continuation_token,
                    "maxResults": max_results,
                },
            )

        super().__init__(handler=_discover_azure_databricks_catalog_schemas, **kwargs)


class FabricDiscoverAzureDatabricksCatalogSchemaTables(_FabricTool):
    name: str = "fabric_discover_azure_databricks_catalog_schema_tables"
    description: str | None = (
        "Returns a list of tables in the given schema from Unity Catalog. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _discover_azure_databricks_catalog_schema_tables(
            workspace_id: str = Field(..., description="The workspace ID."),
            catalog_name: str = Field(..., description="The catalog name."),
            schema_name: str = Field(..., description="The schema name."),
            databricks_workspace_connection_id: str = Field(..., description="The Databricks workspace connection ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
            max_results: int | None = Field(None, description="The maximum number of results to return."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/azureDatabricks/catalogs/{_quote(catalog_name)}/schemas/{_quote(schema_name)}/tables",
                params={
                    "databricksWorkspaceConnectionId": databricks_workspace_connection_id,
                    "continuationToken": continuation_token,
                    "maxResults": max_results,
                },
            )

        super().__init__(handler=_discover_azure_databricks_catalog_schema_tables, **kwargs)


class FabricRefreshMirroredAzureDatabricksCatalogMetadata(_FabricTool):
    name: str = "fabric_refresh_mirrored_azure_databricks_catalog_metadata"
    description: str | None = (
        "Refresh Databricks catalog metadata in mirroredAzureDatabricksCatalogs Item. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _refresh_mirrored_azure_databricks_catalog_metadata(
            workspace_id: str = Field(..., description="The workspace ID."),
            mirrored_azure_databricks_catalog_id: str = Field(
                ..., description="The mirroredAzureDatabricksCatalog ID."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mirroredAzureDatabricksCatalogs/{mirrored_azure_databricks_catalog_id}/refreshCatalogMetadata",
            )

        super().__init__(handler=_refresh_mirrored_azure_databricks_catalog_metadata, **kwargs)


class FabricListCatalogMirroringScopes(_FabricTool):
    name: str = "fabric_list_catalog_mirroring_scopes"
    description: str | None = (
        "Returns a hierarchical list of namespaces from the source catalog. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_catalog_mirroring_scopes(
            workspace_id: str = Field(..., description="The workspace ID."),
            connection_id: str = Field(..., description="The connection ID to the catalog."),
            parent: list[str] | None = Field(
                None,
                description=(
                    "Parent namespace segments. Pass each segment as a separate repeated query parameter (e.g., "
                    "`parent=Accounting&parent=US`). The segments form the fully qualified parent namespace hierarchy and "
                    "**must be provided in hierarchical order** (root-level segment first, leaf segment last). Omit for "
                    "root namespaces."
                ),
            ),
            recursive: bool | None = Field(None, description="If true, returns nested children. Default: false."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/catalogmirroring/scopes",
                params={
                    "connectionId": connection_id,
                    "parent": parent,
                    "recursive": recursive,
                    "continuationToken": continuation_token,
                    "beta": "true",
                },
            )

        super().__init__(handler=_list_catalog_mirroring_scopes, **kwargs)


class FabricListCatalogMirroringTables(_FabricTool):
    name: str = "fabric_list_catalog_mirroring_tables"
    description: str | None = (
        "Returns a list of tables within a specified namespace scope from the source catalog. Preview API: may change."
    )

    def __init__(self, **kwargs: Any) -> None:
        async def _list_catalog_mirroring_tables(
            workspace_id: str = Field(..., description="The workspace ID."),
            connection_id: str = Field(..., description="The connection ID to the catalog."),
            scope: list[str] | None = Field(
                None,
                description=(
                    "Required. Selectable namespace scope segments as repeated query parameters (e.g., "
                    "`scope=Accounting&scope=US`) in root-first order. Pass empty (`scope=`) for root namespace."
                ),
            ),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/catalogmirroring/tables",
                params={
                    "scope": scope,
                    "connectionId": connection_id,
                    "continuationToken": continuation_token,
                    "beta": "true",
                },
            )

        super().__init__(handler=_list_catalog_mirroring_tables, **kwargs)


class FabricGetMirroredCatalogMirroringStatus(_FabricTool):
    name: str = "fabric_get_mirrored_catalog_mirroring_status"
    description: str | None = "Returns the overall mirroring status of the MirroredCatalog. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_mirrored_catalog_mirroring_status(
            workspace_id: str = Field(..., description="The workspace ID."),
            mirrored_catalog_id: str = Field(..., description="The MirroredCatalog ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/mirroredCatalogs/{mirrored_catalog_id}/mirroringStatus",
                params={"beta": "true"},
            )

        super().__init__(handler=_get_mirrored_catalog_mirroring_status, **kwargs)


class FabricRefreshMirroredCatalogMetadata(_FabricTool):
    name: str = "fabric_refresh_mirrored_catalog_metadata"
    description: str | None = "Triggers a refresh of catalog metadata from the source. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _refresh_mirrored_catalog_metadata(
            workspace_id: str = Field(..., description="The workspace ID."),
            mirrored_catalog_id: str = Field(..., description="The MirroredCatalog ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mirroredCatalogs/{mirrored_catalog_id}/refreshCatalogMetadata",
                params={"beta": "true"},
            )

        super().__init__(handler=_refresh_mirrored_catalog_metadata, **kwargs)


class FabricGetMirroredCatalogTablesMirroringStatus(_FabricTool):
    name: str = "fabric_get_mirrored_catalog_tables_mirroring_status"
    description: str | None = "Returns the per-table mirroring status for the MirroredCatalog. Preview API: may change."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_mirrored_catalog_tables_mirroring_status(
            workspace_id: str = Field(..., description="The workspace ID."),
            mirrored_catalog_id: str = Field(..., description="The MirroredCatalog ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "GET",
                f"/workspaces/{workspace_id}/mirroredCatalogs/{mirrored_catalog_id}/tablesMirroringStatus",
                params={"continuationToken": continuation_token, "beta": "true"},
            )

        super().__init__(handler=_get_mirrored_catalog_tables_mirroring_status, **kwargs)


class FabricGetMirroredDatabaseMirroringStatus(_FabricTool):
    name: str = "fabric_get_mirrored_database_mirroring_status"
    description: str | None = "Get the status of the mirrored database."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_mirrored_database_mirroring_status(
            workspace_id: str = Field(..., description="The workspace ID."),
            mirrored_database_id: str = Field(..., description="The mirrored database ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mirroredDatabases/{mirrored_database_id}/getMirroringStatus",
            )

        super().__init__(handler=_get_mirrored_database_mirroring_status, **kwargs)


class FabricGetMirroredDatabaseTablesMirroringStatus(_FabricTool):
    name: str = "fabric_get_mirrored_database_tables_mirroring_status"
    description: str | None = "Get the mirroring status of the tables."

    def __init__(self, **kwargs: Any) -> None:
        async def _get_mirrored_database_tables_mirroring_status(
            workspace_id: str = Field(..., description="The workspace ID."),
            mirrored_database_id: str = Field(..., description="The mirrored database ID."),
            continuation_token: str | None = Field(
                None, description="A token for retrieving the next page of results."
            ),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mirroredDatabases/{mirrored_database_id}/getTablesMirroringStatus",
                params={"continuationToken": continuation_token},
            )

        super().__init__(handler=_get_mirrored_database_tables_mirroring_status, **kwargs)


class FabricStartMirroredDatabaseMirroring(_FabricTool):
    name: str = "fabric_start_mirrored_database_mirroring"
    description: str | None = "Starts the mirroring."

    def __init__(self, **kwargs: Any) -> None:
        async def _start_mirrored_database_mirroring(
            workspace_id: str = Field(..., description="The workspace ID."),
            mirrored_database_id: str = Field(..., description="The mirrored database ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mirroredDatabases/{mirrored_database_id}/startMirroring",
            )

        super().__init__(handler=_start_mirrored_database_mirroring, **kwargs)


class FabricStopMirroredDatabaseMirroring(_FabricTool):
    name: str = "fabric_stop_mirrored_database_mirroring"
    description: str | None = "Stops the mirroring."

    def __init__(self, **kwargs: Any) -> None:
        async def _stop_mirrored_database_mirroring(
            workspace_id: str = Field(..., description="The workspace ID."),
            mirrored_database_id: str = Field(..., description="The mirrored database ID."),
        ) -> Any:
            return await _fabric_request(
                self,
                "POST",
                f"/workspaces/{workspace_id}/mirroredDatabases/{mirrored_database_id}/stopMirroring",
            )

        super().__init__(handler=_stop_mirrored_database_mirroring, **kwargs)
