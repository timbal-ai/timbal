from ...types.field import Field, resolve_default
from ...types.file import File
from ...utils import _platform_api_call


async def upload_file(
    org_id: str = Field(description="The organization ID."),
    file: File = Field(description="The file to upload."),
):
    """
    Upload a file using multipart request to the platform endpoint.

    Args:
        org_id (str): The organization ID.
        file (File): The file to upload.

    Returns:
        dict: The response from the platform.
    """
    org_id = resolve_default("org_id", org_id)
    file = resolve_default("file", file)

    file = file if isinstance(file, File) else File.validate(file)

    file.seek(0)
    content = file.read()

    path = f"orgs/{org_id}/files"
    files = {"file": (file.name, content, file.__content_type__)}

    res = await _platform_api_call("POST", path, files=files)
    return res.json()
