import os

import requests
from pydantic import Field

from ...errors import APIKeyNotFoundError
from ...types.file import File


# TODO Async version of this.
# This right now is intended for internal use. Use at your own risk.
def download_file(
    private_url: str = Field(
        ...,
        description="The private URL of the file to download."
    )
) -> File:
    slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not slack_bot_token:
        raise APIKeyNotFoundError("SLACK_BOT_TOKEN is not defined")

    headers = {"Authorization": f"Bearer {slack_bot_token}"}
    res = requests.get(private_url, headers=headers)

    file_bytes = res.content
    file_name = private_url.split("/")[-1]
    file_extension = "." + file_name.split(".")[-1]
    file = File.validate(file_bytes, {
        "extension": file_extension,
        "name": file_name
    })

    return file
