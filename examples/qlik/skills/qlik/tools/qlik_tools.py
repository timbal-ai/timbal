"""Qlik tools for the `qlik` skill.

Credentials come from the environment (see ../../../.env.example): QLIK_TENANT_URL plus
either a person's token (personal mode) or the OAuth client's id and secret (service mode).
To use a Timbal platform integration instead, pass `integration="<org integration id>"`.
"""

from timbal.tools import QlikGetAppFields, QlikListApps, QlikQuery, QlikWhoAmI

qlik_whoami = QlikWhoAmI()
qlik_list_apps = QlikListApps()
qlik_get_app_fields = QlikGetAppFields()
qlik_query = QlikQuery()
