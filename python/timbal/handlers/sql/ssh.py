from pydantic import BaseModel, ConfigDict
from sshtunnel import SSHTunnelForwarder


class SSHConfig(BaseModel):
    # Allow for extra params in case users want to further customize the tunnel.
    # (SSHTunnelForwarder accepts more params)
    model_config = ConfigDict(extra="allow")

    host: str
    """Hostname or IP address of the SSH server you want to connect to."""
    port: int = 22
    """The port number on which the SSH server is listening (default is usually 22)."""
    username: str
    """The username to use when logging in to the SSH server."""
    password: str | None = None
    """The password for the SSH user (optional if using a private key)."""
    pkey: str | None = None
    """The path to the private key file used for authentication (optional if using a password)."""
    remote_host: str
    """The hostname or IP address of the remote service (e.g., database) as seen from the SSH server."""
    remote_port: int
    """The port number of the remote service (e.g., database) as seen from the SSH server."""
    

def connect_ssh_tunnel(ssh_config: SSHConfig):
    """Establish an SSH tunnel to a remote server.
    Bear in mind this function does not handle closing the tunnel. 
    The caller should be responsible for closing the tunnel when it's no longer needed.
    """
    tunnel = SSHTunnelForwarder(
        (ssh_config.host, ssh_config.port),
        ssh_username=ssh_config.username,
        ssh_password=ssh_config.password,
        ssh_pkey=ssh_config.pkey,
        remote_bind_address=(ssh_config.remote_host, ssh_config.remote_port)
    )
    tunnel.start()
    return tunnel
        