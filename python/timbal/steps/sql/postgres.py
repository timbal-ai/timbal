"""
Postgres Database Utilities
===============================================

This module provides basic PostgreSQL connection and query execution functionality,
with optional SSH tunneling support for secure remote access.

Current Features:
- Simple connection management using context managers
- Basic query execution (SELECT, INSERT, UPDATE, DELETE, etc.)
- Optional SSH tunneling for secure remote DB access

Planned Features (coming soon):
- Connection pooling and caching for efficient resource usage
- Convenience methods for common operations (insert, delete, update, select)
- Transaction helpers and batch operations
- Better error handling and logging

NOTE: This is an initial base implementation. The API and features will evolve!
"""

from contextlib import contextmanager

import psycopg2
from pydantic import BaseModel

from .ssh import SSHConfig, connect_ssh_tunnel


class PGConfig(BaseModel):
    """Configuration for connecting to a PostgreSQL database."""

    host: str
    port: int
    user: str
    password: str
    database: str


@contextmanager
def postgres_connection(db_config: PGConfig, ssh_config: SSHConfig | None = None):
    """
    Context manager for a PostgreSQL connection, with optional SSH tunneling.

    Args:
        db_config (PGConfig): Database connection parameters.
        ssh_config (SSHConfig, optional): SSH tunnel parameters.

    Yields:
        psycopg2.extensions.connection: Active DB connection.
    """
    tunnel = None
    conn = None
    try:
        if ssh_config:
            tunnel = connect_ssh_tunnel(ssh_config)
            host, port = tunnel.local_bind_host, tunnel.local_bind_port
        else:
            host, port = db_config.host, db_config.port
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=db_config.user,
            password=db_config.password,
            dbname=db_config.database
        )
        yield conn
    finally:
        if conn:
            conn.close()
        if tunnel:
            tunnel.stop()


def postgres_query(
    query: str,
    params: tuple | None = None,
    db_config: PGConfig | None = None,
    ssh_config: SSHConfig | None = None,
) -> list | None:
    """
    Execute a SQL query on PostgreSQL and return results (for SELECT) or commit (for INSERT/UPDATE/DELETE).

    Args:
        query (str): The SQL query to execute.
        params (tuple, optional): Query parameters.
        db_config (PGConfig): Database connection parameters.
        ssh_config (SSHConfig, optional): SSH tunnel parameters.

    Returns:
        list or None: Query results for SELECT, None otherwise.

    Note:
        This is a minimal base function. In the future, more convenience methods (insert, update, delete, transactions, etc.)
        will be added for easier and safer DB usage.
    """
    with postgres_connection(db_config, ssh_config) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:  # SELECT
                return cur.fetchall()
            else:
                conn.commit()
                return None
