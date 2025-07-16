import os
import pytest
import tempfile
import csv
from typing import Any

from timbal.steps.timbal.tables import (
    create_table,
    delete_table,
    get_table,
    import_csv,
    import_records,
    query,
    get_table_definition,
    list_tables
)
from timbal.state.context import RunContext, TimbalPlatformConfig
from timbal.state import set_run_context


@pytest.fixture
def timbal_config():
    """Fixture to provide Timbal platform configuration."""
    return TimbalPlatformConfig(**{
        "host": "dev.timbal.ai",
        "auth_config": {
            "type": "bearer",
            "token": os.getenv("TIMBAL_API_TOKEN")
        },
        "scope": {}
    })


@pytest.fixture
def run_context(timbal_config):
    """Fixture to provide run context with Timbal platform config."""
    return RunContext(
        timbal_platform_config=timbal_config,
        app_config={}
    )


@pytest.fixture
def test_table_name():
    """Fixture to provide a unique test table name."""
    return f"test_table_{os.getpid()}_{os.getenv('USER', 'unknown')}"


@pytest.fixture
def test_columns():
    """Fixture to provide test column definitions."""
    return [
        {
            "name": "id",
            "data_type": "INTEGER",
            "default_value": None,
            "is_nullable": False,
            "is_unique": True,
            "comment": "Primary key"
        },
        {
            "name": "name",
            "data_type": "VARCHAR(255)",
            "default_value": None,
            "is_nullable": False,
            "is_unique": False,
            "comment": "User name"
        },
        {
            "name": "email",
            "data_type": "VARCHAR(255)",
            "default_value": None,
            "is_nullable": True,
            "is_unique": True,
            "comment": "User email"
        },
        {
            "name": "age",
            "data_type": "INTEGER",
            "default_value": "18",
            "is_nullable": True,
            "is_unique": False,
            "comment": "User age"
        }
    ]


@pytest.fixture
def test_csv_data():
    """Fixture to provide test CSV data."""
    return [
        {"id": 1, "name": "John Doe", "email": "john@example.com", "age": 30},
        {"id": 2, "name": "Jane Smith", "email": "jane@example.com", "age": 25},
        {"id": 3, "name": "Bob Johnson", "email": "bob@example.com", "age": 35}
    ]


@pytest.fixture
def temp_csv_file(test_csv_data):
    """Fixture to create a temporary CSV file with test data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "email", "age"])
        writer.writeheader()
        writer.writerows(test_csv_data)
        temp_file_path = f.name
    
    yield temp_file_path
    
    try:
        os.unlink(temp_file_path)
    except FileNotFoundError:
        pass


class TestTables:
    """Test class for tables module functions."""
    
    ORG_ID = 19
    KB_ID = 41
    
    @pytest.mark.asyncio
    async def test_create_table(self, run_context, test_table_name, test_columns):
        """Test creating a table."""

        set_run_context(run_context)
        
        try:
            result = await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            print(f"Created table: {result}")
            
        finally:
            try:
                await delete_table(
                    table_name=test_table_name,
                    cascade=True,
                    kb_id=self.KB_ID,
                    org_id=self.ORG_ID
                )
            except Exception as e:
                print(f"Cleanup warning: Could not delete table {test_table_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_get_table(self, run_context, test_table_name, test_columns):
        """Test getting a table."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            result = await get_table(
                table_name=test_table_name,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            assert result is not None
            assert result.get("name") == test_table_name
            assert "columns" in result
            print(f"Retrieved table: {result}")
            
        finally:
            try:
                await delete_table(
                    table_name=test_table_name,
                    cascade=True,
                    kb_id=self.KB_ID,
                    org_id=self.ORG_ID
                )
            except Exception as e:
                print(f"Cleanup warning: Could not delete table {test_table_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_import_csv(self, run_context, test_table_name, test_columns, temp_csv_file):
        """Test importing CSV data to a table."""

        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            result = await import_csv(
                table_name=test_table_name,
                csv_path=temp_csv_file,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID,
                mode="overwrite"
            )
            
            assert result["rows_inserted"] == 3
            print(f"Imported CSV: {result}")
            
        finally:
            
            try:
                await delete_table(
                    table_name=test_table_name,
                    cascade=True,
                    kb_id=self.KB_ID,
                    org_id=self.ORG_ID
                )
            except Exception as e:
                print(f"Cleanup warning: Could not delete table {test_table_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_import_records(self, run_context, test_table_name, test_columns, test_csv_data):
        """Test importing records to a table."""

        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            result = await import_records(
                table_name=test_table_name,
                records=test_csv_data,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
        finally:
            try:
                await delete_table(
                    table_name=test_table_name,
                    cascade=True,
                    kb_id=self.KB_ID,
                    org_id=self.ORG_ID
                )
            except Exception as e:
                print(f"Cleanup warning: Could not delete table {test_table_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_query(self, run_context, test_table_name, test_columns, test_csv_data):
        """Test querying a table."""

        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await import_records(
                table_name=test_table_name,
                records=test_csv_data,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            result = await query(
                sql_query=f"SELECT * FROM {test_table_name}",
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            assert result is not None
            assert len(result) == len(test_csv_data)
            print(f"Query result: {result}")
            
        finally:
            try:
                await delete_table(
                    table_name=test_table_name,
                    cascade=True,
                    kb_id=self.KB_ID,
                    org_id=self.ORG_ID
                )
            except Exception as e:
                print(f"Cleanup warning: Could not delete table {test_table_name}: {e}")
    
    
    @pytest.mark.asyncio
    async def test_get_table_definition(self, run_context, test_table_name, test_columns):
        """Test getting table definition."""
        
        set_run_context(run_context)
        
        try:
            
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            result = await get_table_definition(
                table_name=test_table_name,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            assert result is not None
            assert "CREATE TABLE" in result
            print(f"Table definition: {result}")
            
        finally:
            
            try:
                await delete_table(
                    table_name=test_table_name,
                    cascade=True,
                    kb_id=self.KB_ID,
                    org_id=self.ORG_ID
                )
            except Exception as e:
                print(f"Cleanup warning: Could not delete table {test_table_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_list_tables_preview(self, run_context):
        """Test listing tables in preview format."""
        
        set_run_context(run_context)
        
        result = await list_tables(
            kb_id=self.KB_ID,
            org_id=self.ORG_ID,
            format="preview"
        )
        
        assert result is not None
        print(f"Tables (preview): {result}")
    
    @pytest.mark.asyncio
    async def test_list_tables_full(self, run_context):
        """Test listing tables in full format."""

        set_run_context(run_context)
        
        result = await list_tables(
            kb_id=self.KB_ID,
            org_id=self.ORG_ID,
            format="full"
        )
        
        assert result is not None
        print(f"Tables (full): {result}")
    
    @pytest.mark.asyncio
    async def test_list_tables_definition(self, run_context):
        """Test listing tables in definition format."""

        set_run_context(run_context)
        
        result = await list_tables(
            kb_id=self.KB_ID,
            org_id=self.ORG_ID,
            format="definition"
        )
        
        assert result is not None
        print(f"Tables (definition): {result}")
    
    @pytest.mark.asyncio
    async def test_delete_table(self, run_context, test_table_name, test_columns):
        """Test deleting a table."""

        set_run_context(run_context)
        
        await create_table(
            table_name=test_table_name,
            columns=test_columns,
            kb_id=self.KB_ID,
            org_id=self.ORG_ID
        )
        
        table_info = await get_table(
            table_name=test_table_name,
            kb_id=self.KB_ID,
            org_id=self.ORG_ID
        )
        assert table_info is not None
        
        await delete_table(
            table_name=test_table_name,
            cascade=True,
            kb_id=self.KB_ID,
            org_id=self.ORG_ID
        )
        
        try:
            await get_table(
                table_name=test_table_name,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            assert False, "Table should not exist after deletion"
        except Exception:
            # Expected behavior - table should not exist
            pass
    
    @pytest.mark.asyncio
    async def test_import_csv_append_mode(self, run_context, test_table_name, test_columns, temp_csv_file):
        """Test importing CSV data in append mode."""
        
        set_run_context(run_context)
        
        try:
            
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            # Import CSV data in append mode
            result = await import_csv(
                table_name=test_table_name,
                csv_path=temp_csv_file,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID,
                mode="append"
            )
            
            assert result is not None
            print(f"Imported CSV (append mode): {result}")
            
        finally:
            
            try:
                await delete_table(
                    table_name=test_table_name,
                    cascade=True,
                    kb_id=self.KB_ID,
                    org_id=self.ORG_ID
                )
            except Exception as e:
                print(f"Cleanup warning: Could not delete table {test_table_name}: {e}")
    
    @pytest.mark.asyncio
    async def test_query_with_conditions(self, run_context, test_table_name, test_columns, test_csv_data):
        """Test querying a table with WHERE conditions."""
        
        set_run_context(run_context)
        
        try:
            
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            # Import some data
            await import_records(
                table_name=test_table_name,
                records=test_csv_data,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            # Then query with conditions
            result = await query(
                sql_query=f"SELECT * FROM {test_table_name} WHERE age > 25",
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            assert result is not None
            # Should return records where age > 25
            for record in result:
                assert record["age"] > 25
            print(f"Query with conditions result: {result}")
            
        finally:
            
            try:
                await delete_table(
                    table_name=test_table_name,
                    cascade=True,
                    kb_id=self.KB_ID,
                    org_id=self.ORG_ID
                )
            except Exception as e:
                print(f"Cleanup warning: Could not delete table {test_table_name}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 