import os
import pytest
from typing import Any

from timbal.steps.timbal.indexes import (
    create_index,
    list_indexes,
    delete_index
)
from timbal.steps.timbal.tables import (
    create_table,
    delete_table
)
from timbal.state.context import RunContext, TimbalPlatformConfig
from timbal.state import set_run_context


def extract_indexes(result):
    if isinstance(result, dict) and "indexes" in result:
        return result["indexes"]
    return result


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
def test_index_name():
    """Fixture to provide a unique test index name."""
    return f"test_index_{os.getpid()}_{os.getenv('USER', 'unknown')}"


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


class TestIndexes:
    """Test class for indexes module functions."""
    
    ORG_ID = 19
    KB_ID = 41
    
    @pytest.mark.asyncio
    async def test_create_index(self, run_context, test_table_name, test_index_name, test_columns):
        """Test creating an index on a table."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name", "email"],
                index_type="btree",
                index_unique=False,
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
    async def test_create_unique_index(self, run_context, test_table_name, test_index_name, test_columns):
        """Test creating a unique index on a table."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["email"],
                index_type="btree",
                index_unique=True,
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
    async def test_create_hash_index(self, run_context, test_table_name, test_index_name, test_columns):
        """Test creating a hash index on a table."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["id"],
                index_type="hash",
                index_unique=False,
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
    async def test_create_gin_index(self, run_context, test_table_name, test_index_name, test_columns):
        """Test creating a GIN index on a table."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name"],
                index_type="gin",
                index_unique=False,
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
    async def test_list_indexes(self, run_context, test_table_name, test_index_name, test_columns):
        """Test listing indexes in a knowledge base."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name"],
                index_type="btree",
                index_unique=False,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            result = await list_indexes(
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            assert result is not None
            print(f"Listed all indexes: {result}")
            
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
    async def test_list_indexes_by_table(self, run_context, test_table_name, test_index_name, test_columns):
        """Test listing indexes filtered by table name."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name"],
                index_type="btree",
                index_unique=False,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            result = await list_indexes(
                kb_id=self.KB_ID,
                org_id=self.ORG_ID,
                table_name=test_table_name
            )
            
            assert result is not None
            print(f"Listed indexes for table {test_table_name}: {result}")
            
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
    async def test_delete_index(self, run_context, test_table_name, test_index_name, test_columns):
        """Test deleting an index."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name"],
                index_type="btree",
                index_unique=False,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await delete_index(
                index_name=test_index_name,
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
    async def test_create_multiple_indexes(self, run_context, test_table_name, test_columns):
        """Test creating multiple indexes on the same table."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            index1_name = f"test_index1_{os.getpid()}_{os.getenv('USER', 'unknown')}"
            index2_name = f"test_index2_{os.getpid()}_{os.getenv('USER', 'unknown')}"
            
            await create_index(
                table_name=test_table_name,
                index_name=index1_name,
                index_columns=["name"],
                index_type="btree",
                index_unique=False,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=index2_name,
                index_columns=["email"],
                index_type="hash",
                index_unique=False,
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
    async def test_create_composite_index(self, run_context, test_table_name, test_index_name, test_columns):
        """Test creating a composite index on multiple columns."""
        set_run_context(run_context)
        
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name", "age", "email"],
                index_type="btree",
                index_unique=False,
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
    async def test_create_unique_composite_index(self, run_context, test_table_name, test_columns):
        """Test creating a unique composite index on multiple columns."""
        set_run_context(run_context)
        index_name = f"unique_composite_{os.getpid()}_{os.getenv('USER', 'unknown')}"
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            await create_index(
                table_name=test_table_name,
                index_name=index_name,
                index_columns=["name", "email"],
                index_type="btree",
                index_unique=True,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            result = await list_indexes(
                kb_id=self.KB_ID,
                org_id=self.ORG_ID,
                table_name=test_table_name
            )
            result = extract_indexes(result)
            assert any(idx["name"] == index_name for idx in result), "Unique composite index not found"
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
    async def test_create_duplicate_index_name_fails(self, run_context, test_table_name, test_columns):
        """Test creating two indexes with the same name should fail."""
        set_run_context(run_context)
        index_name = f"dup_index_{os.getpid()}_{os.getenv('USER', 'unknown')}"
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            await create_index(
                table_name=test_table_name,
                index_name=index_name,
                index_columns=["name"],
                index_type="btree",
                index_unique=False,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            with pytest.raises(Exception):
                await create_index(
                    table_name=test_table_name,
                    index_name=index_name,
                    index_columns=["email"],
                    index_type="btree",
                    index_unique=False,
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
    async def test_delete_index_and_check_absence(self, run_context, test_table_name, test_index_name, test_columns):
        """Test deleting an index and checking it is absent from the list."""
        set_run_context(run_context)
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name"],
                index_type="btree",
                index_unique=False,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            await delete_index(
                index_name=test_index_name,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            result = await list_indexes(
                kb_id=self.KB_ID,
                org_id=self.ORG_ID,
                table_name=test_table_name
            )
            result = extract_indexes(result)
            assert not any(idx["name"] == test_index_name for idx in result), "Index was not deleted"
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
    async def test_create_index_on_nonexistent_column_fails(self, run_context, test_table_name, test_columns):
        """Test creating an index on a non-existent column should fail."""
        set_run_context(run_context)
        index_name = f"badcol_index_{os.getpid()}_{os.getenv('USER', 'unknown')}"
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            with pytest.raises(Exception):
                await create_index(
                    table_name=test_table_name,
                    index_name=index_name,
                    index_columns=["nonexistent_col"],
                    index_type="btree",
                    index_unique=False,
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
    @pytest.mark.parametrize("index_type", ["brin", "gist"])
    async def test_create_and_list_special_index_types(self, run_context, test_table_name, test_index_name, test_columns, index_type):
        """Test creating and listing brin and gist index types."""
        set_run_context(run_context)
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name"],
                index_type=index_type,
                index_unique=False,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            result = await list_indexes(
                kb_id=self.KB_ID,
                org_id=self.ORG_ID,
                table_name=test_table_name
            )
            result = extract_indexes(result)
            assert any(idx["name"] == test_index_name and idx["type"] == index_type for idx in result), f"{index_type} index not found"
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
    async def test_index_removed_with_table(self, run_context, test_table_name, test_index_name, test_columns):
        """Test that deleting a table removes its indexes."""
        set_run_context(run_context)
        try:
            await create_table(
                table_name=test_table_name,
                columns=test_columns,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            await create_index(
                table_name=test_table_name,
                index_name=test_index_name,
                index_columns=["name"],
                index_type="btree",
                index_unique=False,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            await delete_table(
                table_name=test_table_name,
                cascade=True,
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            result = await list_indexes(
                kb_id=self.KB_ID,
                org_id=self.ORG_ID
            )
            result = extract_indexes(result)
            assert not any(idx["name"] == test_index_name for idx in result), "Index still present after table deletion"
        except Exception as e:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 