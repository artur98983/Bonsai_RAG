"""
Comprehensive tests for Bonsai RAG API fixes
Tests the fixes for critical errors mentioned in the issue requirements.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from rag.app.main import app

@pytest.fixture
def client():
    """Test client for FastAPI app"""
    return TestClient(app)

# Test 1: Successful indexing endpoint test
def test_index_endpoint_success(client):
    """Test successful indexing response structure"""
    with patch('rag.app.main.get_project') as mock_get_project, \
         patch('rag.app.main.index_project') as mock_index_project:
        
        project_id = 1
        mock_get_project.return_value = {
            "id": project_id,
            "name": "test_project",
            "path": "/test/path",
            "mode": "automatic"
        }
        
        mock_index_project.return_value = {
            "project_id": project_id,
            "files_scanned": 5,
            "files_indexed": 3,
            "symbols_indexed": 42,
            "references_indexed": 18,
            "vectors_upserted": 42,
            "warnings": [],
            "status": "success"
        }
        
        response = client.post(f"/projects/{project_id}/index")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "indexed"
        assert data["project_id"] == project_id
        assert data["files_scanned"] == 5
        assert data["symbols_indexed"] == 42
        print("✅ Successful indexing test passed")

# Test 2: Test that failed indexing returns failed status
def test_index_endpoint_failed_status(client):
    """Test that failed indexing returns failed status"""
    with patch('rag.app.main.get_project') as mock_get_project, \
         patch('rag.app.main.index_project') as mock_index_project:
        
        project_id = 1
        mock_get_project.return_value = {
            "id": project_id,
            "name": "test_project",
            "path": "/test/path",
            "mode": "automatic"
        }
        
        mock_index_project.return_value = {
            "project_id": project_id,
            "files_scanned": 0,
            "files_indexed": 0,
            "symbols_indexed": 0,
            "references_indexed": 0,
            "vectors_upserted": 0,
            "warnings": ["Qdrant cleanup failed: connection timeout"],
            "status": "failed"
        }
        
        response = client.post(f"/projects/{project_id}/index")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        print("✅ Failed indexing status test passed")

# Test 3: Test search endpoint (verify it exists and works)
def test_search_endpoint_exists(client):
    """Test that search endpoint exists and has correct structure"""
    project_id = 1
    
    with patch('rag.app.main.get_project') as mock_get_project, \
         patch('rag.app.main.search') as mock_search:
        
        mock_get_project.return_value = {
            "id": project_id,
            "name": "test_project",
            "path": "/test/path",
            "mode": "automatic"
        }
        
        mock_search.return_value = [
            {
                "id": 1,
                "score": 1.0,
                "project_id": project_id,
                "file": "test.py",
                "symbol": "test_function",
                "symbol_type": "function_definition",
                "language": "python",
                "start_line": 1,
                "end_line": 10,
                "text": "def test_function(): pass",
                "retrieval": "exact_symbol"
            }
        ]
        
        response = client.post(
            f"/projects/{project_id}/search",
            json={"query": "test_function", "limit": 8}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0
        print("✅ Search endpoint test passed")

# Test 4: Test chat endpoint (verify it exists and works)
def test_chat_endpoint_exists(client):
    """Test that chat endpoint exists and has correct structure"""
    project_id = 1
    
    with patch('rag.app.main.get_project') as mock_get_project, \
         patch('rag.app.main.hybrid_search') as mock_hybrid_search, \
         patch('rag.app.main.expand_graph') as mock_expand_graph, \
         patch('rag.app.main.ask_bonsai') as mock_ask_bonsai:
        
        mock_get_project.return_value = {
            "id": project_id,
            "name": "test_project",
            "path": "/test/path",
            "mode": "automatic"
        }
        
        mock_hybrid_search.return_value = []
        mock_expand_graph.return_value = []
        mock_ask_bonsai.return_value = {
            "content": "Test answer",
            "reasoning": "Test reasoning",
            "raw": {}
        }
        
        response = client.post(
            "/chat",
            json={
                "project": project_id,
                "mode": "automatic",
                "message": "Test message",
                "context_limit": 8
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "reasoning" in data
        assert "context" in data
        print("✅ Chat endpoint test passed")

# Test 5: Test statistics structure
def test_statistics_structure():
    """Test that statistics dictionary has all required fields"""
    # Mock the dependencies to test the structure
    with patch('rag.app.indexer.ensure_collection') as mock_ensure_collection, \
         patch('rag.app.indexer.delete_project_points') as mock_delete_points, \
         patch('rag.app.indexer.os.walk') as mock_walk, \
         patch('rag.app.indexer.Path') as mock_path, \
         patch('rag.app.indexer.get_language') as mock_get_language, \
         patch('rag.app.indexer.extract_code_symbols') as mock_extract_symbols, \
         patch('rag.app.indexer.extract_calls') as mock_extract_calls, \
         patch('rag.app.indexer.save_structural_analysis') as mock_save_analysis:
        
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.resolve.return_value = "/test/path"
        mock_get_language.return_value = "python"
        mock_extract_symbols.return_value = []
        mock_extract_calls.return_value = []
        mock_save_analysis.return_value = {}
        mock_ensure_collection.return_value = None
        mock_delete_points.return_value = None
        
        # Import and test the function
        from rag.app.indexer import index_project
        result = index_project(1, "/test/path")
        
        # Verify result structure
        assert isinstance(result, dict)
        required_fields = ["project_id", "files_scanned", "files_indexed", 
                          "symbols_indexed", "references_indexed", 
                          "vectors_upserted", "warnings", "status"]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
        
        # Verify status is valid
        assert result["status"] in ["success", "partial", "failed"]
        print("✅ Statistics structure test passed")

if __name__ == "__main__":
    print("Running comprehensive tests for Bonsai RAG API fixes")
    print("=" * 60)
    
    # Run tests
    test_index_endpoint_success()
    test_index_endpoint_failed_status()
    test_search_endpoint_exists()
    test_chat_endpoint_exists()
    test_statistics_structure()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nTest summary:")
    print("✅ Successful indexing endpoint test")
    print("✅ Failed indexing status test")
    print("✅ Search endpoint existence test")
    print("✅ Chat endpoint existence test")
    print("✅ Statistics structure test")
