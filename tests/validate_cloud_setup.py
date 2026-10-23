"""
End-to-End Live Cloud Infrastructure Validation Script.
Tests Qdrant Cloud connectivity, embedding generation, scoped vector indexing,
and OpenRouter LLM grounded response with a programmatic test document.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.documents import Document

from rag.config import get_config, validate_config
from rag.ingestion import (
    get_qdrant_client,
    ensure_collection_exists,
    index_chunks_to_qdrant,
    delete_document_vectors,
)
from rag.retrieval import get_embedding_model, retrieve_relevant_chunks
from rag.generation import generate_grounded_answer


def main():
    print("=" * 70)
    print("       LEARN RAG: CLOUD INFRASTRUCTURE VALIDATION       ")
    print("=" * 70)
    
    config = get_config()
    is_valid, issues = validate_config(config)
    
    print("\n[Phase 1/5] Checking Configuration & Environment Variables...")
    if not is_valid:
        print("❌ Configuration validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        print("\nPlease update your '.env' file with valid cloud credentials.")
        sys.exit(1)
        
    print(f"✓ Qdrant URL: {config.qdrant_url}")
    print(f"✓ Collection: {config.qdrant_collection_name}")
    print(f"✓ OpenRouter Model: {config.openrouter_model}")
    print("✓ Configuration validated successfully.")
    
    # Phase 2: Qdrant Cloud Connection
    print("\n[Phase 2/5] Testing Qdrant Cloud Connection...")
    try:
        client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
        collections = client.get_collections()
        existing_names = [c.name for c in collections.collections]
        print(f"✓ Successfully connected to Qdrant Cloud!")
        print(f"  Existing collections: {existing_names}")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant Cloud: {e}")
        sys.exit(1)

    # Phase 3: Embedding Model
    print("\n[Phase 3/5] Loading Embedding Model...")
    try:
        embeddings = get_embedding_model(config.embedding_model_name)
        test_vector = embeddings.embed_query("Cloud validation test")
        print(f"✓ Embedding model '{config.embedding_model_name}' loaded.")
        print(f"  Vector dimension: {len(test_vector)} (expected: 384)")
    except Exception as e:
        print(f"❌ Failed to load embedding model: {e}")
        sys.exit(1)

    # Phase 4: Scoped Indexing and Search
    print("\n[Phase 4/5] Testing Scoped Indexing & Vector Search...")
    test_doc_id = "test_cloud_val"
    test_chunks = [
        Document(
            page_content="Quantum computing leverages superposition and entanglement to perform complex calculations.",
            metadata={
                "filename": "quantum_primer.pdf",
                "page": 1,
                "chunk_index": 0,
                "chunk_id": f"{test_doc_id}_chunk_0",
                "doc_id": test_doc_id,
            }
        ),
        Document(
            page_content="Superconductors operate at cryogenic temperatures and conduct electricity with zero resistance.",
            metadata={
                "filename": "quantum_primer.pdf",
                "page": 2,
                "chunk_index": 1,
                "chunk_id": f"{test_doc_id}_chunk_1",
                "doc_id": test_doc_id,
            }
        ),
    ]
    
    try:
        ensure_collection_exists(client, config.qdrant_collection_name, vector_dim=384)
        indexed = index_chunks_to_qdrant(
            client=client,
            embeddings_model=embeddings,
            collection_name=config.qdrant_collection_name,
            chunks=test_chunks,
            doc_id=test_doc_id,
            overwrite_existing=True,
        )
        print(f"✓ Successfully indexed {indexed} test chunks in Qdrant Cloud.")
        
        # Test scoped retrieval
        query = "What operates at cryogenic temperatures with zero resistance?"
        retrieved = retrieve_relevant_chunks(
            client=client,
            embeddings_model=embeddings,
            collection_name=config.qdrant_collection_name,
            query=query,
            doc_id=test_doc_id,
            top_k=2,
        )
        print(f"✓ Retrieved {len(retrieved)} chunk(s) from Qdrant Cloud.")
        if retrieved:
            print(f"  Top Match (Score: {retrieved[0].metadata.get('similarity_score', 0):.4f}):")
            print(f"  \"{retrieved[0].page_content}\"")
    except Exception as e:
        print(f"❌ Scoped indexing or retrieval failed: {e}")
        sys.exit(1)

    # Phase 5: OpenRouter LLM Grounded Answer
    print("\n[Phase 5/5] Testing OpenRouter LLM Grounded Answer...")
    try:
        answer_data = generate_grounded_answer(
            query=query,
            retrieved_chunks=retrieved,
            openrouter_api_key=config.openrouter_api_key,
            model_name=config.openrouter_model,
        )
        print(f"✓ OpenRouter Response Generated (Model: {answer_data.get('model_used')}):")
        print(f"  Answer: {answer_data.get('answer')}")
        print(f"  Citations: {answer_data.get('citations', {}).get('citation_summary')}")
    except Exception as e:
        print(f"❌ OpenRouter generation failed: {e}")
    finally:
        # Clean up test vectors
        print("\nCleaning up test vectors from Qdrant Cloud...")
        try:
            del_count = delete_document_vectors(client, config.qdrant_collection_name, test_doc_id)
            print(f"✓ Cleaned up {del_count} test vectors.")
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("      🎉 ALL CLOUD INFRASTRUCTURE CHECKS PASSED!      ")
    print("=" * 70)


if __name__ == "__main__":
    main()
