from typing import List, cast

import pytest

try:
    import qdrant_client
except ImportError:
    qdrant_client = None

from gpt_index.data_structs import Node
from gpt_index.vector_stores import QdrantVectorStore
from gpt_index.vector_stores.types import NodeEmbeddingResult, VectorStoreQuery


@pytest.fixture
def node() -> Node:
    return Node(text="lorem ipsum")


@pytest.fixture
def node_embeddings(node: Node) -> List[NodeEmbeddingResult]:
    return [
        NodeEmbeddingResult(
            id="c330d77f-90bd-4c51-9ed2-57d8d693b3b0",
            embedding=[1.0, 0.0],
            doc_id="test-0",
            node=node,
        ),
        NodeEmbeddingResult(
            id="c3d1e1dd-8fb4-4b8f-b7ea-7fa96038d39d",
            embedding=[0.0, 1.0],
            doc_id="test-1",
            node=node,
        ),
    ]


@pytest.mark.skipif(qdrant_client is None, reason="qdrant-client not installed")
def test_add_stores_data(node_embeddings: List[NodeEmbeddingResult]) -> None:
    client = qdrant_client.QdrantClient(":memory:")
    qdrant_vector_store = QdrantVectorStore(collection_name="test", client=client)

    with pytest.raises(ValueError):
        client.count("test")  # That indicates the collection does not exist

    qdrant_vector_store.add(node_embeddings)

    assert client.count("test").count == 2


@pytest.mark.skipif(qdrant_client is None, reason="qdrant-client not installed")
def test_build_query_filter_returns_none() -> None:
    client = qdrant_client.QdrantClient(":memory:")
    qdrant_vector_store = QdrantVectorStore(collection_name="test", client=client)

    query = VectorStoreQuery()
    query_filter = qdrant_vector_store._build_query_filter(query)

    assert query_filter is None


@pytest.mark.skipif(qdrant_client is None, reason="qdrant-client not installed")
def test_build_query_filter_returns_match_any() -> None:
    from qdrant_client.http.models import Filter, FieldCondition, MatchAny

    client = qdrant_client.QdrantClient(":memory:")
    qdrant_vector_store = QdrantVectorStore(collection_name="test", client=client)

    query = VectorStoreQuery(doc_ids=["1", "2", "3"])
    query_filter = cast(Filter, qdrant_vector_store._build_query_filter(query))

    assert query_filter is not None
    assert len(query_filter.must) == 1  # type: ignore[index, arg-type]
    assert isinstance(query_filter.must[0], FieldCondition)  # type: ignore[index]
    assert query_filter.must[0].key == "doc_id"  # type: ignore[index]
    assert isinstance(query_filter.must[0].match, MatchAny)  # type: ignore[index]
    assert query_filter.must[0].match.any == ["1", "2", "3"]  # type: ignore[index]


@pytest.mark.skipif(qdrant_client is None, reason="qdrant-client not installed")
def test_build_query_filter_returns_text_filter() -> None:
    from qdrant_client.http.models import Filter, FieldCondition, MatchText

    client = qdrant_client.QdrantClient(":memory:")
    qdrant_vector_store = QdrantVectorStore(collection_name="test", client=client)

    query = VectorStoreQuery(query_str="lorem")
    query_filter = cast(Filter, qdrant_vector_store._build_query_filter(query))

    assert query_filter is not None
    assert len(query_filter.must) == 1  # type: ignore[index, arg-type]
    assert isinstance(query_filter.must[0], FieldCondition)  # type: ignore[index]
    assert query_filter.must[0].key == "text"  # type: ignore[index]
    assert isinstance(query_filter.must[0].match, MatchText)  # type: ignore[index]
    assert query_filter.must[0].match.text == "lorem"  # type: ignore[index]


@pytest.mark.skipif(qdrant_client is None, reason="qdrant-client not installed")
def test_build_query_filter_returns_combined_filter() -> None:
    from qdrant_client.http.models import Filter, FieldCondition, MatchText, MatchAny

    client = qdrant_client.QdrantClient(":memory:")
    qdrant_vector_store = QdrantVectorStore(collection_name="test", client=client)

    query = VectorStoreQuery(query_str="lorem", doc_ids=["1", "2", "3"])
    query_filter = cast(Filter, qdrant_vector_store._build_query_filter(query))

    assert query_filter is not None
    assert len(query_filter.must) == 2  # type: ignore[index, arg-type]
    assert isinstance(query_filter.must[0], FieldCondition)  # type: ignore[index]
    assert query_filter.must[0].key == "doc_id"  # type: ignore[index]
    assert isinstance(query_filter.must[0].match, MatchAny)  # type: ignore[index]
    assert query_filter.must[0].match.any == ["1", "2", "3"]  # type: ignore[index]
    assert isinstance(query_filter.must[1], FieldCondition)  # type: ignore[index]
    assert query_filter.must[1].key == "text"  # type: ignore[index]
    assert isinstance(query_filter.must[1].match, MatchText)  # type: ignore[index]
    assert query_filter.must[1].match.text == "lorem"  # type: ignore[index]
