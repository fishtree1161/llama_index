"""Qdrant reader."""

from typing import List, Optional, cast

from gpt_index.readers.base import BaseReader
from gpt_index.readers.schema.base import Document


class QdrantReader(BaseReader):
    """Qdrant reader.

    Retrieve documents from existing Qdrant collections.

    Args:
        location:
            If `:memory:` - use in-memory Qdrant instance.
            If `str` - use it as a `url` parameter.
            If `None` - use default values for `host` and `port`.
        url:
            either host or str of
            "Optional[scheme], host, Optional[port], Optional[prefix]".
            Default: `None`
        port: Port of the REST API interface. Default: 6333
        grpc_port: Port of the gRPC interface. Default: 6334
        prefer_grpc: If `true` - use gPRC interface whenever possible in custom methods.
        https: If `true` - use HTTPS(SSL) protocol. Default: `false`
        api_key: API key for authentication in Qdrant Cloud. Default: `None`
        prefix:
            If not `None` - add `prefix` to the REST URL path.
            Example: `service/v1` will result in
            `http://localhost:6333/service/v1/{qdrant-endpoint}` for REST API.
            Default: `None`
        timeout:
            Timeout for REST and gRPC API requests.
            Default: 5.0 seconds for REST and unlimited for gRPC
        host: Host name of Qdrant service. If url and host are None, set to 'localhost'.
            Default: `None`
    """

    def __init__(
        self,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
    ):
        """Initialize with parameters."""
        import_err_msg = (
            "`qdrant-client` package not found, please run `pip install qdrant-client`"
        )
        try:
            import qdrant_client  # noqa: F401
        except ImportError:
            raise ImportError(import_err_msg)

        self._client = qdrant_client.QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
        )

    def load_data(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
    ) -> List[Document]:
        """Load data from Qdrant.

        Args:
            collection_name (str): Name of the Qdrant collection.
            query_vector (List[float]): Query vector.
            limit (int): Number of results to return.

        Returns:
            List[Document]: A list of documents.
        """
        from qdrant_client.http.models.models import Payload

        response = self._client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            with_vectors=True,
            with_payload=True,
            limit=limit,
        )

        documents = []
        for point in response:
            payload = cast(Payload, point.payload)
            try:
                vector = cast(List[float], point.vector)
            except ValueError as e:
                raise ValueError("Could not cast vector to List[float].") from e
            document = Document(
                doc_id=payload.get("doc_id"),
                text=payload.get("text"),
                extra_info=payload.get("extra_info"),
                embedding=vector,
            )
            documents.append(document)

        return documents
