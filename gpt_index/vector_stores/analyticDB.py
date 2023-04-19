"""AnalyticDB vector store index.

An index that is built within AnalyticDB.

"""
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from gpt_index.data_structs.node_v2 import DocumentRelationship, Node
from gpt_index.vector_stores.types import (
    NodeEmbeddingResult,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
import io
from datetime import datetime
import psycopg2cffi as psycopg2
from psycopg2cffi.extras import DictCursor

logger = logging.getLogger(__name__)

# fastann index parameters
INDEXPARAM = {
    'distancemeasure' : 'L2',
    'dim' : 1536,
    'pq_segments' : 64,
    'hnsw_m' : 100,
    'pq_centers' : 2048
}

# The fields names that we are going to be storing within AnalyticDB, the field declaration for schema creation, and description
# WARNING: first column must be PRIMARY KEY for upsert
SCHEMA = [
    ("id",          "text",         "PRIMARY KEY"),
    ("doc_id",      "text",         ""),
    ("content",     "text",         ""),
    ("embedding",   "real[]",       ""),
    # ("source",      "text",         ""),
    # ("source_id",   "text",         ""),
    # ("url",         "text",         ""),
    # ("author",      "text",         ""),
    ("created_at",  "timestamptz",  "DEFAULT NOW()"),
]

class AnalyticDBVectorStore(VectorStore):
    """The AnalyticDB Vector Store.

    In this vector store we store the text, its embedding and
    a few pieces of its metadata in a AnalyticDB collection. This implemnetation
    allows the use of an already existing collection if it is one that was created
    this vector store. It also supports creating a new one if the collection doesnt
    exist or if `overwrite` is set to True.

    Args:
        collection_name (str, optional): The name of the collection where data will be
            stored. Defaults to "llamalection".
        index_params (dict, optional): The index parameters for AnalyticDB, if none are
            provided an HNSW index will be used. Defaults to None.
        search_params (dict, optional): The search parameters for a AnalyticDB query.
            If none are provided, default params will be generated. Defaults to None.
        dim (int, optional): The dimension of the embeddings. If it is not provided,
            collection creation will be done on first insert. Defaults to None.
        host (str, optional): The host address of AnalyticDB. Defaults to "localhost".
        port (int, optional): The port of AnalyticDB. Defaults to 19530.
        user (str, optional): The username for RBAC. Defaults to "".
        password (str, optional): The password for RBAC. Defaults to "".
        use_secure (bool, optional): Use https. Required for Zilliz Cloud.
            Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing collection with same
            name. Defaults to False.

    Raises:
        ImportError: Unable to import `pyanalyticDB`.
        AnalyticDBException: Error communicating with AnalyticDB, more can be found in logging
            under Debug.

    Returns:
        AnalyticDBVectorstore: Vectorstore that supports add, delete, and query.
    """

    stores_text: bool = True
    stores_node: bool = False

    def __init__(
        self,
        collection_name: str = "llamalection",
        index_params: Optional[dict] = None,
        search_params: Optional[dict] = None,
        dim: Optional[int] = None,
        host: str = "localhost",
        port: int = 5432,
        database: str = "postgres",
        user: str = "user",
        password: str = "password",
        use_secure: bool = False,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        self.collection_name = collection_name
        self.search_params = search_params
        self.index_params = index_params or INDEXPARAM
        self.dim = dim or self.index_params["dim"]
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.use_secure = use_secure
        self.overwrite = overwrite

        self.conn = psycopg2.connect(
            user=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        )

        with self.conn.cursor() as cur:
            self.has_collection = self._has_collection(cur)

            # If a collection already exists and we are overwriting, delete it
            if self.has_collection and self.overwrite is True:
                try:
                    self._drop_collection(cur)
                    logger.debug(
                        f"Successfully removed old collection: {self.collection_name}"
                    )
                except Exception as e:
                    logger.debug(f"Failed to remove old collection: {self.collection_name}")
                    raise e

            # If there is no collection and a dim is provided, we can create a collection
            if not self.has_collection:
                self._create_collection(cur)

            # If there is a collection and no index exists on it, create an index
            if self.has_collection and self.dim is not None:
                self._create_index(cur)

            # If using an existing index and no search params were provided,
            #   generate the correct params
            elif self.has_collection and self.search_params is not None:
                self._set_search_params(cur)

    def _has_collection(self, cur: psycopg2.extensions.cursor) -> bool:
        try:
            cur.execute(
                f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{self.collection_name}');")
            exists = cur.fetchone()[0]
            return exists
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
            return False

    def _drop_collection(self, cur: psycopg2.extensions.cursor) -> bool:
        try:
            self.has_collection = False
            cur.execute(f"DROP TABLE IF EXISTS {self.collection_name};")
            self.conn.commit()
            return True
        except Exception as e:
            print(e)
            return False

    def _create_collection(self, cur: psycopg2.extensions.cursor) -> None:
        try:
            fields = SCHEMA
            cur.execute(
                f"""
                  CREATE TABLE IF NOT EXISTS {self.collection_name} (
                    {','.join([' '.join(col) for col in fields])}
                );
                """
            )
            logger.debug(
                f"Successfully created a new collection: {self.collection_name}"
            )
            self.has_collection = True
        except Exception as e:
            logger.debug(f"Failure to create a new collection: {self.collection_name}")
            raise e

    def _create_index(self, cur: psycopg2.extensions.cursor) -> None:
        cur.execute(
            f"""
            SELECT * FROM pg_indexes WHERE tablename='{self.collection_name}';
            """
        )
        index_exists = any(
            index[2] == f"{self.collection_name}_embedding_idx"
            for index in cur.fetchall()
        )
        if not index_exists:
            try:
                cur.execute(
                    f"""
                    CREATE INDEX {self.collection_name}_embedding_idx
                    ON {self.collection_name}
                    USING ann(embedding)
                    WITH ({','.join([str(key) + '=' + str(value) for (key, value) in self.index_params.items()])});
                    """
                )

            except Exception as e:
                logger.debug(
                    f"Failed to create an index on collection: {self.collection_name}"
                )
                raise e

    def _set_search_params(self, cur: psycopg2.extensions.cursor) -> None:
        for param_name, param_value in self.index_params.items():
            cur.execute(f"SET {param_name} = '{param_value}';")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "VectorStore":
        return cls(**config_dict)

    @property
    def client(self) -> Any:
        """Get client."""
        return self.collection

    @property
    def config_dict(self) -> dict:
        """Return config dict."""
        return {
            "collection_name": self.collection_name,
            "index_params": self.index_params,
            "search_params": self.search_params,
            "dim": self.dim,
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "use_secure": self.use_secure,
            # Set to false, dont want subsequent object to rewrite store
            # "overwrite": False,
        }

    def add(self, embedding_results: List[NodeEmbeddingResult]) -> List[str]:
        """
        Takes in a dict of document_ids to list of document chunks and inserts them into the database.
        Return a list of document ids.
        """
        tasks = self._upsert_chunk(embedding_results)

        return list([chunk.ids for chunk in embedding_results])


    def _upsert_chunk(self, embedding_results: List[NodeEmbeddingResult]):
        created_at = None # use default value Now() in database
        data = [(row.id,result.doc_id,result.text,result.embedding,created_at) for row in embedding_results]

        with self.conn.cursor() as cur:
            # Construct the SQL query and data
            query = f"""
                            COPY {self.collection_name} ({','.join([col[0] for col in SCHEMA])}) FROM STDIN WITH CSV
                            DO ON CONFLICT DO UPDATE;
                    """

            str_data = "\n".join([', '.join(map(str, row)) for row in data])
            # Execute the query
            cur.copy_expert(query, io.StringIO(str_data))

            # Commit the transaction
            self.conn.commit()

    def delete(self, doc_id: str, **delete_kwargs: Any) -> None:
        """Delete a document from AnalyticDB.

        Args:
            doc_id (str): The document id to delete.

        Raises:
            AnalyticDBException: Failed to delete the doc.
        """
        if not self.has_collection:
            return

        # Adds ability for multiple doc delete in future.
        doc_ids: List[str]
        if type(doc_id) != list:
            doc_ids = [doc_id]
        else:
            doc_ids = doc_id  # type: ignore

        try:
            # Begin by querying for the primary keys to delete
            doc_ids = ['"' + entry + '"' for entry in doc_ids]
            query = f"DELETE FROM {self.collection_name} WHERE doc_id in ({','.join(doc_ids)});"
            # Execute the query
            cur.execute(query, data)
            # Commit the transaction
            self.conn.commit()
            logger.debug(f"Successfully deleted embedding with doc_id: {doc_ids}")
        except Exception as e:
            logger.debug(f"Unsuccessfully deleted embedding with doc_id: {doc_ids}")
            raise e

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes.

        Args:
            query_embedding (List[float]): query embedding
            similarity_top_k (int): top k most similar nodes
            doc_ids (Optional[List[str]]): list of doc_ids to filter by
        """

        if not self.has_collection:
            raise ValueError("AnalyticDB instance not initialized.")

        if not self.conn:
            raise ValueError(f"The connection to AnalyticDB has not been established.")

        def generate_query(query: VectorStoreQuery) -> Tuple[str, List[Any]]:
            q = f"""
                SELECT
                    {','.join([col[0] for col in SCHEMA])},
                    l2_distance(embedding,array{embedding}::real[]) AS similarity
                FROM
                    {self.collection_name}
            """
            where_clause, params = generate_where_clause(query)
            q += where_clause
            embedding = "[" + ", ".join(str(x) for x in query.query_embedding) + "]"
            q += f"ORDER BY embedding <-> array{embedding}::real[] LIMIT {query.similarity_top_k};"
            return q, params

        def generate_where_clause(query: VectorStoreQuery) -> Tuple[str, List[Any]]:
            if query.doc_ids is not None and len(query.doc_ids) != 0:
                where_clause = f"WHERE doc_id in ({','.join(['%d' for _ in expr_list])})"
                return where_clause, query.doc_ids
            return "", []

        def fetch_data(cur, q: str, params: List[Any]):
            cur.execute(q, params)
            return cur.fetchall()

        def create_results(data, query_doc_id) -> VectorStoreQueryResult:
            results = []
            for row in data:
                node = Node(
                    doc_id=row["doc_id"],
                    text=row["content"],
                    relationships={
                        DocumentRelationship.SOURCE: row["doc_id"],
                    },
                )
                results.append(node)
                similarities.append(float(row["similarity"]))
                ids.append(row["id"])
            return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

        q, params = generate_query(query)
        try:
            cur = self.conn.cursor(cursor_factory=DictCursor)
            data = fetch_data(cur, q, params)
            results = create_results(data)
            logger.debug(
                f"Successfully searched embedding in collection: {self.collection_name}"
                f" Num Results: {len(res[0])}"
            )
            return results;
        except Exception as e:
            logger.debug(
                f"Unsuccessfully searched embedding in collection: "
                f"{self.collection_name}"
            )
            raise e

