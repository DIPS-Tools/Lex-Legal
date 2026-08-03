import asyncio
from typing import Optional

from elasticsearch import AsyncElasticsearch
from elasticsearch import NotFoundError, RequestError

from app.config.es_config import ESConfig, es_config


class ESClientManager:
    """
    Elasticsearch 异步客户端管理器。

    主要职责：
    1. 根据配置生成 Elasticsearch 地址；
    2. 初始化 AsyncElasticsearch 客户端；
    3. 对外提供 Elasticsearch 客户端；
    4. 在应用退出时关闭连接。

    在真实项目中，通常会在 FastAPI / Flask / Django 应用启动时调用 init()，
    在应用关闭时调用 close()。
    """

    def __init__(self, _es_config: ESConfig):
        """
        初始化客户端管理器。

        Args:
            _es_config:
                Elasticsearch 配置对象。

                例如：
                host = "localhost"
                port = 9200
        """
        self.es_config = _es_config

        # 初始状态下还没有创建 Elasticsearch 客户端。
        self.client: Optional[AsyncElasticsearch] = None

    def _get_url(self) -> str:
        """
        根据配置生成 Elasticsearch 服务地址。

        Returns:
            Elasticsearch URL，例如：
            http://localhost:9200
        """
        endpoint = f"{self.es_config.endpoint}"
        # print("es endpoint:", endpoint)
        return endpoint

    def init(self) -> None:
        """
        初始化 Elasticsearch 异步客户端。

        注意：
        - 这个方法本身不需要 async；
        - 创建 AsyncElasticsearch 对象时不会立即发送请求；
        - 真正的网络请求会在 ping、search、index 等操作时发生。
        """
        if self.client is not None:
            # 防止重复初始化客户端。
            return

        self.client = AsyncElasticsearch(
            hosts=[self._get_url()],

            # 单个请求的超时时间，单位为秒。
            request_timeout=30,

            # 当请求超时时，自动重试。
            retry_on_timeout=True,

            # 最大重试次数。
            max_retries=3,
        )

    def get_client(self) -> AsyncElasticsearch:
        """
        获取已经初始化的 Elasticsearch 客户端。

        Returns:
            AsyncElasticsearch 客户端实例。

        Raises:
            RuntimeError:
                如果调用 get_client() 前没有先调用 init()。
        """
        if self.client is None:
            raise RuntimeError(
                "Elasticsearch client has not been initialized. "
                "Please call es_client_manager.init() first."
            )

        return self.client

    async def close(self) -> None:
        """
        关闭 Elasticsearch 客户端连接。

        应用退出时必须关闭客户端，
        否则可能出现未关闭连接或 Event Loop 警告。
        """
        if self.client is not None:
            await self.client.close()
            self.client = None


# ============================================================
# 创建全局 Elasticsearch 客户端管理器
# ============================================================
#
# app_config.es 是项目中的 Elasticsearch 配置对象。
#
# 这里使用全局实例，使整个项目可以共享同一个客户端连接池，
# 避免每次请求都重新创建 Elasticsearch 客户端。
#
es_client_manager = ESClientManager(es_config)

# ----- test-----:

async def create_books_index(
    client: AsyncElasticsearch,
    index_name: str,
) -> None:
    """
    创建图书索引。

    如果索引已经存在，则不会重复创建。

    Args:
        client:
            Elasticsearch 异步客户端。

        index_name:
            要创建的索引名称。
    """

    # 检查索引是否已经存在。
    index_exists = await client.indices.exists(index=index_name)

    if index_exists:
        print(f"Index '{index_name}' already exists.")
        return

    # 创建 Elasticsearch 索引。
    await client.indices.create(
        index=index_name,

        # mappings 用于定义索引中每个字段的数据类型。
        mappings={
            # dynamic=False 表示：
            # 如果插入的数据中包含 mappings 里没有定义的字段，
            # Elasticsearch 不会自动把这个字段加入 mapping。
            #
            # 但是该字段仍可能保存在 _source 中。
            "dynamic": False,

            "properties": {
                # text 类型用于全文搜索。
                #
                # Elasticsearch 会对 text 字段进行分词。
                # 例如：
                # "Brave New World"
                #
                # 可能被拆分成：
                # brave、new、world
                "name": {
                    "type": "text"
                },

                # 作者字段也支持全文搜索。
                "author": {
                    "type": "text"
                },

                # date 类型用于保存日期。
                #
                # format 指定输入日期必须符合 yyyy-MM-dd 格式。
                # 例如：
                # 2000-03-15
                "release_date": {
                    "type": "date",
                    "format": "yyyy-MM-dd"
                },

                # integer 类型用于保存整数。
                "page_count": {
                    "type": "integer"
                }
            }
        },
    )

    print(f"Index '{index_name}' created successfully.")


async def insert_books(
    client: AsyncElasticsearch,
    index_name: str,
) -> None:
    """
    使用 Bulk API 批量插入图书数据。

    Bulk API 比逐条调用 client.index() 更高效，
    特别适合一次插入大量数据。

    Bulk API 的 operations 格式是：

    [
        操作描述,
        文档内容,
        操作描述,
        文档内容,
        ...
    ]

    Args:
        client:
            Elasticsearch 异步客户端。

        index_name:
            Elasticsearch 索引名称。
    """

    books = [
        {
            "_id": "book-001",
            "name": "Revelation Space",
            "author": "Alastair Reynolds",
            "release_date": "2000-03-15",
            "page_count": 585,
        },
        {
            "_id": "book-002",
            "name": "1984",
            "author": "George Orwell",

            # 注意：
            # George Orwell 的《1984》通常出版于 1949 年。
            # 这里使用 1949-06-08 作为示例日期。
            "release_date": "1949-06-08",
            "page_count": 328,
        },
        {
            "_id": "book-003",
            "name": "Fahrenheit 451",
            "author": "Ray Bradbury",
            "release_date": "1953-10-15",
            "page_count": 227,
        },
        {
            "_id": "book-004",
            "name": "Brave New World",
            "author": "Aldous Huxley",
            "release_date": "1932-06-01",
            "page_count": 268,
        },
        {
            "_id": "book-005",
            "name": "The Handmaid's Tale",
            "author": "Margaret Atwood",
            "release_date": "1985-06-01",
            "page_count": 311,
        },
    ]

    operations = []

    for book in books:
        # 从字典中取出 Elasticsearch 文档 ID。
        #
        # 使用固定的 _id 有两个优点：
        # 1. 可以方便地更新指定文档；
        # 2. 重复执行程序时不会创建重复记录。
        document_id = book["_id"]

        # 复制数据，并移除内部使用的 _id 字段。
        document = {
            key: value
            for key, value in book.items()
            if key != "_id"
        }

        # Bulk API 的第一行是操作信息。
        operations.append(
            {
                "index": {
                    "_index": index_name,
                    "_id": document_id,
                }
            }
        )

        # Bulk API 的第二行是要写入的文档数据。
        operations.append(document)

    response = await client.bulk(
        operations=operations,

        # refresh="wait_for" 表示等待索引刷新，
        # 确保 bulk 插入完成后，数据可以立即被搜索到。
        #
        # 开发测试时非常方便。
        # 大规模生产环境批量导入时，不建议每次都强制等待刷新。
        refresh="wait_for",
    )

    # Bulk API 即使部分文档插入失败，也可能返回正常响应。
    # 因此需要检查 errors 字段。
    if response.get("errors"):
        print("Some documents failed to index:")

        for item in response["items"]:
            index_result = item.get("index", {})

            if "error" in index_result:
                print(
                    f"Document ID: {index_result.get('_id')}, "
                    f"Error: {index_result['error']}"
                )

        raise RuntimeError("Some books failed to index.")

    print(f"{len(books)} books indexed successfully.")


async def search_books(
    client: AsyncElasticsearch,
    index_name: str,
    keyword: str,
) -> None:
    """
    根据书名执行全文搜索。

    Args:
        client:
            Elasticsearch 异步客户端。

        index_name:
            Elasticsearch 索引名称。

        keyword:
            搜索关键词，例如：
            brave
    """

    response = await client.search(
        index=index_name,

        # query 定义 Elasticsearch 查询条件。
        query={
            # match 查询适合 text 类型字段。
            #
            # Elasticsearch 会对输入内容进行分词，
            # 然后根据相关性返回结果。
            "match": {
                "name": keyword
            }
        },

        # 最多返回 10 条结果。
        size=10,
    )

    # 搜索结果保存在 hits.hits 中。
    hits = response["hits"]["hits"]

    print("\n================ Search Result ================")
    print(f"Search keyword: {keyword}")
    print(f"Total matched: {response['hits']['total']['value']}")
    print("================================================")

    if not hits:
        print("No matching books found.")
        return

    for hit in hits:
        # _source 是原始文档内容。
        source = hit["_source"]

        # _score 是 Elasticsearch 计算出的相关性分数。
        score = hit["_score"]

        print(f"\nDocument ID: {hit['_id']}")
        print(f"Score: {score}")
        print(f"Name: {source.get('name')}")
        print(f"Author: {source.get('author')}")
        print(f"Release date: {source.get('release_date')}")
        print(f"Page count: {source.get('page_count')}")


async def delete_books_index(
    client: AsyncElasticsearch,
    index_name: str,
) -> None:
    """
    删除图书索引。

    这个函数主要用于测试和清理数据。

    注意：
    删除索引会同时删除索引中的所有文档，
    生产环境中应谨慎使用。
    """

    index_exists = await client.indices.exists(index=index_name)

    if not index_exists:
        print(f"Index '{index_name}' does not exist.")
        return

    await client.indices.delete(index=index_name)

    print(f"Index '{index_name}' deleted successfully.")

async def show_all_books(
    client: AsyncElasticsearch,
    index_name: str,
) -> None:
    """
    Retrieve and display all documents from the specified index.

    Note:
        Elasticsearch returns only 10 results by default.
        Here, size=100 is used for a small demonstration dataset.
    """

    response = await client.search(
        index=index_name,
        size=100,
        query={
            "match_all": {}
        },
        sort=[
            {
                "release_date": {
                    "order": "asc"
                }
            }
        ],
    )

    hits = response["hits"]["hits"]

    print("\n================ All Books ================")
    print(f"Total documents: {response['hits']['total']['value']}")

    for hit in hits:
        source = hit["_source"]

        print("-------------------------------------------")
        print(f"Document ID: {hit['_id']}")
        print(f"Name: {source.get('name')}")
        print(f"Author: {source.get('author')}")
        print(f"Release date: {source.get('release_date')}")
        print(f"Page count: {source.get('page_count')}")


async def test() -> None:
    """
    Elasticsearch 完整测试流程。

    流程：
    1. 获取客户端；
    2. 检查 Elasticsearch 是否可用；
    3. 创建索引；
    4. 批量插入数据；
    5. 搜索数据；
    6. 关闭客户端。
    """

    index_name = "my-books3"

    # 初始化 Elasticsearch 客户端。
    es_client_manager.init()

    # 获取已经初始化的客户端。
    client = es_client_manager.get_client()

    try:
        # ping() 用于检查 Elasticsearch 服务是否可连接。
        connected = await client.ping()

        if not connected:
            raise ConnectionError(
                f"Cannot connect to Elasticsearch: "
                f"{es_client_manager._get_url()}"
            )

        print(
            "Connected to Elasticsearch successfully: "
            f"{es_client_manager._get_url()}"
        )

        # 获取 Elasticsearch 服务信息。
        info = await client.info()

        print(
            "Elasticsearch version: "
            f"{info['version']['number']}"
        )

        # 创建索引。
        await create_books_index(
            client=client,
            index_name=index_name,
        )

        # 插入测试数据。
        await insert_books(
            client=client,
            index_name=index_name,
        )

        await show_all_books(
            client=client,
            index_name="my-books",
        )

        # 搜索书名中包含 brave 的文档。
        await search_books(
            client=client,
            index_name=index_name,
            keyword="brave",
        )

        # 如需测试删除索引，可以取消下面代码的注释。
        #
        # await delete_books_index(
        #     client=client,
        #     index_name=index_name,
        # )

    except NotFoundError as exc:
        # 当索引或文档不存在时可能进入这里。
        print(f"Elasticsearch resource was not found: {exc}")

    except RequestError as exc:
        # 当 mapping、查询语法或请求参数错误时可能进入这里。
        print(f"Elasticsearch request error: {exc}")

    except ConnectionError as exc:
        # Elasticsearch 服务不可连接。
        print(f"Connection error: {exc}")

    except Exception as exc:
        # 捕获其他未预期异常。
        print(
            f"Unexpected error: "
            f"{type(exc).__name__}: {exc}"
        )

    finally:
        # 无论程序成功还是失败，都关闭客户端连接。
        await es_client_manager.close()

        print("Elasticsearch client closed.")


if __name__ == "__main__":
    # asyncio.run() 会创建事件循环，
    # 执行 test()，然后关闭事件循环。
    asyncio.run(test())