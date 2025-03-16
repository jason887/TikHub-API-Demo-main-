from pymilvus import connections, Collection
import json

# 连接 Milvus
MILVUS_HOST = "localhost"  # 根据你的配置修改
MILVUS_PORT = "19530"

connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# 查询集合数据
def query_milvus_data():
    try:
        collection_name = "user_data"  # 你的集合名称
        collection = Collection(collection_name)

        # 加载集合到内存
        collection.load()

        # 打印集合中的数据量
        print(f"集合 '{collection_name}' 中的数据总量: {collection.num_entities}")

        # 查询前 10 条数据
        results = collection.query(expr="", output_fields=["metadata", "keyword"], limit=10)
        for i, result in enumerate(results):
            metadata = json.loads(result["metadata"])  # 将 JSON 格式的 metadata 转换为字典
            print(f"数据 {i+1}:")
            print(f"  Metadata: {metadata}")
            print(f"  Keyword: {result['keyword']}")

    except Exception as e:
        print(f"查询 Milvus 数据时出错: {e}")

# 执行查询
query_milvus_data()
