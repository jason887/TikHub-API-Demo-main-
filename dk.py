import os
import asyncio
import httpx
import json
import urllib.parse
from typing import List, Dict, Tuple, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusException
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 配置 API 和 Milvus 信息 (从环境变量获取)
DOUYIN_API_URL = os.getenv("DOUYIN_API_URL", "INVALID_URL")
KUAISHOU_API_URL = os.getenv("KUAISHOU_API_URL", "INVALID_URL")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
API_KEY = os.getenv("API_KEY")

# 检查 API_KEY 和 API URL
if not API_KEY or API_KEY == "your_private_api_key":
    with open(".env", "w") as file:
        file.write("API_KEY=your_private_api_key\n")
        file.write(f"DOUYIN_API_URL={DOUYIN_API_URL}\n")
        file.write(f"KUAISHOU_API_URL={KUAISHOU_API_URL}\n")
        file.write(f"MILVUS_HOST={MILVUS_HOST}\n")
        file.write(f"MILVUS_PORT={MILVUS_PORT}\n")
    raise ValueError(
        "API_KEY is not set in .env file.  A default .env file has been created."
    )

if DOUYIN_API_URL == "INVALID_URL" or KUAISHOU_API_URL == "INVALID_URL":
    raise ValueError("DOUYIN_API_URL or KUAISHOU_API_URL is not set correctly in .env file.")

# 初始化 SentenceTransformer 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化 Milvus 数据库
async def init_milvus() -> Optional[Collection]:
    try:
        print(f"尝试连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
        # 删除这两行，直接使用全局变量
        # MILVUS_HOST = "localhost"  # 这里错误地重新定义了局部变量
        # MILVUS_PORT = "19530"     # 这里错误地重新定义了局部变量
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("Milvus连接成功")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="keyword", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields, "用户数据集合")

        if "user_data" in utility.list_collections():
            print("集合 'user_data' 已存在，准备删除并重新创建")
            collection = Collection("user_data")
            collection.drop()

        print("创建新集合 'user_data'")
        collection = Collection("user_data", schema)

        # 创建索引 (在创建集合后立即创建)
        index_params = {
            "metric_type": "L2",  # 或 "IP" (内积)，根据你的需求选择
            "index_type": "IVF_FLAT",  # 或其他索引类型，如 HNSW, IVF_SQ8, 等
            "params": {"nlist": 1024}  # 根据你的数据量调整 nlist
        }
        print("为 vector 字段创建索引")
        collection.create_index(field_name="vector", index_params=index_params)

        return collection
    except Exception as e:
        print(f"初始化Milvus时出错: {e}")
        return None

async def fetch_data(api_url: str, keyword: str, cursor: str, platform: str) -> Tuple[List[Dict], str]:
    """从 API 获取数据"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    # 根据平台设置不同的参数
    if platform == "快手":
        params = {
            "keyword": keyword,
            "page": 1 if cursor == "0" else int(cursor)
        }
    else:  # 抖音
        params = {
            "keyword": keyword,
            "cursor": cursor
        }

    async with httpx.AsyncClient() as client:
        try:
            # 对于快手 API，确保参数正确编码
            if platform == "快手":
                encoded_keyword = urllib.parse.quote(params["keyword"])
                api_url = f"{api_url}?keyword={encoded_keyword}&page={params['page']}"
                response = await client.get(api_url, headers=headers, timeout=60)
            else:
                response = await client.get(api_url, headers=headers, params=params, timeout=60)
            print(f"请求 URL: {response.url}")
            print(f"请求参数: {params}")
            response.raise_for_status()
            data = response.json()
            print(f"{platform} API返回数据: {data}")  # 添加调试输出

            if platform == "抖音":
                # 修改数据解析逻辑
                if data.get("data", {}).get("data", {}).get("user_list"):  # 注意这里改变了路径
                    users = data.get("data", {}).get("data", {}).get("user_list", [])
                    next_cursor = str(data.get("data", {}).get("data", {}).get("cursor", ""))
                    
                    extracted_users = [
                        {
                            "name": user.get("nick_name", ""),
                            "uid": user.get("user_id", ""),
                            "description": "",
                            "following": 0,
                            "followers": user.get("fans_cnt", 0)
                        }
                        for user in users
                    ]
                    return extracted_users, next_cursor
                else:
                    print(f"抖音API返回数据为空 (实际数据结构: {data.keys()})")
                    return [], ""

            elif platform == "快手":
                if data.get("result") == 1:  # 成功
                    users = data.get("users", [])
                    next_cursor = data.get("pcursor", "")
                    
                    extracted_users = [
                        {
                            "name": user.get("user_name", ""),  # 直接从根对象获取
                            "uid": user.get("user_id", ""),
                            "description": user.get("user_text", ""),  # 改用 user_text
                            "following": 0,  # 这些字段在返回数据中没有
                            "followers": user.get("fansCount", 0)  # 使用 fansCount
                        }
                        for user in users
                    ]
                    # 检查是否还有更多数据
                    next_cursor = "" if data.get("recoPcursor") == "no_more" else next_cursor
                    return extracted_users, next_cursor
                else:
                    print(f"快手API返回错误: {data}")
                    return [], ""

            return [], ""

        except Exception as e:
            print(f"获取数据时发生错误 ({platform}, 关键词: {keyword}): {e}")
            return [], ""


async def vectorize_data(users: List[Dict]) -> List[List[float]]:
    """将用户名向量化"""
    names = [user.get('name', '') for user in users]
    try:
        vectors = model.encode(names).tolist()
        return vectors
    except Exception as e:
        print(f"向量化时出错: {e}")
        return []


async def process_platform(collection: Collection, platform: str, api_url: str, filename: str):
    try:
        print(f"正在读取文件: {filename}")
        # 首先尝试直接以二进制方式读取文件内容
        with open(filename, 'rb') as f:
            content = f.read()
            
        # 检测文件是否包含 BOM
        if content.startswith(b'\xef\xbb\xbf'):
            encoding = 'utf-8-sig'
        elif content.startswith(b'\xff\xfe') or content.startswith(b'\xfe\xff'):
            encoding = 'utf-16'
        else:
            # 依次尝试不同的编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16']
            for enc in encodings:
                try:
                    content.decode(enc)
                    encoding = enc
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("无法检测文件编码")

        # 使用检测到的编码读取文件
        with open(filename, 'r', encoding=encoding) as f:
            keywords = [line.strip() for line in f if line.strip()]
            
        if not keywords:
            raise ValueError("文件内容为空")
            
        print(f"成功使用 {encoding} 编码读取文件")
        print(f"读取到的关键词: {keywords}")
        
        for keyword in keywords:
            print(f"\n开始处理关键词 ({platform}): {keyword}")
            cursor = "0"
            while cursor:
                print(f"正在获取数据，cursor: {cursor}")
                users, cursor = await fetch_data(api_url, keyword, cursor, platform)
                if users:
                    vectors = await vectorize_data(users)
                    if vectors:
                        insert_data = [
                            vectors,
                            [json.dumps(user, ensure_ascii=False) for user in users],
                            [keyword] * len(users)
                        ]
                        try:
                            mr = collection.insert(insert_data)
                            print(f"成功插入 {len(mr.primary_keys)} 条数据 ({platform}, 关键词: {keyword})")
                        except MilvusException as e:
                            print(f"Milvus 插入错误 ({platform}, 关键词: {keyword}): {e}")
                        except Exception as e:
                            print(f"未知插入错误 ({platform}, 关键词: {keyword}): {e}")
                else:
                    print(f"没有要插入的数据 ({platform}, 关键词: {keyword})")

    except FileNotFoundError:
        print(f"文件 '{filename}' 不存在")
    except Exception as e:
        print(f"处理文件 '{filename}' 时出错: {e}")


async def main():
    print("正在初始化...")
    print(f"当前配置信息:")
    print(f"DOUYIN_API_URL: {DOUYIN_API_URL}")
    print(f"MILVUS_HOST: {MILVUS_HOST}")
    print(f"MILVUS_PORT: {MILVUS_PORT}")
    
    collection = await init_milvus()
    if not collection:
        print("Milvus 初始化失败")
        return

    # 暂时只测试抖音
    await process_platform(collection, "抖音", DOUYIN_API_URL, "抖音.txt")

    # 加载集合并验证数据
    print("\n开始验证数据:")
    collection.load()
    print(f"集合中的实体数量: {collection.num_entities}")
    
    # 验证写入的数据
    results = collection.query(expr="", output_fields=["metadata", "keyword"], limit=5)
    for i, result in enumerate(results):
        metadata = json.loads(result["metadata"])
        print(f"\n数据 {i+1}:")
        print(f"  Metadata: {metadata}")
        print(f"  Keyword: {result['keyword']}")

    print("\n所有任务完成")


if __name__ == "__main__":
    print("开始运行程序...")
    try:
        asyncio.run(main())
    except ValueError as ve:
        print(f"配置错误: {ve}")
    except Exception as e:
        print(f"程序异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("程序结束")
