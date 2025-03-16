import os
import asyncio
import httpx
import json
import urllib.parse
from typing import List, Dict, Tuple, Optional
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, MilvusException
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm  # 添加到文件开头的导入部分

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
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        print("Milvus连接成功")

        # 检查是否存在旧集合
        if "user_data" in utility.list_collections():
            print("检测到现有集合，继续使用...")
            collection = Collection("user_data")
            return collection

        # 如果不存在，创建新集合
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="keyword", dtype=DataType.VARCHAR, max_length=100),
        ]
        schema = CollectionSchema(fields, "用户数据集合")
        
        print("创建新集合 'user_data'")
        collection = Collection("user_data", schema)
        
        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
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
            if platform == "快手":
                encoded_keyword = urllib.parse.quote(params["keyword"])
                api_url = f"{api_url}?keyword={encoded_keyword}&page={params['page']}"
                response = await client.get(api_url, headers=headers, timeout=60)
                data = response.json()
                
                # 添加调试输出
                print(f"\n快手API返回数据结构: {json.dumps(data, ensure_ascii=False)[:200]}...")
                
                if data.get("data", {}).get("mixFeeds"):
                    mix_feeds = data.get("data", {}).get("mixFeeds", [])
                    users = []
                    for feed in mix_feeds:
                        if isinstance(feed, dict):
                            user = feed.get("user", {})
                            if user:
                                user_data = {
                                    "name": user.get("user_name", ""),
                                    "uid": str(user.get("user_id", "")),
                                    "description": user.get("user_text", ""),
                                    "following": 0,
                                    "followers": user.get("fansCount", 0)
                                }
                                if user_data["name"] and user_data["uid"]:
                                    users.append(user_data)
                    
                    # 添加调试输出
                    if not users:
                        print(f"未能从数据中提取到用户信息，原始数据结构: {json.dumps(mix_feeds[:1], ensure_ascii=False)}")
                    
                    next_cursor = str(int(params["page"]) + 1)
                    next_cursor = "" if not users else next_cursor
                    return users, next_cursor
                else:
                    print(f"快手API返回数据结构不符合预期: {json.dumps(data.get('data', {}), ensure_ascii=False)[:200]}...")
                    return [], ""
            else:
                response = await client.get(api_url, headers=headers, params=params, timeout=60)
                data = response.json()
                # 移除详细的数据打印
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
        print(f"\n开始处理 {platform} 平台数据...")
        with open(filename, 'r', encoding='utf-8') as f:
            keywords = [line.strip() for line in f if line.strip()]
        
        if not keywords:
            print(f"警告: {filename} 文件内容为空")
            return
            
        print(f"读取到 {len(keywords)} 个关键词")
        
        total_inserted = 0
        for keyword in tqdm(keywords, desc=f"{platform}关键词处理"):
            cursor = "0"
            keyword_total = 0
            
            while cursor:
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
                            keyword_total += len(mr.primary_keys)
                            total_inserted += len(mr.primary_keys)
                        except Exception as e:
                            print(f"\n插入数据时出错 ({platform}, {keyword}): {str(e)[:100]}...")
                
            if keyword_total > 0:
                print(f"\n√ {keyword}: 已插入 {keyword_total} 条数据")
                
        print(f"\n✓ {platform}平台处理完成，共插入 {total_inserted} 条数据")

    except Exception as e:
        print(f"\n处理 {platform} 数据时出错: {str(e)[:100]}...")

async def main():
    print("正在初始化系统...")
    collection = await init_milvus()
    if not collection:
        return

    # 先显示现有数据统计
    collection.load()
    total = collection.num_entities
    print(f"\n当前数据库统计:")
    print(f"总数据量: {total} 条")

    # 先处理快手平台
    print("\n=== 第一阶段：处理快手平台数据 ===")
    await process_platform(collection, "快手", KUAISHOU_API_URL, "快手.txt")

    # 再处理抖音平台
    print("\n=== 第二阶段：处理抖音平台数据 ===")
    await process_platform(collection, "抖音", DOUYIN_API_URL, "抖音.txt")

    # 显示最终统计
    print("\n=== 最终数据统计 ===")
    collection.load()
    final_total = collection.num_entities
    print(f"√ 总计采集数据: {final_total} 条")
    
    if final_total > 0:
        print("\n数据样例:")
        results = collection.query(expr="", output_fields=["metadata", "keyword"], limit=2)
        for i, result in enumerate(results, 1):
            metadata = json.loads(result["metadata"])
            print(f"  {i}. {metadata.get('name')} ({result['keyword']})")

    # 查看每个平台的数据量
    results = collection.query(
        expr="",
        output_fields=["keyword"],
        limit=total
    )
    
    douyin_count = 0
    kuaishou_count = 0
    for result in results:
        if "抖音" in result["keyword"]:
            douyin_count += 1
        elif "快手" in result["keyword"]:
            kuaishou_count += 1
    
    print(f"抖音数据: {douyin_count} 条")
    print(f"快手数据: {kuaishou_count} 条")

    # 只处理快手平台的数据
    platforms = [
        # ("抖音", DOUYIN_API_URL, "抖音.txt"),  # 暂时注释掉抖音平台
        ("快手", KUAISHOU_API_URL, "快手.txt")
    ]
    
    for platform, url, filename in platforms:
        await process_platform(collection, platform, url, filename)

    # 显示最终统计
    print("\n数据采集完成:")
    collection.load()
    total = collection.num_entities
    print(f"√ 总计采集数据: {total} 条")
    
    # 简化的数据验证
    if total > 0:
        print("\n数据样例:")
        results = collection.query(expr="", output_fields=["metadata", "keyword"], limit=2)
        for i, result in enumerate(results, 1):
            metadata = json.loads(result["metadata"])
            print(f"  {i}. {metadata.get('name')} ({result['keyword']})")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序出错: {str(e)[:100]}...")
    finally:
        print("\n程序结束")
