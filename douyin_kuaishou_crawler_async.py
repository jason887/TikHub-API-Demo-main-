import sys
import os
import asyncio
import httpx
import json
import urllib.parse
from typing import List, Dict, Tuple, Optional, Union

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 配置 API 和 Milvus 信息 (从环境变量获取)
DOUYIN_API_URL = os.getenv("DOUYIN_API_URL", "INVALID_URL")
KUAISHOU_API_URL = os.getenv("KUAISHOU_API_URL", "INVALID_URL")  # 暂时保留，但未使用
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
API_KEY = os.getenv("API_KEY")

# 检查 API_KEY 和 API URL
if not API_KEY or API_KEY == "your_private_api_key":
    with open(".env", "w") as file:
        file.write("API_KEY=your_private_api_key\n")
        file.write(f"DOUYIN_API_URL={DOUYIN_API_URL}\n")
        file.write(f"KUAISHOU_API_URL={KUAISHOU_API_URL}\n")  # 保留
        file.write(f"MILVUS_HOST={MILVUS_HOST}\n")
        file.write(f"MILVUS_PORT={MILVUS_PORT}\n")
    raise ValueError(
        "API_KEY is not set in .env file.  A default .env file has been created."
    )

if DOUYIN_API_URL == "INVALID_URL":
    raise ValueError("DOUYIN_API_URL is not set correctly in .env file.")


# 初始化 SentenceTransformer 模型
model = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化 Milvus 数据库
async def init_milvus() -> Collection:
    try:
        print(f"尝试连接Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
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
        return collection
    except Exception as e:
        print(f"初始化Milvus时出错: {e}")
        import traceback
        traceback.print_exc()
        return None


async def vectorize_data(users: List[Dict]) -> List[List[float]]:
    """将用户数据列表转换为向量列表。"""
    try:
        names = [user["name"] for user in users]
        print(f"待向量化的用户名列表：{names}")
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, model.encode, names)
        print(f"已成功向量化 {len(vectors)} 个用户名")
        return vectors.tolist()
    except Exception as e:
        print(f"向量化数据时出错: {e}")
        return []

async def fetch_data(api_url: str, keyword: str, cursor: str) -> Tuple[List[Dict], Optional[str]]:
    """从抖音 API 获取数据。"""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}" if API_KEY else "",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        encoded_keyword = urllib.parse.quote(keyword.encode("utf-8"))
        encoded_cursor = urllib.parse.quote(cursor.encode("utf-8"))

        full_url = f"{api_url}?keyword={encoded_keyword}&cursor={encoded_cursor}"
        print(f"请求API: {urllib.parse.unquote(full_url)}")

        async with httpx.AsyncClient() as client:
            response = await client.get(full_url, headers=headers, timeout=10)
            response.raise_for_status()

            print(f"API 响应状态码: {response.status_code}")
            print(f"API 响应头: {dict(response.headers)}")

            data = response.json()

            print(f"抖音原始数据: {json.dumps(data, ensure_ascii=False, indent=2)}")

            if isinstance(data, str):  # 额外检查
                data = json.loads(data)

            users = []
            business_data_list = data.get("data", {}).get("business_data", [])
            print(f"business_data 列表长度: {len(business_data_list)}")

            for item in business_data_list:
                if (isinstance(item, dict) and item.get("type") == 1 and
                        isinstance(item.get("data"), dict)):
                    aweme_info = item["data"].get("aweme_info")
                    if isinstance(aweme_info, dict):
                        author_info = aweme_info.get("author")
                        if isinstance(author_info, dict):
                            user_data = {
                                "name": author_info.get("nickname", ""),
                                "uid": str(author_info.get("uid", "")),
                            }
                            if user_data["name"] and user_data["uid"]:
                                users.append(user_data)

            print(f"抖音获取到 {len(users)} 个用户数据")
            if users:
                print(f"第一个用户数据示例: {json.dumps(users[0], ensure_ascii=False)}")

            next_page_cursor = data.get("data", {}).get("cursor")
            print(f"下一页游标: {next_page_cursor}")
            return users, next_page_cursor

    except httpx.RequestError as e:
        print(f"请求失败: {e}")
        return [], None
    except httpx.HTTPStatusError as e:
        print(f"HTTP 错误: {e}")
        print(f"响应内容: {response.text}")  # 打印响应内容以帮助调试
        return [], None
    except Exception as e:
        print(f"处理抖音数据时出错: {str(e)}")
        print(f"抖音原始数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
        return [], None



async def main():
    collection = await init_milvus()
    if collection is None:
        return

    # 从文件中读取关键词
    try:
        with open("抖音.txt", "r", encoding="utf-8") as f:
            keywords = [line.strip() for line in f]
    except FileNotFoundError:
        print("未找到 抖音.txt 文件，请创建该文件并输入关键词。")
        return

    for keyword in keywords:
        cursor = "0"  # 初始化游标
        while cursor:
            users, next_cursor = await fetch_data(DOUYIN_API_URL, keyword, cursor)

            if users:
                vectors = await vectorize_data(users)
                metadatas = [
                    json.dumps({"uid": user["uid"], "name": user["name"]}, ensure_ascii=False)
                    for user in users
                ]
                

                keywords_list = [keyword] * len(users)  # 为每个用户添加关键词

                # 插入数据到 Milvus
                try:
                    # 确保插入的数据与 schema 匹配
                    insert_result = collection.insert([vectors, metadatas, keywords_list])
                    print(f"成功插入 {len(vectors)} 条数据到 Milvus")
                    collection.flush()  # 刷新以确保数据写入
                except Exception as e:
                    print(f"插入数据到 Milvus 时出错：{e}")


            cursor = next_cursor
            await asyncio.sleep(1)  # 避免请求过于频繁

    print("数据抓取和插入完成")
    connections.disconnect("default")
    print("已关闭 Milvus 连接")

if __name__ == "__main__":
    asyncio.run(main())
