import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import os
import asyncio
import httpx
import aiofiles
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import json
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Union
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import urllib.parse

# 加载 .env 文件
load_dotenv()

# 配置 API 和 Milvus 信息 (从环境变量获取)
DOUYIN_API_URL = os.getenv("DOUYIN_API_URL", "INVALID_URL")  # 使用占位符
KUAISHOU_API_URL = os.getenv("KUAISHOU_API_URL", "INVALID_URL") # 使用占位符
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
API_KEY = os.getenv("API_KEY")  # 假设您有通用的 API 密钥

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

# 初始化 Milvus 数据库 (修改后的 init_milvus 函数)
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

        # 正确的检查集合是否存在的方式
        if "user_data" in utility.list_collections():
            print("集合 'user_data' 已存在，准备删除并重新创建")
            collection = Collection("user_data")
            collection.drop()  # 如果存在，先删除

        print("创建新集合 'user_data'")
        collection = Collection("user_data", schema)  # 创建新集合
        return collection
    except Exception as e:
        print(f"初始化Milvus时出错: {e}")
        import traceback
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return None


# 调用 API 获取数据
# 调用 API 获取数据
async def fetch_data(api_url: str, keyword: str, page_or_cursor: Union[int, str], platform: str) -> Tuple[List[Dict], Optional[Union[int, str]]]:
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}" if API_KEY else "",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        
        encoded_keyword = urllib.parse.quote(keyword, safe='', encoding='utf-8')
        encoded_page_or_cursor = str(page_or_cursor)

        # 构建完整的 URL
        full_url = ""  # 初始化 full_url
        if platform == "douyin":
            full_url = f"{api_url}?keyword={encoded_keyword}&cursor={encoded_page_or_cursor}"
        elif platform == "kuaishou":
            full_url = f"{api_url}?keyword={encoded_keyword}&page={encoded_page_or_cursor}"
        else:
            raise ValueError(f"不支持的平台: {platform}")

        if not full_url:  # 检查 URL 是否为空
            raise ValueError(f"URL 构建失败: platform={platform}")
            
        print(f"请求API: {urllib.parse.unquote(full_url, encoding='utf-8')}")

        async with httpx.AsyncClient() as client:
            response = await client.get(full_url, headers=headers, timeout=10)
            response.raise_for_status()
            print(f"API 响应状态码: {response.status_code}")
            print(f"API 响应头: {dict(response.headers)}")
            
            data = response.json()
            
            if platform == "douyin":
                try:
                    print(f"抖音原始数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
                    
                    if isinstance(data, str):
                        data = json.loads(data)
                    
                    user_list = data.get("user_list", [])
                    print(f"用户列表长度: {len(user_list)}")
                    
                    users = []
                    for user in user_list:
                        if isinstance(user, dict):
                            user_info = user.get("user_info", {})
                            if user_info:
                                user_data = {
                                    "name": user_info.get("nickname", ""),
                                    "uid": str(user_info.get("uid", ""))
                                }
                                if user_data["name"] and user_data["uid"]:
                                    users.append(user_data)
                    
                    print(f"抖音获取到 {len(users)} 个用户数据")
                    if users:
                        print(f"第一个用户数据示例: {json.dumps(users[0], ensure_ascii=False)}")
                    
                    next_page_cursor = data.get("cursor")
                    print(f"下一页游标: {next_page_cursor}")
                    return users, next_page_cursor
                    
                except Exception as e:
                    print(f"处理抖音数据时出错: {str(e)}")
                    print(f"抖音原始数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
                    return [], None
            
            elif platform == "kuaishou":
                try:
                    # 检查数据格式
                    if isinstance(data, str):
                        data = json.loads(data)
                        
                    # 获取 mixFeeds 列表
                    mix_feeds = data.get("data", {}).get("mixFeeds", [])
                    
                    users = []
                    for feed in mix_feeds:
                        if isinstance(feed, dict):
                            user = feed.get("user", {})
                            if user:
                                user_data = {
                                    "name": user.get("user_name", ""),
                                    "uid": str(user.get("user_id", ""))
                                }
                                if user_data["name"] and user_data["uid"]:
                                    users.append(user_data)
                    
                    # 获取下一页的游标
                    next_page = data.get("data", {}).get("pcursor")
                    if next_page == "no_more":
                        next_page = None
                        
                    print(f"成功获取到 {len(users)} 个用户数据")
                    if users:
                        print(f"第一个用户数据示例: {json.dumps(users[0], ensure_ascii=False)}")
                    
                    return users, next_page
                    
                except Exception as e:
                    print(f"处理快手数据时出错: {str(e)}")
                    print(f"原始数据: {json.dumps(data, ensure_ascii=False, indent=2)}")
                    return [], None

    except Exception as e:
        print(f"请求出错: {e}")
        return [], None








# 数据向量化
async def vectorize_data(data_list: List[Dict]) -> List[List[float]]:
    try:
        if not data_list:  # 添加空列表检查
            return []
            
        model = SentenceTransformer('all-MiniLM-L6-v2')
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(None, model.encode, [item['name'] for item in data_list])
        
        print(f"向量化结果: {vectors[:3]}...")
        return vectors.tolist()  # 转换为普通列表
    except Exception as e:
        print(f"向量化数据时出错: {e}")
        return []



# 从文件读取关键词
async def read_keywords(platform: str) -> List[str]:
    filename = f"{platform}.txt"  # 根据平台名称构建文件名
    encodings = ['utf-8', 'gbk']  # 尝试的编码列表
    for encoding in encodings:
        try:
            async with aiofiles.open(filename, "r", encoding=encoding) as f:
                content = await f.read()
                print(f"读取文件 {filename} 内容: {content[:100]}...")  # 只打印前100个字符
                keywords = [line.strip() for line in content.split('\n') if line.strip()]
                return keywords
        except UnicodeDecodeError as e:
            print(f"尝试使用 {encoding} 编码失败: {e}")
        except FileNotFoundError:
            print(f"找不到文件: {filename}")
            return []
    print(f"文件读取失败: {filename}")
    return []






# 插入数据到 Milvus
async def insert_into_milvus(collection: Collection, data_list: List[Dict], vectors: List[List[float]], keyword: str):
    try:
        entities = [
            vectors,  # 向量数据
            [
                json.dumps({"name": item["name"], "uid": item["uid"]})
                for item in data_list
            ],  # metadata
            [keyword] * len(data_list),  # 所有数据使用同一个 keyword
        ]
        
        print(f"准备插入的数据: {json.dumps(entities[1][:3], ensure_ascii=False, indent=2)}...")  # 打印前三个 metadata
        print(f"准备插入的关键词: {keyword}")
        
        insert_result = await asyncio.to_thread(collection.insert, entities)  # 使用 asyncio.to_thread
        print(f"插入数据到 Milvus 成功, 数量: {len(data_list)}, 关键词: {keyword}")
        return insert_result
    except Exception as e:
        print(f"插入数据到 Milvus 时出错: {e}")
        return None


# 主函数
async def main():
    try:
        # 初始化 Milvus
        collection = await init_milvus()
        if collection is None:
            print("Milvus 初始化失败，退出程序。")
            return

        # 平台映射
        platform_mapping = {
            "抖音": {"file": "抖音", "api": "douyin", "url": DOUYIN_API_URL},
            "快手": {"file": "快手", "api": "kuaishou", "url": KUAISHOU_API_URL}
        }

        for platform, config in platform_mapping.items():
            # 获取关键词列表
            keywords = await read_keywords(config["file"])
            if not keywords:
                print(f"未能读取到{platform}的关键词，跳过此平台。")
                continue

            print(f"\n开始处理{platform}平台的关键词...")
            for keyword in keywords:
                print(f"\n处理关键词: {keyword}")
                
                page = 1
                next_page = "0"  # 初始化 next_page
                while True:
                    # 获取数据
                    users, next_page = await fetch_data(
                        config["url"], 
                        keyword, 
                        page if config["api"] == "kuaishou" else "0" if page == 1 else next_page,
                        config["api"]
                    )
                    
                    if not users:
                        print(f"未找到与关键词 '{keyword}' 相关的用户数据或已到达最后一页")
                        break
                        
                    # 向量化数据
                    vectors = await vectorize_data(users)
                    if not vectors:
                        print("向量化数据失败")
                        break
                        
                    # 插入数据到 Milvus
                    result = await insert_into_milvus(collection, users, vectors, keyword)
                    if result is None:
                        print("数据插入失败")
                        break
                        
                    print(f"成功处理第 {page} 页数据")
                    
                    # 检查是否继续获取下一页
                    if config["api"] == "kuaishou":
                        if next_page is None or next_page == "no_more":
                            print("已到达最后一页")
                            break
                    else:  # douyin
                        if not next_page:
                            print("已到达最后一页")
                            break
                    page += 1
                    
                    # 可选：添加延时避免请求过快
                    await asyncio.sleep(1)
                
                print(f"完成关键词 {keyword} 的数据处理")

        print("\n所有平台数据处理完成")
        
        # 创建索引
        print("开始创建索引...")
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        await asyncio.to_thread(collection.create_index, "vector", index_params)
        print("索引创建完成")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭 Milvus 连接
        connections.disconnect("default")
        print("已关闭 Milvus 连接")



if __name__ == "__main__":
    asyncio.run(main())
