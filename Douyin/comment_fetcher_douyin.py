import os
import asyncio
import httpx
from dotenv import load_dotenv
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer

# 加载环境变量
load_dotenv()
API_KEY = os.getenv("API_KEY")

# Milvus 配置
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
model = SentenceTransformer('all-MiniLM-L6-v2')

def init_milvus():
    """初始化 Milvus 连接和集合"""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection_name = "douyin_comments"
    
    try:
        collection = Collection(name=collection_name)
        print(f"集合已存在: {collection_name}")
    except Exception:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="comment_id", dtype=DataType.INT64),
            FieldSchema(name="photo_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="author_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="author_id", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="content_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="time", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="likes", dtype=DataType.INT64),
            FieldSchema(name="area", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="is_reply", dtype=DataType.BOOL),
            FieldSchema(name="video_author_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="video_author_name", dtype=DataType.VARCHAR, max_length=200)
        ]
        schema = CollectionSchema(fields, description="抖音视频评论集合")
        collection = Collection(name=collection_name, schema=schema)
        
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="content_vector", index_params=index_params)
        print(f"创建集合: {collection_name}")
    
    collection.load()
    return collection

async def save_to_milvus(collection, comments, photo_id, video_author_id, video_author_name):
    """保存评论到 Milvus"""
    try:
        print(f"\n开始处理评论数据，共 {len(comments)} 条评论")
        
        def process_and_insert_comment(comment, is_reply=False):
            try:
                content = comment.get("text", "")
                content_vector = model.encode(content).tolist()
                
                entity = [{
                    "comment_id": int(comment.get("cid", 0)),
                    "photo_id": str(photo_id),
                    "author_name": str(comment.get("user", {}).get("nickname", "")),
                    "author_id": int(comment.get("user", {}).get("uid", 0)),
                    "content": str(content),
                    "content_vector": content_vector,
                    "time": str(comment.get("create_time", "")),
                    "likes": int(comment.get("digg_count", 0)),
                    "area": str(comment.get("user", {}).get("region", "")),
                    "is_reply": is_reply,
                    "video_author_id": str(video_author_id),
                    "video_author_name": str(video_author_name)
                }]
                
                print(f"\n处理评论: {content[:30]}...")
                mr = collection.insert(entity)
                print(f"评论已保存，插入结果：{mr}")
                
            except Exception as e:
                print(f"处理单条评论时出错: {str(e)}")
                print(f"评论原始数据: {comment}")
        
        # 处理评论
        total_count = 0
        for comment in comments:
            process_and_insert_comment(comment)
            total_count += 1
            
            # 处理回复评论
            if "reply_comment" in comment and comment["reply_comment"]:
                process_and_insert_comment(comment["reply_comment"], True)
                total_count += 1
        
        # 确保数据写入
        collection.flush()
        print(f"\n总共保存了 {total_count} 条评论到 Milvus")
            
    except Exception as e:
        print(f"保存到 Milvus 时出错: {str(e)}")
        print(f"错误详情: {type(e).__name__}")

async def fetch_video_comments(aweme_id: str, collection, video_author_id: str, video_author_name: str, cursor: str = "0"):
    """获取指定视频的评论信息"""
    api_url = "https://api.tikhub.io/api/v1/douyin/app/v1/fetch_video_comments"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Referer": "https://github.com/TikHub/TikHub-API-Demo",
        "User-Agent": "TikHub-Demo"
    }
    
    params = {
        "aweme_id": aweme_id,
        "cursor": cursor
    }
    
    async with httpx.AsyncClient(verify=False) as client:
        try:
            if cursor == "0":
                print(f"\n正在获取视频 {aweme_id} 的评论...")
            else:
                print(f"\n正在获取下一页评论 (cursor: {cursor})...")
                
            response = await client.get(api_url, headers=headers, params=params, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                comments = data.get("data", {}).get("comments", [])
                
                if comments:
                    # 保存到 Milvus
                    await save_to_milvus(collection, comments, aweme_id, video_author_id, video_author_name)
                
                if not comments:
                    if cursor == "0":
                        print("没有找到评论")
                    return
                
                # 检查是否有更多评论并递归获取
                has_more = data.get("data", {}).get("has_more", False)
                next_cursor = str(int(cursor) + len(comments))
                
                if has_more:
                    await asyncio.sleep(1)  # 添加延迟避免请求过快
                    await fetch_video_comments(aweme_id, collection, video_author_id, video_author_name, next_cursor)
                else:
                    print("\n已获取全部评论")
                    
            else:
                print(f"获取评论失败: {response.text}")
                
        except Exception as e:
            print(f"获取评论时出错: {str(e)}")

async def main():
    try:
        collection = init_milvus()
        
        # 获取已存在的 photo_id 列表
        existing_videos = set()
        try:
            res = collection.query(
                expr="photo_id != ''",
                output_fields=["photo_id"]
            )
            existing_videos = set(item['photo_id'] for item in res)
            print(f"数据库中已存在 {len(existing_videos)} 个视频的评论")
        except Exception as e:
            print(f"查询现有数据时出错: {str(e)}")
        
        # 从文件中读取数据
        comment_file_path = "e:\\TikHub-API-Demo-main\\抖音作品评论.txt"
        with open(comment_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 分割行数据
                parts = line.split()
                if len(parts) >= 3:
                    aweme_id = parts[0]
                    video_author_id = parts[1]
                    video_author_name = parts[2]
                    
                    # 检查是否已存在
                    if aweme_id in existing_videos:
                        print(f"\n视频 {aweme_id} 的评论已存在，跳过")
                        continue
                    
                    print(f"\n正在处理新视频 ID: {aweme_id}")
                    print(f"视频作者 ID: {video_author_id}")
                    print(f"视频作者昵称: {video_author_name}")
                    
                    await fetch_video_comments(aweme_id, collection, video_author_id, video_author_name)
                    # 添加到已处理集合
                    existing_videos.add(aweme_id)
                else:
                    print(f"行格式错误，跳过: {line}")
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    asyncio.run(main())