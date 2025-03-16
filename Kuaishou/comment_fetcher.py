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
    collection_name = "kuaishou_comments"
    
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
            # 添加新字段
            FieldSchema(name="video_author_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="video_author_name", dtype=DataType.VARCHAR, max_length=200)
        ]
        schema = CollectionSchema(fields, description="快手视频评论集合")
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

async def save_to_milvus(collection, comments, sub_comments_map, photo_id, video_author_id, video_author_name):
    """保存评论到 Milvus"""
    try:
        print(f"\n开始处理评论数据，共 {len(comments)} 条主评论")
        
        def process_and_insert_comment(comment, is_reply=False):
            try:
                content = comment.get("content", "")
                content_vector = model.encode(content).tolist()
                
                entity = [{
                    "comment_id": int(comment.get("comment_id", 0)),
                    "photo_id": str(photo_id),
                    "author_name": str(comment.get("author_name", "")),
                    "author_id": int(comment.get("author_id", 0)),
                    "content": str(content),
                    "content_vector": content_vector,
                    "time": str(comment.get("time", "")),
                    "likes": int(comment.get("likedCount", 0)),
                    "area": str(comment.get("authorArea", "")),
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
        
        # 处理主评论和子评论
        total_count = 0
        for comment in comments:
            process_and_insert_comment(comment)
            total_count += 1
            
            comment_id = str(comment.get("comment_id"))
            if comment_id in sub_comments_map:
                sub_comments = sub_comments_map[comment_id].get("subComments", [])
                print(f"发现 {len(sub_comments)} 条子评论")
                for sub in sub_comments:
                    process_and_insert_comment(sub, True)
                    total_count += 1
        
        # 确保数据写入
        collection.flush()
        print(f"\n总共保存了 {total_count} 条评论到 Milvus")
            
    except Exception as e:
        print(f"保存到 Milvus 时出错: {str(e)}")
        print(f"错误详情: {type(e).__name__}")

async def fetch_video_comments(photo_id: str, collection, video_author_id: str, video_author_name: str, pcursor: str = ""):
    """获取指定视频的评论信息"""
    api_url = "https://api.tikhub.io/api/v1/kuaishou/app/fetch_one_video_comment"
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Referer": "https://github.com/TikHub/TikHub-API-Demo",
        "User-Agent": "TikHub-Demo"
    }
    
    params = {
        "photo_id": photo_id,
        "pcursor": pcursor
    }
    
    async with httpx.AsyncClient(verify=False) as client:
        try:
            if not pcursor:
                print(f"\n正在获取视频 {photo_id} 的评论...")
            else:
                print(f"\n正在获取下一页评论 (pcursor: {pcursor})...")
                
            response = await client.get(api_url, headers=headers, params=params, timeout=30.0)
            
            if response.status_code == 200:
                data = response.json()
                root_comments = data.get("data", {}).get("rootComments", [])
                sub_comments_map = data.get("data", {}).get("subCommentsMap", {})
                
                if root_comments:
                    # 保存到 Milvus
                    await save_to_milvus(collection, root_comments, sub_comments_map, photo_id, video_author_id, video_author_name)
                
                if not root_comments:
                    if not pcursor:
                        print("没有找到评论")
                    return
                
                if not pcursor:
                    print("\n评论列表：")
                    
                for comment in root_comments:
                    print("\n" + "="*50)
                    print(f"用户名: {comment.get('author_name', '未知用户')}")
                    print(f"用户ID: {comment.get('author_id', '未知ID')}")
                    print(f"评论内容: {comment.get('content', '无内容')}")
                    print(f"评论时间: {comment.get('time', '未知时间')}")
                    print(f"点赞数: {comment.get('likedCount', 0)}")
                    print(f"地区: {comment.get('authorArea', '未知地区')}")
                    
                    # 获取子评论
                    comment_id = str(comment.get('comment_id'))
                    if comment_id in sub_comments_map:
                        sub_comments = sub_comments_map[comment_id].get('subComments', [])
                        if sub_comments:
                            print("\n回复：")
                            for sub in sub_comments:
                                print(f"\n  ↳ {sub.get('author_name')}: {sub.get('content')}")
                                print(f"    时间: {sub.get('time')}")
                                print(f"    点赞: {sub.get('likedCount', 0)}")
                
                # 检查是否有更多评论并递归获取
                next_cursor = data.get("data", {}).get("pcursor")
                if next_cursor and next_cursor != "no_more":
                    await asyncio.sleep(1)  # 添加延迟避免请求过快
                    await fetch_video_comments(photo_id, collection, video_author_id, video_author_name, next_cursor)
                elif pcursor:
                    print("\n已获取全部评论")
                    
            else:
                print(f"获取评论失败: {response.text}")
                
        except Exception as e:
            print(f"获取评论时出错: {str(e)}")

async def main():
    try:
        collection = init_milvus()
        
        # 获取已存在的 photo_id 列表
        existing_photos = set()
        try:
            res = collection.query(
                expr="photo_id != ''",
                output_fields=["photo_id"]
            )
            existing_photos = set(item['photo_id'] for item in res)
            print(f"数据库中已存在 {len(existing_photos)} 个视频的评论")
        except Exception as e:
            print(f"查询现有数据时出错: {str(e)}")
        
        # 从文件中读取数据
        comment_file_path = "e:\\TikHub-API-Demo-main\\快手作品评论.txt"
        with open(comment_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 分割行数据
                parts = line.split()
                if len(parts) >= 3:
                    photo_id = parts[0]
                    video_author_id = parts[1]
                    video_author_name = parts[2]
                    
                    # 检查是否已存在
                    if photo_id in existing_photos:
                        print(f"\n视频 {photo_id} 的评论已存在，跳过")
                        continue
                    
                    print(f"\n正在处理新视频 ID: {photo_id}")
                    print(f"视频作者 ID: {video_author_id}")
                    print(f"视频作者昵称: {video_author_name}")
                    
                    await fetch_video_comments(photo_id, collection, video_author_id, video_author_name)
                    # 添加到已处理集合
                    existing_photos.add(photo_id)
                else:
                    print(f"行格式错误，跳过: {line}")
    finally:
        connections.disconnect("default")

if __name__ == "__main__":
    asyncio.run(main())