import os
import asyncio
import httpx
import aiofiles

from tikhub import Client
from dotenv import load_dotenv

# 加载 .env 文件 | Load .env file
load_dotenv()

# 从环境变量中获取 API_KEY | Get API_KEY from environment variables
api_key = os.getenv("API_KEY")

if api_key is None or api_key == "your_private_api_key":
    # 创建一个默认的 .env 文件 | Create a default .env file
    with open(".env", "w") as file:
        file.write("API_KEY=your_private_api_key")
    raise ValueError("API_KEY is not set in .env file")

# 初始化 TikHub 客户端 | Initialize TikHub client
client = Client(api_key=api_key)


# 下载视频函数 | Download video function
async def download_file(aweme_id: str, play_addr: str, output_dir: str = "downloads"):
    # 创建下载目录 | Create download directory
    os.makedirs(output_dir, exist_ok=True)
    # 请求文件 | Request file
    async with httpx.AsyncClient() as http_client:
        response = await http_client.get(play_addr)
        # 文件名 | File name
        # $.data.aweme_detail.aweme_id
        file_name = os.path.join(output_dir, aweme_id + ".mp4")
        # 写入文件 | Write file
        async with aiofiles.open(file_name, "wb") as file:
            await file.write(response.content)
        return file_name


# 获取主页视频信息 | Get profile videos info
async def get_profile_videos_info(sec_user_id: str):
    async def fetch_videos(max_cursor: int):
        print(f"Fetching videos with max_cursor: {max_cursor}")
        # 执行API请求 | Perform API request
        response = await client.DouyinAppV3.fetch_user_post_videos(sec_user_id, max_cursor=max_cursor, count=20)
        # 提取并返回需要的信息 | Extract and return the required information
        return response["data"]["aweme_list"], response["data"]["has_more"], response["data"]["max_cursor"]

    # 创建一个空列表来存储所有视频信息 | Create an empty list to store all videos info
    all_videos_info = []
    has_more = True
    max_cursor = 0

    # 循环获取视频信息直到没有更多 | Loop to get video info until there are no more
    while has_more:
        videos_info, has_more, max_cursor = await fetch_videos(max_cursor)
        all_videos_info.extend(videos_info)
        print(f"Total videos fetched: {len(all_videos_info)}")

    return all_videos_info


def get_video_play_address(video_info):
    """
    Tries to retrieve the video play address from the available formats.
    If 265 format is not found, it falls back to h264.
    """
    for format_key in ["play_addr_265", "play_addr_h264"]:
        try:
            return video_info["video"][format_key]["url_list"][0]
        except (KeyError, IndexError):
            continue
    raise ValueError("No valid video URL found, this post is not a video, or the video has been deleted, or its an album.")


if __name__ == "__main__":
    # 主页链接 | Profile URL
    profile_url = "https://www.douyin.com/user/MS4wLjABAAAAH6qtuglSMr7givzADiJu6mr2S4ufCtRvIGvV1O1T85uqlCNX4SVct8TWIs8BU2x6"
    sec_user_id = profile_url.split("/")[-1]
    # 获取所有视频信息 | Get all videos info
    all_videos_info = asyncio.run(get_profile_videos_info(sec_user_id))

    # 下载所有视频 | Download all videos
    for video_info in all_videos_info:
        # 从响应中获取视频链接 | Get video URL from response
        aweme_id = video_info["aweme_id"]
        try:
            # Try to get the video play address
            play_addr = get_video_play_address(video_info)
        except Exception as e:
            # 如果报错大概是因为作品不是视频而是图集，或视频已被删除
            # If error, probably because the work is not a video but a set of pictures, or the video has been deleted
            print(f"Error retrieving video info: {e}")
            print(f"Skipping video: {aweme_id}")
            continue
        # 下载视频 | Download video
        file_name = asyncio.run(download_file(aweme_id, play_addr))
        print(f"Video downloaded: {file_name}")
