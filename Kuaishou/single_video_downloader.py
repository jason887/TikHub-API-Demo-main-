import os
import asyncio
import httpx
import aiofiles
from dotenv import load_dotenv

# 加载 .env 文件 | Load .env file
load_dotenv()

# 从环境变量中获取 API_KEY | Get API_KEY from environment variables
api_key = os.getenv("API_KEY")

if not api_key or api_key == "your_private_api_key":
    # 检查并创建默认的 .env 文件 | Check and create default .env file
    async def create_default_env_file():
        async with aiofiles.open(".env", "w") as file:
            await file.write("API_KEY=your_private_api_key")
        raise ValueError("API_KEY is not set in .env file")


    asyncio.run(create_default_env_file())


# 下载视频函数 | Download video function
async def download_file(video_info: dict, play_addr: str, output_dir: str = "downloads"):
    os.makedirs(output_dir, exist_ok=True)
    # $.data[0].photoId
    file_name = os.path.join(output_dir, f"{video_info['data'][0]['photoId']}.mp4")

    # 请求文件并下载 | Request file and download
    async with httpx.AsyncClient() as http_client:
        try:
            response = await http_client.get(play_addr)
            response.raise_for_status()  # 检查响应状态 | Check response status
        except httpx.HTTPStatusError as exc:
            print(f"Error downloading video: {exc.response.status_code}")
            return None

    # 保存文件 | Save file
    async with aiofiles.open(file_name, "wb") as file:
        await file.write(response.content)

    return file_name


# 获取视频信息 | Get video info
async def get_video_info(video_url: str):
    # TikHub API Header
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Referer": "https://github.com/TikHub/TikHub-API-Demo",
        "User-Agent": "TikHub-Demo"
    }
    try:
        async with httpx.AsyncClient(timeout=30) as http_client:
            url = f"https://api.tikhub.io/api/v1/kuaishou/web/fetch_one_video?share_text={video_url}"
            response = await http_client.get(url, headers=headers)
            response.raise_for_status()
            video_info = response.json()
        # $.data[0].mainMvUrls[0].url
        play_addr = video_info["data"][0]["mainMvUrls"][0]["url"]
        return video_info, play_addr
    except KeyError as e:
        print(f"Error retrieving video info: {e}")
        return None, None


async def main(video_url: str):
    video_info, play_addr = await get_video_info(video_url)
    if not play_addr:
        return
    file_name = await download_file(video_info, play_addr)
    if file_name:
        print(f"Video downloaded: {file_name}")
    else:
        print("Failed to download video.")


if __name__ == "__main__":
    video_url = "https://www.kuaishou.com/f/X-f2k5KJpiXN1SY"
    asyncio.run(main(video_url))
