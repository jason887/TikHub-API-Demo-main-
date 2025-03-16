import os
import asyncio
import httpx
from dotenv import load_dotenv

async def test_kuaishou_api():
    # 确保从正确的路径加载 .env 文件
    load_dotenv(dotenv_path="e:\\TikHub-API-Demo-main\\.env")
    
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("KUAISHOU_API_URL")
    
    # 打印调试信息
    print(f"API Key: {api_key[:10]}...") # 只显示前10个字符，确保安全
    print(f"API URL: {api_url}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "apikey": api_key,
        "Content-Type": "application/json"
    }
    params = {"keyword": "测试"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(api_url, headers=headers, params=params)
            print(f"状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
        except Exception as e:
            print(f"错误: {e}")

if __name__ == "__main__":
    asyncio.run(test_kuaishou_api())