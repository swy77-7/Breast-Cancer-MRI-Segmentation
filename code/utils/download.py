import synapseclient 
import synapseutils 
import os
import requests
from requests.exceptions import RequestException

# 设置代理
PROXY_HOST = "127.0.0.1"
PROXY_PORT = "7897"

def test_connection():
    """测试网络连接"""
    try:
        proxies = {
            "http": f"http://{PROXY_HOST}:{PROXY_PORT}",
            "https": f"http://{PROXY_HOST}:{PROXY_PORT}"
        }
        response = requests.get(
            "https://repo-prod.prod.sagebase.org/repo/v1/version",
            proxies=proxies,
            timeout=10
        )
        print(f"连接测试状态码: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"连接测试失败: {e}")
        return False

def main():
    # 设置自定义下载路径
    download_dir = r"E:\MAMA_MIA\data\Image"
    # 确保下载目录存在
    os.makedirs(download_dir, exist_ok=True)
    print(f"下载路径设置为: {download_dir}")

    
    # 设置代理
    os.environ['HTTP_PROXY'] = f'http://{PROXY_HOST}:{PROXY_PORT}'
    os.environ['HTTPS_PROXY'] = f'http://{PROXY_HOST}:{PROXY_PORT}'

    # 测试连接
    if not test_connection():
        print("无法连接到 Synapse 服务器，请检查网络设置")
        return

    try:
        # 初始化 Synapse 客户端
        syn = synapseclient.Synapse(debug=True)  # 启用调试模式
        
        # 使用新的 token 登录（请替换为你的新 token）
        auth_token = "eyJ0eXAiOiJKV1QiLCJraWQiOiJXN05OOldMSlQ6SjVSSzpMN1RMOlQ3TDc6M1ZYNjpKRU9VOjY0NFI6VTNJWDo1S1oyOjdaQ0s6RlBUSCIsImFsZyI6IlJTMjU2In0.eyJhY2Nlc3MiOnsic2NvcGUiOlsidmlldyIsImRvd25sb2FkIiwibW9kaWZ5Il0sIm9pZGNfY2xhaW1zIjp7fX0sInRva2VuX3R5cGUiOiJQRVJTT05BTF9BQ0NFU1NfVE9LRU4iLCJpc3MiOiJodHRwczovL3JlcG8tcHJvZC5wcm9kLnNhZ2ViYXNlLm9yZy9hdXRoL3YxIiwiYXVkIjoiMCIsIm5iZiI6MTc0NTY3MjUyOSwiaWF0IjoxNzQ1NjcyNTI5LCJqdGkiOiIxOTU2OSIsInN1YiI6IjM1Mzk3ODMifQ.U64Y_gwCko-tBxcJs6Ec94rfGDnCRANAgp7u4zFV5gWt9WJg3S5u7id_Vvi-LYFlZvF20hrwZ3KaSRKICkuYo36AdE6FSgFi6YdrQ1TonRB8dWYmH2290etPxF-0eIXxPGvkosuBMgsymvDnIANkpr5U8Wm4pTWG7V_rAOmPVFFfokW9QGFldOHzPsmJLtWoNbhhAoiQ5aH7nr8ok758xMlzR_Xp03v9m-U62Gaqbn6fSdsyFtBAUPEXqp_8dHsRDAGEyzvuJ6ZMFWKO0x9gdjQpTvbXtZc80KfeSv-m8LZ1Rn_lWvVvUDodA4vvK84IeibR_OSy5dYMzZJQc33Vyg"  # 替换为新生成的 token
        syn.login(authToken=auth_token)
        
        # 下载文件
        files = synapseutils.syncFromSynapse(syn, 'syn64871114',  path=download_dir)
        print(f"下载成功！文件列表：{files}")
        print(f"\n文件已下载到: {download_dir}")
        
        # 在资源管理器中打开下载目录
        os.system(f'explorer "{download_dir}"')
    except synapseclient.core.exceptions.SynapseAuthenticationError as e:
        print(f"认证错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    main() 