import subprocess
import os

def is_git_repository(path):
    """检查指定路径是否是一个Git仓库"""
    if not os.path.isdir(path):
        return False
    
    # 检查.git目录是否存在
    git_dir = os.path.join(path, '.git')
    if os.path.isdir(git_dir):
        return True
    
    # 进一步通过git命令确认
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--is-inside-work-tree'],
            cwd=path,
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip() == 'true'
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def git_clone(repo_url, target_path):
    """
    克隆Git仓库到指定路径，如果仓库已存在则不执行克隆
    
    参数:
        repo_url (str): 仓库的URL
        target_path (str): 目标路径
    
    返回:
        bool: 成功（包括仓库已存在的情况）返回True，失败返回False
    """
    # 检查仓库是否已存在
    if os.path.exists(target_path) and is_git_repository(target_path):
        print(f"仓库已存在: {target_path}")
        return True
    
    try:
        # 确保目标路径的父目录存在
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # 执行git clone命令
        result = subprocess.run(
            ['git', 'clone', repo_url, target_path],
            check=True,
            capture_output=True,
            text=True
        )
        
        print(f"仓库克隆成功: {repo_url} -> {target_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"克隆失败: {e.stderr}")
        return False
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return False

# 使用示例
if __name__ == "__main__":
    repository_url = "https://github.com/example/repository.git"  # 替换为实际仓库URL
    destination_path = "/path/to/your/target/directory"          # 替换为目标路径
    
    git_clone(repository_url, destination_path)
