#!/usr/bin/env python3
"""
Git 手动上传脚本（Windows / macOS / Linux 通用）

用法：在项目根目录运行
    python git.py

修改下方「配置区」里的 FILES_TO_UPLOAD，填入要上传的文件或文件夹路径，
然后运行本脚本即可完成 add → commit → push。不会自动监听或后台上传。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ============================================================
# 配置区 —— 按需修改
# ============================================================

# 要上传的文件/文件夹（相对项目根目录）
FILES_TO_UPLOAD = [
    "main.py",
    "data_preprocessing.py",
    "git.py",
    # "dataset_split/",   # 取消注释可上传整个文件夹
]

# 远程仓库地址（留空则使用已有 origin）
REMOTE_URL = "https://github.com/oldlinmiao/Garbage_classification.git"

# 提交说明
COMMIT_MESSAGE = "上传垃圾分类代码"

# 分支名
BRANCH = "main"

# ============================================================
# 脚本逻辑（一般无需修改）
# ============================================================

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """执行 git 命令并打印。"""
    print(f">>> {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True)


def run_ok(cmd: list[str]) -> bool:
    """执行命令，失败时不抛异常，返回是否成功。"""
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return result.returncode == 0


def ensure_git_repo() -> None:
    if not (ROOT / ".git").is_dir():
        print("初始化 Git 仓库...")
        run(["git", "init"])
        run(["git", "branch", "-M", BRANCH])


def ensure_remote() -> None:
    if not REMOTE_URL:
        if not run_ok(["git", "remote", "get-url", "origin"]):
            print("错误：未配置 REMOTE_URL，且本地没有 origin 远程。")
            sys.exit(1)
        print("使用已有远程 origin")
        return

    if run_ok(["git", "remote", "get-url", "origin"]):
        print(f"更新远程 origin → {REMOTE_URL}")
        run(["git", "remote", "set-url", "origin", REMOTE_URL])
    else:
        print(f"添加远程 origin → {REMOTE_URL}")
        run(["git", "remote", "add", "origin", REMOTE_URL])


def current_branch() -> str:
    result = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return BRANCH


def has_local_commits() -> bool:
    return run_ok(["git", "rev-parse", "HEAD"])


def sync_with_remote(branch: str) -> None:
    """若远程已有该分支，先与远程对齐（兼容远程 README 等）。"""
    if not run_ok(["git", "ls-remote", "--exit-code", "--heads", "origin", branch]):
        print(f"远程尚无 {branch} 分支，将直接推送。")
        return

    if not has_local_commits():
        print(f"本地尚无提交，从远程 {branch} 拉取基础内容...")
        run(["git", "fetch", "origin", branch])
        run(["git", "checkout", "-B", branch, f"origin/{branch}"])
        return

    print(f"远程已有 {branch} 分支，正在拉取并合并...")
    ok = run_ok(
        [
            "git",
            "pull",
            "origin",
            branch,
            "--allow-unrelated-histories",
            "--no-rebase",
        ]
    )
    if not ok:
        print("拉取失败，可能存在冲突。请手动解决后重新运行本脚本。")
        print("提示：git status 查看冲突 → 解决 → git add → git commit → 再运行 python git.py")
        sys.exit(1)


def validate_files() -> list[str]:
    """检查配置的文件是否存在。"""
    if not FILES_TO_UPLOAD:
        print("错误：FILES_TO_UPLOAD 为空，请在 git.py 顶部添加要上传的文件路径。")
        sys.exit(1)

    valid: list[str] = []
    missing: list[str] = []
    for item in FILES_TO_UPLOAD:
        path = ROOT / item
        if path.exists():
            valid.append(item.replace("\\", "/"))
        else:
            missing.append(item)

    if missing:
        print("以下路径不存在，已跳过：")
        for m in missing:
            print(f"  - {m}")

    if not valid:
        print("错误：没有可上传的有效文件。")
        sys.exit(1)

    return valid


def stage_files(paths: list[str]) -> None:
    print("添加文件到暂存区...")
    for p in paths:
        run(["git", "add", "--", p])


def has_staged_changes() -> bool:
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=ROOT,
    )
    return result.returncode != 0


def commit_and_push(branch: str) -> None:
    if has_staged_changes():
        print("提交到本地仓库...")
        run(["git", "commit", "-m", COMMIT_MESSAGE])
    else:
        print("暂存区无变更，跳过提交。")

    print(f"推送到远程 {branch} 分支...")
    try:
        run(["git", "push", "-u", "origin", branch])
    except subprocess.CalledProcessError:
        print("推送失败，请检查网络或 GitHub 权限。")
        print("提示：GitHub 需使用个人访问令牌（PAT），不能用登录密码。")
        sys.exit(1)


def main() -> None:
    print("=" * 50)
    print("Git 手动上传")
    print("=" * 50)

    paths = validate_files()
    print("将要上传：")
    for p in paths:
        print(f"  - {p}")
    print()

    ensure_git_repo()
    ensure_remote()
    branch = current_branch()
    print(f"当前分支: {branch}\n")

    sync_with_remote(branch)
    stage_files(paths)
    commit_and_push(branch)

    url = REMOTE_URL or subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    print()
    print("上传成功！")
    print(f"仓库地址：{url}")


if __name__ == "__main__":
    main()
