import os
import argparse
from openai import OpenAI
from component_hf import scrape_hf, process_hf_with_ai, get_beijing_time
from component_github import scrape_github_trending, process_github_with_ai
import re
import requests
import json
from datetime import datetime, timedelta

# --- 1. 初始化配置 ---
client_llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")
GITHUB_PAGES_URL = f"https://{os.getenv('GITHUB_REPOSITORY_OWNER')}.github.io/{os.getenv('GITHUB_REPOSITORY').split('/')[-1]}/"

# 加载历史记录
def load_history(file_path='processed_repos.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# 保存历史记录
def save_history(history, file_path='processed_repos.json'):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def filter_repos(repos, history, cooldown_days=7):
    new_repos = []
    now = datetime.now()
    
    for repo in repos:
        name = repo['name']
        last_date_str = history.get(name)
        
        if last_date_str:
            last_date = datetime.strptime(last_date_str, '%Y-%m-%d')
            # 如果上次总结时间在冷却期内，则跳过
            if now - last_date < timedelta(days=cooldown_days):
                continue
        
        new_repos.append(repo)
    return new_repos

def generate_page(content, mode="daily"):
    """生成详情页与索引页"""
    now = get_beijing_time()
    if mode == "weekly":
        year, week, _ = now.isocalendar()
        date_label = f"{year}-W{week:02d}"
        page_title = f"AI 技术周报 - {date_label}"
    else:
        date_label = now.strftime('%Y-%m-%d')
        page_title = f"前沿科研情报 - {date_label}"
    
    filename = f"archive/{date_label.replace('-', '_')}.html"
    os.makedirs('archive', exist_ok=True)

    # HTML 模版
    html_template = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{page_title}</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <style>
            .markdown-body {{ box-sizing: border-box; max-width: 900px; margin: 0 auto; padding: 30px; }}
            details {{ background: #f6f8fa; padding: 15px; border-radius: 8px; border: 1px solid #d0d7de; margin-bottom: 20px; }}
            summary {{ cursor: pointer; color: #0969da; font-weight: bold; font-size: 1.1em; }}
        </style>
    </head>
    <body class="markdown-body">
        <a href='../index.html'>← 返回索引</a>
        <h1>{page_title}</h1>
        <div id="content"></div>
        <script type="text/markdown" id="raw-md">{content}</script>
        <script>document.getElementById('content').innerHTML = marked.parse(document.getElementById('raw-md').textContent);</script>
    </body>
    </html>
    """

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_template)

    # 更新 index.html
    index_file = "index.html"
    # --- 修改点 1: 使用唯一的非空占位符 ---
    placeholder = "``" 
    new_link = f"<p><a href='{filename}'>{page_title}</a></p>\n"
    
    if not os.path.exists(index_file):
        old_content = f"<h1>📚 全球 AI 技术雷达 - 历史索引</h1>\n{placeholder}\n"
    else:
        with open(index_file, "r", encoding="utf-8") as f:
            old_content = f.read()
    
    # 1. 清理旧链接
    pattern = rf"<p><a href='[^']*'>{re.escape(page_title)}</a></p>\n?"
    clean_content = re.sub(pattern, "", old_content)
    
    # 2. 插入新链接
    if placeholder in clean_content:
        # --- 修改点 2: 在占位符下方插入，确保占位符本身不消失且不重复触发 ---
        updated_index = clean_content.replace(placeholder, f"{placeholder}\n{new_link}")
    else:
        # 备用逻辑
        if "</h1>" in clean_content:
            parts = clean_content.split("</h1>", 1)
            updated_index = f"{parts[0]}</h1>\n{placeholder}\n{new_link}{parts[1].lstrip()}"
        else:
            updated_index = f"{new_link}\n{placeholder}\n{clean_content}"

    with open(index_file, "w", encoding="utf-8") as f:
        f.write(updated_index)

    # 飞书推送
    if FEISHU_WEBHOOK:
        requests.post(FEISHU_WEBHOOK, json={
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": f"🚀 {page_title}"}, "template": "orange"},
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": "今日 AI 论文与开源趋势已更新。"}},
                    {"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": "阅读全文"}, "type": "primary", "url": GITHUB_PAGES_URL}]}
                ]
            }
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='daily', choices=['daily', 'weekly'])
    args = parser.parse_args()

    # 执行 Hugging Face 流程
    hf_data = scrape_hf(mode=args.mode)
    hf_md = process_hf_with_ai(client_llm, hf_data, mode=args.mode)

    # 执行 GitHub Trending 流程
    gh_md = ""
    if args.mode == "daily":
        gh_data = scrape_github_trending() #
        
        # --- 新增过滤逻辑 ---
        history = load_history()
        filtered_gh_data = filter_repos(gh_data, history)
        
        if filtered_gh_data:
            # 只把过滤后的新仓库发给 AI
            gh_md = process_github_with_ai(client_llm, filtered_gh_data)
            
            # 更新历史记录
            today_str = datetime.now().strftime('%Y-%m-%d')
            for repo in filtered_gh_data:
                history[repo['name']] = today_str
            save_history(history)
        else:
            gh_md = "> 今日无新增趋势项目（已在近期总结过）。"

    # 汇总生成
    full_md = hf_md + "\n\n" + gh_md
    if hf_data or gh_data:
        generate_page(full_md, mode=args.mode)
