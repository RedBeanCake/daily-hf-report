import os
import argparse
from openai import OpenAI
from component_hf import scrape_hf, process_hf_with_ai, get_beijing_time
from component_github import scrape_github_trending, process_github_with_ai
import re
import requests

# --- 1. 初始化配置 ---
client_llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")
GITHUB_PAGES_URL = f"https://{os.getenv('GITHUB_REPOSITORY_OWNER')}.github.io/{os.getenv('GITHUB_REPOSITORY').split('/')[-1]}/"

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
    new_link = f"<p><a href='{filename}'>{page_title}</a></p>\n"
    old_content = open(index_file, "r", encoding="utf-8").read() if os.path.exists(index_file) else "<h1>📚 全球 AI 技术雷达</h1>"
    
    # 避免重复并插入最新
    clean_content = re.sub(rf"<p><a href='[^']*'>{re.escape(page_title)}</a></p>\n?", "", old_content)
    updated_index = clean_content.replace("</h1>", f"</h1>\n{new_link}")
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
        gh_data = scrape_github_trending()
        gh_md = process_github_with_ai(client_llm, gh_data)

    # 汇总生成
    full_md = hf_md + "\n\n" + gh_md
    if hf_data or gh_data:
        generate_page(full_md, mode=args.mode)
    # 2. 分批处理（建议每批 10-12 篇，保证 AI 输出详尽）
    chunk_size = 10
    all_chunks_md = []
    global_counter = 1

    report_type = "每周热门" if mode == "weekly" else "今日热门"
    
    for i in range(0, len(simple_list), chunk_size):
        chunk = simple_list[i : i + chunk_size]
        
        prompt = f"""你是一个 AI 大模型专家。请为以下 Hugging Face {report_type}论文提供深度中文解析。
        要求：
        1. **不要剔除**任何论文，全部保留并翻译。
        2. 请从编号 {global_counter} 开始连续编号。
        3. 为每篇论文提供：中文标题翻译、核心亮点（一句话）、深度解析（技术方案简述）、领域归类。
        4. 输出格式（Markdown）：
           ### {global_counter}. [英文标题] (中文标题翻译)
           - **社区热度**: `👍 [对应 upvotes] Upvotes`
           - **论文链接**: [点击跳转](https://arxiv.org/abs/[对应 id])
           - **核心亮点**: ...
           - **深度解析**: ...
           - **领域归类**: [...]
           ---

        待处理数据内容：
        {json.dumps(chunk)}
        """

        try:
            completion = client_llm.chat.completions.create(
                model="qwen-flash", 
                messages=[{"role": "user", "content": prompt}]
            )
            all_chunks_md.append(completion.choices[0].message.content)
        except Exception as e:
            all_chunks_md.append(f"\n> ⚠️ 批次 {global_counter}-{global_counter+len(chunk)-1} AI 解析失败: {e}\n")
        
        global_counter += len(chunk)

    # 3. 汇总所有批次内容并封装进折叠框
    full_content = "\n\n".join(all_chunks_md)
    summary_title = "Weekly Community Choice" if mode == "weekly" else "Daily Community Choice"
    
    hf_md = f"<details>\n<summary><b>🤗 Hugging Face {summary_title} (点击展开详情)</b></summary>\n\n"
    hf_md += f"## 🤗 Hugging Face {summary_title}\n\n"
    hf_md += full_content
    hf_md += "\n</details>"
    return hf_md

def generate_page(content, mode="daily"):
    """生成网页并增量更新索引"""
    now = get_beijing_time()

    if mode == "weekly":
        year, week, _ = now.isocalendar()
        date_label = f"{year}-W{week:02d}"
        page_title = f"AI 技术周报 - {date_label}"
        filename = f"archive/{date_label.replace('-', '_')}.html"
    else:
        date_label = now.strftime('%Y-%m-%d')
        page_title = f"前沿科研情报 - {date_label}"
        filename = f"archive/{date_label.replace('-', '_')}.html"
    
    os.makedirs('archive', exist_ok=True)

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
            details {{ background: #f6f8fa; padding: 15px; border-radius: 8px; border: 1px solid #d0d7de; }}
            summary {{ cursor: pointer; color: #0969da; font-weight: bold; }}
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
    
    # 写入当天的详情页
    with open(filename, "w", encoding="utf-8") as f: 
        f.write(html_template)
    
    index_file = "index.html"
    new_link = f"<p><a href='{filename}'>{page_title}</a></p>\n"
    
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            old_content = f.read()
        
        # 该正则会匹配包含相同标题的整个 <p> 标签块，并将其替换为空字符串
        pattern = rf"<p><a href='[^']*'>{re.escape(page_title)}</a></p>\n?"
        clean_content = re.sub(pattern, "", old_content)
        
        # 在第一个 </h1> 标签后插入新链接，确保最新的始终在最上面
        if "</h1>" in clean_content:
            parts = clean_content.split("</h1>", 1)
            updated_index = f"{parts[0]}</h1>\n{new_link}{parts[1].lstrip()}"
        else:
            updated_index = new_link + clean_content
    else:
        # 如果 index.html 不存在，创建初始内容
        updated_index = f"<h1>📚 全球 AI 技术雷达 - 历史索引</h1>\n{new_link}"

    with open(index_file, "w", encoding="utf-8") as f:
        f.write(updated_index)

    # 飞书推送逻辑
    if FEISHU_WEBHOOK:
        try:
            requests.post(FEISHU_WEBHOOK, json={
                "msg_type": "interactive",
                "card": {
                    "header": {"title": {"tag": "plain_text", "content": f"🌟 {page_title}"}, "template": "orange" if mode=="daily" else "blue"},
                    "elements": [
                        {"tag": "div", "text": {"tag": "lark_md", "content": f"{'今日' if mode=='daily' else '本周'}已自动汇总社区最受关注的科研成果。"}},
                        {"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": "查看详情"}, "type": "primary", "url": GITHUB_PAGES_URL}]}
                    ]
                }
            }, timeout=10)
        except Exception as e:
            print(f"Feishu Push Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='daily', choices=['daily', 'weekly'], help='运行模式：daily(日报) 或 weekly(周报)')
    args = parser.parse_args()

    raw_papers = scrape_hf(mode=args.mode)
    if raw_papers:
        md_content = process_hf_with_ai(raw_papers, mode=args.mode)
        generate_page(md_content, mode=args.mode)
    else:
        print(f"未抓取到数据，跳过生成步骤。模式: {args.mode}")
