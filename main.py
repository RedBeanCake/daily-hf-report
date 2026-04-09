import requests
from bs4 import BeautifulSoup
import datetime
from openai import OpenAI
import os
import re
import json
import time

# --- 1. 核心配置 ---
client_llm = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
FEISHU_WEBHOOK = os.getenv("FEISHU_WEBHOOK")

repo_full_name = os.getenv('GITHUB_REPOSITORY', 'owner/repo')
repo_owner = os.getenv('GITHUB_REPOSITORY_OWNER', 'owner')
repo_name = repo_full_name.split('/')[-1]
GITHUB_PAGES_URL = f"https://{repo_owner}.github.io/{repo_name}/"

def scrape_hf_daily():
    """抓取 Hugging Face Daily Papers（智能处理时差与数据沉淀）"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
    try:
        # 获取当前 UTC 时间
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        today_str = utc_now.strftime('%Y-%m-%d')
        yesterday_str = (utc_now - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        # 1. 获取今天的论文
        def fetch_data(date_str):
            res = requests.get(f"https://huggingface.co/api/daily_papers?date={date_str}&limit=100", headers=headers, timeout=15)
            if res.status_code == 200:
                try:
                    return res.json()
                except Exception:
                    return []
            return []
        
        papers = fetch_data(today_str)
        # 如果今天数据不足，抓取昨天
        if not isinstance(papers, list) or len(papers) < 10:
            print(f"Fetching yesterday ({yesterday_str})...")
            papers = fetch_data(yesterday_str)
            
        return papers if isinstance(papers, list) else []
    except Exception as e:
        print(f"HF Scrape Error: {e}")
        return []

def process_hf_with_ai(hf_papers):
    """分批次调用 AI 处理 HF 论文，彻底解决篇幅限制导致的截断问题"""
    if not hf_papers or not isinstance(hf_papers, list): return ""
    
    # 1. 提取信息并预先按点赞数排序
    simple_list = []
    for p in hf_papers:
        paper_info = p.get('paper', {})
        if not paper_info or 'id' not in paper_info: continue
        upvotes_val = p.get('upvotes') or paper_info.get('upvotes', 0)
        simple_list.append({
            "id": paper_info.get('id', ''),
            "title": paper_info.get('title', 'Unknown Title'),
            "upvotes": upvotes_val
        })
    simple_list.sort(key=lambda x: x.get('upvotes', 0), reverse=True)

    # 2. 分批处理（建议每批 10-12 篇，保证 AI 输出详尽）
    chunk_size = 10
    all_chunks_md = []
    global_counter = 1
    
    for i in range(0, len(simple_list), chunk_size):
        chunk = simple_list[i : i + chunk_size]
        
        prompt = f"""你是一个 AI 大模型专家。请为以下 Hugging Face 热门论文提供深度中文解析。
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
            res_content = completion.choices[0].message.content
            all_chunks_md.append(res_content)
        except Exception as e:
            # 记录失败但不中断流程
            all_chunks_md.append(f"\n> ⚠️ 批次 {global_counter}-{global_counter+len(chunk)-1} 处理失败: {e}\n")
            
        # 更新计数器，确保下一批次编号连续
        global_counter += len(chunk)

    # 3. 汇总所有批次内容并封装进折叠框
    full_content = "\n\n".join(all_chunks_md)
    hf_md = "<details>\n<summary><b>🤗 Hugging Face Community Choice (点击展开今日全部热门详情)</b></summary>\n\n"
    hf_md += "## 🤗 Hugging Face Community Choice\n\n"
    hf_md += full_content
    hf_md += "\n</details>"
    
    return hf_md

def generate_page(content):
    """生成网页，如果 index.html 中已存在同名标题，则先覆盖（移除旧的）再添加"""
    now = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8))
    display_date = now.strftime('%Y-%m-%d')
    page_title = f"前沿科研情报 - {display_date}"
    
    os.makedirs('archive', exist_ok=True)
    filename = f"archive/{display_date.replace('-', '_')}.html"

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
        
        # --- 修改部分：使用正则移除已存在的相同标题链接 ---
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
                    "header": {"title": {"tag": "plain_text", "content": f"🌟 前沿科研发现 | {display_date}"}, "template": "orange"},
                    "elements": [
                        {"tag": "div", "text": {"tag": "lark_md", "content": f"今日已自动汇总社区最受关注的科研成果。"}},
                        {"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": "查看详情"}, "type": "primary", "url": GITHUB_PAGES_URL}]}
                    ]
                }
            }, timeout=10)
        except Exception as e:
            print(f"Feishu Push Error: {e}")

if __name__ == "__main__":
    raw_papers = scrape_hf_daily()
    if raw_papers:
        md_content = process_hf_with_ai(raw_papers)
        generate_page(md_content)
