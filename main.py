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

def get_arxiv_abstract(paper_id):
    """根据 Arxiv ID 抓取摘要"""
    if not paper_id or not re.match(r'\d+\.\d+', paper_id):
        return ""
    url = f"https://arxiv.org/abs/{paper_id}"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            soup = BeautifulSoup(res.text, 'html.parser')
            abs_tag = soup.find('blockquote', class_='abstract')
            if abs_tag:
                # 移除 "Abstract:" 前缀并精简
                return abs_tag.text.replace('Abstract:', '').strip()[:1200]
    except Exception as e:
        print(f"Error fetching abstract for {paper_id}: {e}")
    return ""

def scrape_hf_daily():
    """抓取 HF 热门论文数据"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
    try:
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        today_str = utc_now.strftime('%Y-%m-%d')
        yesterday_str = (utc_now - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        res_today = requests.get(f"https://huggingface.co/api/daily_papers?date={today_str}&limit=100", headers=headers, timeout=15)
        papers = res_today.json() if res_today.status_code == 200 else []
        
        if not isinstance(papers, list) or len(papers) < 15:
            res_yesterday = requests.get(f"https://huggingface.co/api/daily_papers?date={yesterday_str}&limit=100", headers=headers, timeout=15)
            papers = res_yesterday.json() if res_yesterday.status_code == 200 else []
            
        return papers if isinstance(papers, list) else []
    except Exception as e:
        print(f"HF Scrape Error: {e}")
        return []

def process_hf_with_ai(hf_papers):
    """结合摘要进行 AI 总结，采用通用化命名并保留折叠框"""
    if not hf_papers: return ""
    
    # 提取信息并排序
    simple_list = []
    for p in hf_papers:
        paper_info = p.get('paper', {})
        pid = paper_info.get('id', '')
        if not pid: continue
        
        # 实时获取摘要以供总结
        print(f"Fetching abstract for {pid}...")
        abstract = get_arxiv_abstract(pid)
        time.sleep(0.5) # 避免请求过快
        
        simple_list.append({
            "id": pid,
            "title": paper_info.get('title', 'Unknown'),
            "upvotes": p.get('upvotes', 0),
            "abstract": abstract
        })
    
    simple_list.sort(key=lambda x: x['upvotes'], reverse=True)

    # 分批处理防止 Token 超限
    chunk_size = 8
    all_chunks_md = []
    global_counter = 1
    
    for i in range(0, len(simple_list), chunk_size):
        chunk = simple_list[i : i + chunk_size]
        # 根据用户偏好：要求简明扼要 [cite: 2026-02-26]
        prompt = f"""你是一个前沿 AI 研究专家。请根据提供的论文摘要进行深度且精炼的中文解析。
        要求：
        1. 必须保留所有提供的论文。
        2. 从编号 {global_counter} 开始。
        3. 请结合摘要(abstract)总结核心贡献，不要只看标题。
        4. 语言要专业且简明扼要，直接击中技术要点。
        5. 输出 Markdown 格式：
           ### 编号. [英文标题] (中文简译)
           - **热度/链接**: `👍 {global_counter} Upvotes` | [Arxiv](https://arxiv.org/abs/[id])
           - **核心突破**: (基于摘要的一句话总结)
           - **技术路径**: (简要描述方案，如：引入了XX机制，解决了XX问题)
           ---
        待处理数据：{json.dumps(chunk)}
        """
        try:
            completion = client_llm.chat.completions.create(
                model="qwen-flash", 
                messages=[{"role": "user", "content": prompt}]
            )
            all_chunks_md.append(completion.choices[0].message.content)
            global_counter += len(chunk)
        except Exception as e:
            print(f"AI Error: {e}")

    # 封装为通用命名的折叠框
    full_content = "\n\n".join(all_chunks_md)
    hf_md = "<details>\n<summary><b>🔥 社区高热度动态 (点击展开今日趋势详情)</b></summary>\n\n"
    hf_md += "## 🌐 全球科研趋势快报\n\n"
    hf_md += full_content
    hf_md += "\n</details>"
    
    return hf_md

def generate_page(content):
    """生成网页，使用更笼统的命名"""
    now = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8))
    display_date = now.strftime('%Y-%m-%d')
    # 笼统化标题
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
    
    with open(filename, "w", encoding="utf-8") as f: f.write(html_template)
    
    # 更新通用化的历史索引
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(f"<h1>📚 全球 AI 技术雷达 - 历史索引</h1><p><a href='{filename}'>{page_title}</a></p>")

    # 飞书推送使用笼统化标题
    if FEISHU_WEBHOOK:
        requests.post(FEISHU_WEBHOOK, json={
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": f"🌟 前沿科研发现 | {display_date}"}, "template": "orange"},
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": f"今日已自动汇总社区最受关注的科研成果。"}},
                    {"tag": "action", "actions": [{"tag": "button", "text": {"tag": "plain_text", "content": "查看详情"}, "type": "primary", "url": GITHUB_PAGES_URL}]}
                ]
            }
        })

if __name__ == "__main__":
    raw_papers = scrape_hf_daily()
    if raw_papers:
        md_content = process_hf_with_ai(raw_papers)
        generate_page(md_content)
