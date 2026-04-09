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
    """抓取 Hugging Face Daily Papers（增加健壮性）"""
    headers = {'User-Agent': 'Mozilla/5.0 ...'}
    try:
        utc_now = datetime.datetime.now(datetime.timezone.utc)
        today_str = utc_now.strftime('%Y-%m-%d')
        yesterday_str = (utc_now - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
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
        if not isinstance(papers, list) or len(papers) < 20:
            print(f"Fetching yesterday ({yesterday_str})...")
            papers = fetch_data(yesterday_str)
            
        return papers if isinstance(papers, list) else []
    except Exception as e:
        print(f"HF Scrape Error: {e}")
        return []

def process_hf_with_ai(hf_papers):
    """分批次处理，增加失败处理"""
    # ... 前面提取 simple_list 的逻辑不变 ...
    
    chunk_size = 10
    all_chunks_md = []
    global_counter = 1
    
    for i in range(0, len(simple_list), chunk_size):
        chunk = simple_list[i : i + chunk_size]
        # ... prompt 逻辑不变 ...
        try:
            completion = client_llm.chat.completions.create(...)
            res_content = completion.choices[0].message.content
            all_chunks_md.append(res_content)
        except Exception as e:
            # 记录失败但不中断流程
            all_chunks_md.append(f"\n> ⚠️ 批次 {global_counter}-{global_counter+len(chunk)-1} 处理失败: {e}\n")
        
        global_counter += len(chunk)

    # 汇总封装
    full_content = "\n\n".join(all_chunks_md)
    # ... 拼接 hf_md 逻辑不变 ...
    return hf_md

def generate_page(content):
    """生成网页并增量更新索引"""
    now = (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8))
    display_date = now.strftime('%Y-%m-%d')
    page_title = f"前沿科研情报 - {display_date}"
    
    os.makedirs('archive', exist_ok=True)
    filename = f"archive/{display_date.replace('-', '_')}.html"

    # 写入新页面
    with open(filename, "w", encoding="utf-8") as f: 
        f.write(html_template) # 这里的 html_template 为你原代码中的模板

    # --- 核心修改：增量更新 index.html ---
    index_file = "index.html"
    new_entry = f"<p><a href='{filename}'>{page_title}</a></p>\n"
    
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            old_content = f.read()
        
        # 将新链接插入到第一个 <p> 之前，保持最新日期在最上面
        if "<p>" in old_content:
            header, rest = old_content.split("<p>", 1)
            updated_content = f"{header}{new_entry}<p>{rest}"
        else:
            updated_content = old_content + new_entry
    else:
        updated_content = f"<h1>📚 冒烟小脑瓜简报 - 历史索引</h1>\n{new_entry}"

    with open(index_file, "w", encoding="utf-8") as f:
        f.write(updated_content)

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
