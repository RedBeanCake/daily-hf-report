import requests
import json
import datetime
from bs4 import BeautifulSoup

def get_beijing_time():
    return datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=8)

def scrape_hf(mode="daily"):
    """抓取 Hugging Face Daily Papers"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
    now = get_beijing_time()
    
    if mode == "weekly":
        year, week, _ = now.isocalendar()
        target_str = f"{year}-W{week:02d}"
        url = f"https://huggingface.co/api/daily_papers?week={target_str}&limit=50"
    else:
        target_str = now.strftime('%Y-%m-%d')
        url = f"https://huggingface.co/api/daily_papers?date={target_str}&limit=50"

    def fetch(api_url):
        res = requests.get(api_url, headers=headers, timeout=15)
        return res.json() if res.status_code == 200 else []

    papers = fetch(url)
    # 日报补偿机制
    if mode == "daily" and (not papers or len(papers) < 5):
        yesterday = (now - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        papers = fetch(f"https://huggingface.co/api/daily_papers?date={yesterday}&limit=50")
    
    return papers

def process_hf_with_ai(client_llm, hf_papers, mode="daily"):
    """使用 AI 处理 HF 论文"""
    if not hf_papers: return ""
    
    simple_list = []
    for p in hf_papers:
        paper_info = p.get('paper', {})
        upvotes = p.get('upvotes') or paper_info.get('upvotes', 0)
        simple_list.append({
            "id": paper_info.get('id', ''),
            "title": paper_info.get('title', 'Unknown'),
            "upvotes": upvotes
        })
    simple_list.sort(key=lambda x: x.get('upvotes', 0), reverse=True)

    chunk_size = 10
    all_chunks_md = []
    report_type = "每周" if mode == "weekly" else "今日"
    
    global_counter = 1
    for i in range(0, len(simple_list[:30]), chunk_size):  # 最多处理前30篇
        chunk = simple_list[i : i + chunk_size]
      
        prompt = f"""你是一个 AI 大模型专家。请用平实、地道的中文对以下 Hugging Face {report_type}论文进行高信息密度的总结。像在组会上给同事分享一样，直接讲清楚论文做了什么、改了哪里、效果如何。严禁过度修饰，严禁使用炫技式的词汇。

        要求：
        1. 请从编号 {global_counter} 开始连续编号。
        2. 为每篇论文提供：中文标题翻译、研究任务、核心亮点（一句话）、深度解析（技术方案简述）、领域归类。
        3. 输出格式（Markdown）：
           ### {global_counter}. [英文标题] (中文标题翻译)
           - **社区热度**: `👍 [对应 upvotes] Upvotes`
           - **论文链接**: [点击跳转](https://arxiv.org/abs/[对应 id])
           - **研究任务**: ...
           - **核心亮点**: ...
           - **深度解析**: ...
           - **领域归类**: [...]
           ---

        待处理数据内容：
        {json.dumps(chunk)}
        """
      
        try:
            completion = client_llm.chat.completions.create(
                model="qwen3.7-plus", 
                messages=[{"role": "user", "content": prompt}]
            )
            all_chunks_md.append(completion.choices[0].message.content)
        except Exception as e:
            print(f"第 {global_counter} 批论文大模型调用失败: {e}")
            continue

        global_counter += len(chunk)

    full_content = "\n\n".join(all_chunks_md)
    return f"<details>\n<summary><b>🤗 Hugging Face {report_type}论文 (点击展开)</b></summary>\n\n{full_content}\n</details>"
