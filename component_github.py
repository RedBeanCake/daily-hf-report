import requests
from bs4 import BeautifulSoup
import json

def scrape_github_trending(language="python"):
    """抓取 GitHub Trending 榜单"""
    url = f"https://github.com/trending/{language}?since=daily"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        res = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.text, 'html.parser')
        repos = []
        
        for row in soup.select('article.Box-row'):
            title_tag = row.select_one('h2 a')
            name = title_tag.get_text(strip=True).replace(' / ', '/')
            link = "https://github.com" + title_tag['href']
            desc = row.select_one('p')
            desc_text = desc.get_text(strip=True) if desc else "No description"
            stars = row.select_one('div.f6').get_text(strip=True)
            
            repos.append({
                "name": name,
                "link": link,
                "description": desc_text,
                "stars_info": stars
            })
        return repos[:15]  # 取前 15 个
    except Exception as e:
        print(f"GitHub Scrape Error: {e}")
        return []

def process_github_with_ai(client_llm, repos):
    """使用 AI 处理 GitHub 项目"""
    if not repos: return ""
    
    prompt = f"""你是一个开源项目专家。请解析以下 GitHub 今日趋势项目。
    要求：用中文总结项目的用途、目标人群及技术核心，并根据描述判断其是否与 AI/大模型相关。
    格式：
    ### [项目名](链接)
    - **今日趋势**: [stars/forks信息]
    - **项目简介**: [一句话总结项目做什么]
    - **核心价值**: [深入解析该项目解决了什么痛点]
    ---
    待处理数据：{json.dumps(repos)}
    """
    
    try:
        completion = client_llm.chat.completions.create(
            model="qwen3.7-plus", 
            messages=[{"role": "user", "content": prompt}]
        )
        content = completion.choices[0].message.content
        return f"<details>\n<summary><b>🚀 GitHub 今日热门项目 (点击展开)</b></summary>\n\n{content}\n</details>"
    except:
        return ""
