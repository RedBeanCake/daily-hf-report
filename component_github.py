from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from http_utils import RequestError, request_with_retry, retry_call


LOGGER = logging.getLogger(__name__)
GITHUB_HEADERS = {"User-Agent": "daily-hf-report/1.0"}


class AIProcessingError(RuntimeError):
    """Raised when GitHub project summarization fails."""


def scrape_github_trending(language: str = "python", *, session: Any = None) -> List[Dict[str, str]]:
    from bs4 import BeautifulSoup

    url = f"https://github.com/trending/{language}?since=daily"
    response = request_with_retry(
        "GET", url, session=session, headers=GITHUB_HEADERS, timeout=15
    )
    soup = BeautifulSoup(response.text, "html.parser")
    rows = soup.select("article.Box-row")
    if not rows:
        raise RequestError("GitHub Trending returned no repository rows")

    repos: List[Dict[str, str]] = []
    for row in rows:
        title_tag = row.select_one("h2 a")
        if not title_tag or not title_tag.get("href"):
            LOGGER.warning("Skipping malformed GitHub Trending row")
            continue
        name = " ".join(title_tag.get_text(" ", strip=True).split()).replace(" / ", "/")
        desc = row.select_one("p")
        stats = row.select_one("div.f6")
        repos.append(
            {
                "name": name,
                "link": "https://github.com" + title_tag["href"],
                "description": desc.get_text(" ", strip=True) if desc else "No description",
                "stars_info": stats.get_text(" ", strip=True) if stats else "Unknown",
            }
        )
    if not repos:
        raise RequestError("GitHub Trending rows could not be parsed")
    return repos[:15]


def process_github_with_ai(client_llm: Any, repos: List[Dict[str, str]]) -> str:
    if not repos:
        return ""
    prompt = f"""你是一个开源项目专家。请解析以下 GitHub 今日趋势项目。
只能依据输入中的名称、链接、描述和统计信息作答；描述不足时请明确写“仓库描述未提供”，不要根据项目名猜测功能。
要求：用中文总结项目用途、目标人群、技术核心，并判断是否与 AI/大模型相关。
格式：
### [项目名](链接)
- **今日趋势**: [stars/forks信息]
- **项目简介**: [一句话总结]
- **核心价值**: [基于描述的解析]
---
待处理数据：{json.dumps(repos, ensure_ascii=False)}
"""

    def call() -> str:
        completion = client_llm.chat.completions.create(
            model="qwen3.7-plus",
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        return content

    try:
        content = retry_call(call, attempts=3, base_delay=1.0)
    except Exception as exc:
        LOGGER.exception("GitHub summary failed: %s", exc)
        raise AIProcessingError("GitHub project summary failed") from exc
    return f"<details>\n<summary><b>🚀 GitHub 今日热门项目 (点击展开)</b></summary>\n\n{content}\n</details>"
