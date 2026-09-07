from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests

from component_github import process_github_with_ai, scrape_github_trending
from component_hf import get_beijing_time, get_report_period_time, process_hf_with_ai, scrape_hf


LOGGER = logging.getLogger(__name__)
PLACEHOLDER = "<!-- ARCHIVE_LIST -->"


def normalize_repo_name(name: str) -> str:
    return re.sub(r"\s*/\s*", "/", " ".join(str(name).split()))


def load_history(file_path: str = "processed_repos.json") -> Dict[str, str]:
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("history file must contain a JSON object")
    return {normalize_repo_name(str(key)): str(value) for key, value in data.items()}


def save_history(history: Dict[str, str], file_path: str = "processed_repos.json") -> None:
    temporary_path = f"{file_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary_path, file_path)


def filter_repos(
    repos: list[Dict[str, Any]],
    history: Dict[str, str],
    cooldown_days: int = 7,
    *,
    now: Optional[datetime] = None,
) -> list[Dict[str, Any]]:
    current = (now or get_beijing_time()).date()
    new_repos = []
    for repo in repos:
        name = normalize_repo_name(str(repo["name"]))
        last_date_string = history.get(name)
        if last_date_string:
            try:
                last_date = datetime.strptime(last_date_string, "%Y-%m-%d").date()
            except ValueError:
                LOGGER.warning("Ignoring malformed history date for %s", name)
            else:
                if current - last_date < timedelta(days=cooldown_days):
                    continue
        new_repos.append(repo)
    return new_repos


def _safe_script_json(value: str) -> str:
    """Prevent a Markdown string from escaping an application/json script tag."""

    return (
        json.dumps(value, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def generate_page(
    content: str,
    mode: str = "daily",
    *,
    now: Optional[datetime] = None,
    output_dir: str = ".",
) -> str:
    current = now or get_beijing_time()
    if mode == "weekly":
        year, week, _ = get_report_period_time(mode, current).isocalendar()
        date_label = f"{year}-W{week:02d}"
        page_title = f"AI 技术周报 - {date_label}"
    else:
        date_label = current.strftime("%Y-%m-%d")
        page_title = f"前沿科研情报 - {date_label}"

    archive_dir = os.path.join(output_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    filename = os.path.join(archive_dir, f"{date_label.replace('-', '_')}.html")
    relative_filename = f"archive/{date_label.replace('-', '_')}.html"
    safe_title = html.escape(page_title, quote=True)
    safe_content = _safe_script_json(content)

    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
    <script src="https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js" defer></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js" defer></script>
    <style>
        .markdown-body {{ box-sizing: border-box; max-width: 900px; margin: 0 auto; padding: 30px; }}
        details {{ background: #f6f8fa; padding: 15px; border-radius: 8px; border: 1px solid #d0d7de; margin-bottom: 20px; }}
        summary {{ cursor: pointer; color: #0969da; font-weight: bold; font-size: 1.1em; }}
    </style>
</head>
<body class="markdown-body">
    <a href="../index.html">&#8592; 返回索引</a>
    <h1>{safe_title}</h1>
    <div id="content"></div>
    <script type="application/json" id="raw-md">{safe_content}</script>
    <script>
      window.addEventListener("DOMContentLoaded", function () {{
        const markdown = JSON.parse(document.getElementById("raw-md").textContent);
        const target = document.getElementById("content");
        if (!window.marked || !window.DOMPurify) {{
          target.textContent = markdown;
          return;
        }}
        const rendered = marked.parse(markdown);
        target.innerHTML = DOMPurify.sanitize(rendered);
      }});
    </script>
</body>
</html>
"""
    with open(filename, "w", encoding="utf-8") as file:
        file.write(html_template)

    index_file = os.path.join(output_dir, "index.html")
    if not os.path.exists(index_file) or os.path.getsize(index_file) == 0:
        old_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>全球 AI 技术雷达 - 历史索引</title></head>
<body class="markdown-body"><h1>全球 AI 技术雷达 - 历史索引</h1>
{PLACEHOLDER}
</body></html>"""
    else:
        with open(index_file, "r", encoding="utf-8") as file:
            old_content = file.read()

    safe_relative = html.escape(relative_filename, quote=True)
    new_link = f'<p><a href="{safe_relative}">{safe_title}</a></p>\n'
    pattern = rf"<p><a href=['\"][^'\"]*['\"]>{re.escape(page_title)}</a></p>\s*"
    clean_content = re.sub(pattern, "", old_content)
    if PLACEHOLDER in clean_content:
        updated_index = clean_content.replace(PLACEHOLDER, f"{PLACEHOLDER}\n{new_link}", 1)
    elif "</h1>" in clean_content:
        head, tail = clean_content.split("</h1>", 1)
        updated_index = f"{head}</h1>\n{PLACEHOLDER}\n{new_link}{tail.lstrip()}"
    else:
        updated_index = f"{new_link}\n{PLACEHOLDER}\n{clean_content}"
    with open(index_file, "w", encoding="utf-8") as file:
        file.write(updated_index)
    return filename


def _pages_url() -> str:
    repository = os.getenv("GITHUB_REPOSITORY", "")
    owner = os.getenv("GITHUB_REPOSITORY_OWNER", "")
    if owner and "/" in repository:
        return f"https://{owner}.github.io/{repository.split('/', 1)[1]}/"
    return os.getenv("GITHUB_PAGES_URL", "")


def send_feishu(webhook: Optional[str], title: str, message: str, *, url: str = "") -> None:
    if not webhook:
        return
    elements: list[Dict[str, Any]] = [{"tag": "div", "text": {"tag": "lark_md", "content": message}}]
    if url:
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "阅读全文"},
                        "type": "primary",
                        "url": url,
                    }
                ],
            }
        )
    response = requests.post(
        webhook,
        json={
            "msg_type": "interactive",
            "card": {
                "header": {"title": {"tag": "plain_text", "content": title}, "template": "orange"},
                "elements": elements,
            },
        },
        timeout=10,
    )
    response.raise_for_status()


def create_llm_client() -> Any:
    from openai import OpenAI

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    return OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def run_report(
    mode: str,
    client_llm: Any,
    *,
    now: Optional[datetime] = None,
    history_path: str = "processed_repos.json",
    output_dir: str = ".",
) -> Dict[str, Any]:
    current = now or get_beijing_time()
    hf_data = scrape_hf(mode=mode, now=current)
    hf_md = process_hf_with_ai(client_llm, hf_data, mode=mode)

    # Initialize this for both modes; weekly reports intentionally have no GitHub section.
    gh_data: list[Dict[str, Any]] = []
    gh_md = ""
    processed_repos: list[Dict[str, Any]] = []
    if mode == "daily":
        gh_data = scrape_github_trending()
        history = load_history(history_path)
        processed_repos = filter_repos(gh_data, history, now=current)
        if processed_repos:
            gh_md = process_github_with_ai(client_llm, processed_repos)
        else:
            gh_md = "> 今日无新增趋势项目（已在近期总结过）。"

    full_md = f"{hf_md}\n\n{gh_md}"
    page_path = ""
    if hf_data or gh_md:
        page_path = generate_page(full_md, mode=mode, now=current, output_dir=output_dir)

    # State is committed only after all AI work and page generation succeed.
    if processed_repos:
        history = load_history(history_path)
        today = current.strftime("%Y-%m-%d")
        for repo in processed_repos:
            history[normalize_repo_name(str(repo["name"]))] = today
        save_history(history, history_path)

    return {"hf_count": len(hf_data), "github_count": len(gh_data), "page_path": page_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    args = parser.parse_args()
    webhook = os.getenv("FEISHU_WEBHOOK")
    try:
        result = run_report(args.mode, create_llm_client())
        send_feishu(
            webhook,
            f"🚀 {'AI 技术周报' if args.mode == 'weekly' else '前沿科研情报'}已更新",
            f"已整理 {result['hf_count']} 篇 HF 论文。",
            url=_pages_url(),
        )
    except Exception as exc:
        LOGGER.exception("Report generation failed: %s", exc)
        try:
            send_feishu(webhook, "❌ AI 论文报告生成失败", f"报告任务失败：{exc}")
        except Exception:
            LOGGER.exception("Failure notification also failed")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
