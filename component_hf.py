from __future__ import annotations

import datetime as dt
import json
import logging
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

from http_utils import RequestError, request_json, request_text, retry_call


LOGGER = logging.getLogger(__name__)
BEIJING_TZ = ZoneInfo("Asia/Shanghai")
HF_HEADERS = {
    "User-Agent": "daily-hf-report/1.0 (+https://github.com/RedBeanCake/daily-hf-report)",
}
# How many days to walk back when Hugging Face rejects the requested date as
# being in the future. Guards against the scheduler firing after Beijing
# midnight while Hugging Face still settles the "latest" date in UTC.
MAX_DATE_FALLBACK_DAYS = 3


class AIProcessingError(RuntimeError):
    """Raised when one or more paper summaries could not be generated."""


def get_beijing_time() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).astimezone(BEIJING_TZ)


def _value_from_paper(paper: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = paper.get(key)
        if value not in (None, ""):
            return value
    return default


def fetch_arxiv_abstracts(paper_ids: Iterable[str], *, session: Any = None) -> Dict[str, str]:
    """Fetch many abstracts in one arXiv API request."""

    ids = [paper_id for paper_id in paper_ids if paper_id]
    if not ids:
        return {}
    url = f"https://export.arxiv.org/api/query?id_list={quote(','.join(ids))}"
    try:
        xml_text = request_text(url, session=session, headers=HF_HEADERS, timeout=15)
        root = ET.fromstring(xml_text)
    except (RequestError, ET.ParseError) as exc:
        LOGGER.warning("Could not fetch arXiv abstracts: %s", exc)
        return {}

    abstracts: Dict[str, str] = {}
    for entry in root.iter():
        if not entry.tag.endswith("entry"):
            continue
        paper_id = ""
        abstract = ""
        for child in entry:
            if child.tag.endswith("id") and child.text:
                paper_id = child.text.rsplit("/", 1)[-1]
            elif child.tag.endswith("summary") and child.text:
                abstract = " ".join(child.text.split())
        if paper_id and abstract:
            abstracts[paper_id] = abstract
    return abstracts


def _hf_daily_papers_url(mode: str, current: dt.datetime) -> str:
    if mode == "weekly":
        year, week, _ = current.isocalendar()
        return f"https://huggingface.co/api/daily_papers?week={year}-W{week:02d}&limit=50"
    date_string = current.strftime("%Y-%m-%d")
    return f"https://huggingface.co/api/daily_papers?date={date_string}&limit=50"


def _fetch_hf_payload(mode: str, current: dt.datetime, *, session: Any) -> Any:
    """Fetch the HF daily/weekly payload, tolerating a future-dated request.

    Hugging Face settles its "latest" date in UTC, so a job that fires after
    Beijing midnight (but before UTC midnight) asks for a date HF still regards
    as the future and receives HTTP 400. In that case we walk the date back a
    day at a time. A *successful but empty* response is returned as-is: an
    empty day is a real answer, not an error, and must not be papered over with
    a different date's results.
    """

    attempt_time = current
    for offset in range(MAX_DATE_FALLBACK_DAYS + 1):
        url = _hf_daily_papers_url(mode, attempt_time)
        try:
            return request_json(url, session=session, headers=HF_HEADERS, timeout=15)
        except RequestError as exc:
            # Only a 400 (invalid/future date) is recoverable by stepping back;
            # anything else propagates unchanged.
            if exc.status_code != 400 or offset == MAX_DATE_FALLBACK_DAYS:
                raise
            attempt_time = attempt_time - dt.timedelta(days=1)
            LOGGER.warning(
                "HF rejected %s as future-dated; retrying with %s",
                url,
                attempt_time.strftime("%Y-%m-%d"),
            )
    # Unreachable: the loop either returns or raises.
    raise RequestError("Hugging Face date fallback exhausted")


def scrape_hf(
    mode: str = "daily",
    *,
    now: Optional[dt.datetime] = None,
    session: Any = None,
) -> List[Dict[str, Any]]:
    """Fetch HF Daily Papers without substituting another date's results."""

    current = now or get_beijing_time()

    payload = _fetch_hf_payload(mode, current, session=session)
    if not isinstance(payload, list):
        raise RequestError("Hugging Face API returned an unexpected payload")

    items = [item for item in payload if isinstance(item, dict)]
    missing_ids = [
        str((item.get("paper") or {}).get("id", ""))
        for item in items
        if not ((item.get("summary") or item.get("abstract")) or (item.get("paper") or {}).get("summary") or (item.get("paper") or {}).get("abstract"))
    ]
    abstracts = fetch_arxiv_abstracts(missing_ids, session=session)
    # Enrichment is best-effort: a missing arXiv abstract must not discard the HF item.
    return [_enrich_paper(item, abstracts) for item in items]


def _enrich_paper(item: Dict[str, Any], abstracts: Dict[str, str]) -> Dict[str, Any]:
    paper = item.get("paper") or {}
    paper_id = str(_value_from_paper(paper, "id", default=""))
    abstract = item.get("summary") or item.get("abstract")
    abstract = abstract or paper.get("summary") or paper.get("abstract") or ""
    abstract = str(abstract or "").strip()
    source = "huggingface" if abstract else "arxiv"
    if not abstract:
        abstract = abstracts.get(paper_id, "")
    if not abstract:
        source = "unavailable"

    return {
        "id": paper_id,
        "title": _value_from_paper(paper, "title", default="Unknown"),
        "upvotes": item.get("upvotes") or paper.get("upvotes", 0) or 0,
        "abstract": abstract,
        "abstract_source": source,
        "authors": _value_from_paper(paper, "authors", default=[]),
        "published": _value_from_paper(paper, "publishedAt", "published", default=""),
        "paper_url": f"https://arxiv.org/abs/{paper_id}" if paper_id else "",
    }


def _llm_call_with_retry(client_llm: Any, prompt: str) -> str:
    def call() -> str:
        completion = client_llm.chat.completions.create(
            model="qwen3.7-plus",
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty content")
        return content

    return retry_call(call, attempts=3, base_delay=1.0)


def process_hf_with_ai(
    client_llm: Any,
    hf_papers: Iterable[Dict[str, Any]],
    mode: str = "daily",
) -> str:
    """Summarize paper metadata and abstracts; fail loudly on incomplete batches."""

    papers = list(hf_papers)
    if not papers:
        return ""

    papers.sort(key=lambda item: item.get("upvotes", 0) or 0, reverse=True)
    # Process every paper returned by the HF endpoint. Chunking keeps each prompt bounded.
    selected = papers
    chunk_size = 10
    report_type = "每周" if mode == "weekly" else "今日"
    chunks: List[str] = []
    failed_chunks: List[int] = []

    for start in range(0, len(selected), chunk_size):
        chunk = selected[start : start + chunk_size]
        start_number = start + 1
        prompt = f"""你是一个 AI 大模型专家。请用平实、地道的中文总结以下 Hugging Face {report_type}论文。
只允许依据每篇论文提供的标题、摘要和元数据作答。摘要没有提到的实验结果、数字、数据集或技术细节必须写“摘要未提供”，禁止猜测或补全。

要求：
1. 从编号 {start_number} 开始连续编号。
2. 每篇论文提供：中文标题翻译、研究任务、核心亮点、深度解析、领域归类。
3. 保留论文链接和社区热度，不要虚构作者或实验数据。
4. 输出 Markdown，严格遵循：
   ### {start_number}. [英文标题] (中文标题翻译)
   - **社区热度**: `👍 [对应 upvotes] Upvotes`
   - **论文链接**: [点击跳转](https://arxiv.org/abs/[对应 id])
   - **研究任务**: ...
   - **核心亮点**: ...
   - **深度解析**: ...
   - **领域归类**: [...]
   ---

待处理数据：
{json.dumps(chunk, ensure_ascii=False)}
"""
        try:
            chunks.append(_llm_call_with_retry(client_llm, prompt))
        except Exception as exc:
            failed_chunks.append(start_number)
            LOGGER.exception("HF summary batch %s failed: %s", start_number, exc)

    if failed_chunks:
        raise AIProcessingError(f"HF summary batches failed: {failed_chunks}")

    content = "\n\n".join(chunks)
    return (
        f"<details>\n<summary><b>🤗 Hugging Face {report_type}论文 (点击展开)</b></summary>\n\n"
        f"{content}\n</details>"
    )
