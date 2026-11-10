"""Stage public files and verify their exact content before announcing success."""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time

import requests

from main import _pages_url, send_feishu

LOGGER = logging.getLogger(__name__)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(result_file: str = ".run/report.json", site_dir: str = "_site") -> dict:
    result = json.loads(Path(result_file).read_text(encoding="utf-8"))
    page = Path(result["page_path"])
    if page.is_absolute() or ".." in page.parts or page.parts[0] != "archive":
        raise ValueError("Report must be a relative archive path")
    required = [Path("index.html"), page, page.with_suffix(".json")]
    if any(not path.is_file() or not path.stat().st_size for path in required):
        raise ValueError("Missing or empty publication files")
    if page.as_posix() not in Path("index.html").read_text(encoding="utf-8"):
        raise ValueError("Generated report is missing from the index")
    manifest = {
        "run_id": os.getenv("GITHUB_RUN_ID", "local"),
        "run_attempt": os.getenv("GITHUB_RUN_ATTEMPT", "1"),
        "commit": os.getenv("GITHUB_SHA", "local"),
        "report": result,
        "files": {path.as_posix(): digest(path) for path in required},
    }
    site = Path(site_dir)
    site.mkdir(parents=True, exist_ok=True)
    shutil.copy2("index.html", site / "index.html")
    shutil.copytree("archive", site / "archive", dirs_exist_ok=True)
    if Path("CNAME").is_file():
        shutil.copy2("CNAME", site / "CNAME")
    (site / ".nojekyll").touch()
    payload = json.dumps(manifest, ensure_ascii=False, indent=2)
    (site / "publication.json").write_text(payload, encoding="utf-8")
    Path(".run").mkdir(exist_ok=True)
    Path(".run/publication.json").write_text(payload, encoding="utf-8")
    return manifest


def verify(base_url: str, manifest: dict, *, attempts: int = 40, delay: float = 15,
           session=None) -> None:
    if not base_url.startswith("https://"):
        raise ValueError("A public HTTPS Pages URL is required")
    client = session or requests.Session()
    base_url = base_url.rstrip("/") + "/"
    for attempt in range(attempts):
        try:
            params = {"run": manifest["run_id"], "attempt": manifest["run_attempt"], "check": attempt}
            response = client.get(base_url + "publication.json", params=params, timeout=15)
            response.raise_for_status()
            if response.json() != manifest:
                raise ValueError("Pages is still serving a different publication")
            for path, expected in manifest["files"].items():
                response = client.get(base_url + path, params=params, timeout=15)
                response.raise_for_status()
                if hashlib.sha256(response.content).hexdigest() != expected:
                    raise ValueError(f"Pages content has not updated: {path}")
            LOGGER.info("Verified published index, report and source snapshot at %s", base_url)
            return
        except (requests.RequestException, ValueError) as exc:
            LOGGER.info("Waiting for Pages (%s/%s): %s", attempt + 1, attempts, exc)
            if attempt + 1 < attempts:
                time.sleep(delay)
    raise RuntimeError("Pages publication could not be verified; no success notification sent")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "verify", "notify"])
    parser.add_argument("--status", choices=["success", "failure"], default="success")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "verify":
        manifest = json.loads(Path(".run/publication.json").read_text(encoding="utf-8"))
        verify(_pages_url(), manifest)
        Path(".run/verified").write_text("verified", encoding="utf-8")
    elif args.status == "success":
        if not Path(".run/verified").is_file():
            raise RuntimeError("Refusing to announce an unverified publication")
        result = json.loads(Path(".run/report.json").read_text(encoding="utf-8"))
        send_feishu(os.getenv("FEISHU_WEBHOOK"), "✅ AI 论文报告已发布",
                    f"已整理 {result['hf_count']} 篇 HF 论文，网页内容已验证。",
                    url=_pages_url().rstrip("/") + "/" + result["page_path"])
    else:
        run_url = f"https://github.com/{os.getenv('GITHUB_REPOSITORY', '')}/actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
        send_feishu(os.getenv("FEISHU_WEBHOOK"), "❌ AI 论文报告生成或发布失败",
                    "报告未确认发布，请检查工作流日志。", url=run_url)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
