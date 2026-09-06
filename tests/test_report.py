import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import main
from component_hf import process_hf_with_ai, scrape_hf
from http_utils import request_with_retry


class ReportPipelineTests(unittest.TestCase):
    def test_legacy_history_repo_spacing_is_normalized(self):
        self.assertEqual(main.normalize_repo_name(" owner / repo "), "owner/repo")

    def test_weekly_does_not_require_github_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("main.scrape_hf", return_value=[]), patch(
                "main.process_hf_with_ai", return_value=""
            ), patch("main.scrape_github_trending") as github_scrape, patch(
                "main.generate_page"
            ) as generate:
                result = main.run_report(
                    "weekly",
                    Mock(),
                    now=datetime(2026, 8, 10),
                    output_dir=directory,
                    history_path=str(Path(directory) / "history.json"),
                )
        github_scrape.assert_not_called()
        generate.assert_not_called()
        self.assertEqual(result["github_count"], 0)

    def test_weekly_page_is_named_for_previous_completed_week(self):
        with tempfile.TemporaryDirectory() as directory:
            page = main.generate_page(
                "Weekly content",
                mode="weekly",
                now=datetime(2026, 9, 1),
                output_dir=directory,
            )
        self.assertTrue(page.endswith("archive/2026_W35.html"))

    def test_failed_github_ai_does_not_update_history(self):
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.json"
            history_path.write_text(json.dumps({}), encoding="utf-8")
            repos = [{"name": "owner/repo", "link": "https://github.com/owner/repo"}]
            with patch("main.scrape_hf", return_value=[]), patch(
                "main.process_hf_with_ai", return_value=""
            ), patch("main.scrape_github_trending", return_value=repos), patch(
                "main.process_github_with_ai", side_effect=RuntimeError("LLM down")
            ):
                with self.assertRaises(RuntimeError):
                    main.run_report(
                        "daily",
                        Mock(),
                        now=datetime(2026, 8, 10),
                        history_path=str(history_path),
                        output_dir=directory,
                    )
            self.assertEqual(json.loads(history_path.read_text(encoding="utf-8")), {})

    def test_script_json_escapes_script_terminator(self):
        encoded = main._safe_script_json("</script><script>alert(1)</script>")
        self.assertNotIn("</script>", encoded)
        self.assertEqual(json.loads(encoded), "</script><script>alert(1)</script>")

    def test_generated_page_uses_sanitized_json_and_pinned_sanitizer(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(
                main.generate_page(
                    "</script><img src=x onerror=alert(1)>",
                    now=datetime(2026, 8, 10),
                    output_dir=directory,
                )
            )
            html = page.read_text(encoding="utf-8")
        self.assertIn('type="application/json"', html)
        self.assertNotIn("</script><img", html)
        self.assertIn("marked@12.0.2", html)
        self.assertIn("dompurify@3.1.6", html)


class PaperInputTests(unittest.TestCase):
    def test_weekly_fetches_previous_completed_week(self):
        with patch("component_hf.request_json", return_value=[]) as fetch, patch(
            "component_hf.fetch_arxiv_abstracts"
        ):
            papers = scrape_hf("weekly", now=datetime(2026, 9, 1))
        self.assertEqual(papers, [])
        self.assertIn("week=2026-W35", fetch.call_args.args[0])

    def test_weekly_fetches_all_pages_and_deduplicates_papers(self):
        first = {"paper": {"id": "1234.0001", "title": "First", "abstract": "One"}}
        second = {"paper": {"id": "1234.0002", "title": "Second", "abstract": "Two"}}
        with patch(
            "component_hf.request_json",
            side_effect=[[first], [first, second], []],
        ) as fetch, patch("component_hf.fetch_arxiv_abstracts"):
            papers = scrape_hf("weekly", now=datetime(2026, 8, 31))
        self.assertEqual([paper["id"] for paper in papers], ["1234.0001", "1234.0002"])
        self.assertEqual(fetch.call_count, 3)
        self.assertIn("p=2", fetch.call_args.args[0])

    def test_weekly_period_handles_year_boundary(self):
        with patch("component_hf.request_json", return_value=[]) as fetch, patch(
            "component_hf.fetch_arxiv_abstracts"
        ):
            scrape_hf("weekly", now=datetime(2027, 1, 4))
        self.assertIn("week=2026-W53", fetch.call_args.args[0])

    def test_hf_scraper_keeps_empty_today_result_without_fetching_yesterday(self):
        with patch("component_hf.request_json", return_value=[]) as fetch, patch(
            "component_hf.fetch_arxiv_abstracts"
        ) as abstracts:
            papers = scrape_hf("daily", now=datetime(2026, 8, 10))
        self.assertEqual(papers, [])
        fetch.assert_called_once()
        abstracts.assert_called_once_with([], session=None)

    def test_existing_hf_abstract_skips_arxiv_lookup(self):
        payload = [{"paper": {"id": "1234.5678", "title": "Test", "abstract": "HF abstract"}}]
        with patch("component_hf.request_json", return_value=payload), patch(
            "component_hf.fetch_arxiv_abstracts"
        ) as abstracts:
            papers = scrape_hf("daily", now=datetime(2026, 8, 10))
        self.assertEqual(papers[0]["abstract"], "HF abstract")
        abstracts.assert_called_once_with([], session=None)

    def test_summary_prompt_contains_abstract(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            Mock(choices=[Mock(message=Mock(content="### 1. Test"))]),
            Mock(choices=[Mock(message=Mock(content="### 11. Test"))]),
            Mock(choices=[Mock(message=Mock(content="### 21. Test"))]),
            Mock(choices=[Mock(message=Mock(content="### 31. Test"))]),
        ]
        papers = [
            {"id": f"1234.{index:04d}", "title": f"Test {index}", "upvotes": index, "abstract": "A real abstract."}
            for index in range(1, 32)
        ]
        process_hf_with_ai(client, papers)
        prompt = client.chat.completions.create.call_args_list[0].kwargs["messages"][0]["content"]
        self.assertIn("A real abstract.", prompt)
        self.assertEqual(client.chat.completions.create.call_count, 4)


class RetryTests(unittest.TestCase):
    def test_transient_http_error_is_retried(self):
        response = Mock(status_code=200, text="ok")
        response.raise_for_status.return_value = None
        session = Mock()
        session.request.side_effect = [Mock(status_code=503), response]
        with patch("http_utils.time.sleep") as sleep:
            result = request_with_retry("GET", "https://example.test", session=session)
        self.assertIs(result, response)
        self.assertEqual(session.request.call_count, 2)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
