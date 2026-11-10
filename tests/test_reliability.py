import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import component_hf as hf
import main
import publication
from llm_utils import complete


def paper(number=1, abstract='The experiment reached 42%.'):
    return {'id': f'1234.{number:04d}', 'title': f'Paper {number}', 'upvotes': number,
            'abstract': abstract, 'abstract_source': 'huggingface'}


def answer(p):
    return {'id': p['id'], 'title_zh': '标题', 'translation': '实验达到 42%。',
            'task': '任务', 'highlights': '亮点', 'analysis': '摘要未提供', 'category': 'AI'}


def client_for_answers():
    client = Mock()
    def respond(**kwargs):
        data = json.loads(kwargs['messages'][0]['content'].split('待处理数据：\n')[1])
        return Mock(choices=[Mock(message=Mock(content=json.dumps({'papers': [answer(p) for p in data]})),
                                  finish_reason='stop')])
    client.chat.completions.create.side_effect = respond
    return client


class SourceAndCacheTests(unittest.TestCase):
    def test_missing_abstract_never_calls_model(self):
        client = Mock()
        content = hf.process_hf_with_ai(client, [paper(abstract=''), paper(2, '   ')])
        client.chat.completions.create.assert_not_called()
        self.assertEqual(content.count('摘要获取失败'), 2)
        self.assertNotIn('中文摘要翻译', content)

    def test_arxiv_versioned_response_matches_unversioned_hf_id(self):
        xml = '<feed><entry><id>http://arxiv.org/abs/1234.0001v2</id><summary>Real abstract</summary></entry></feed>'
        with patch('component_hf.request_text', return_value=xml):
            abstracts = hf.fetch_arxiv_abstracts(['1234.0001'])
        result = hf._enrich_paper({'paper': {'id': '1234.0001', 'title': 'Test'}}, abstracts)
        self.assertEqual(result['abstract'], 'Real abstract')
        self.assertEqual(result['abstract_source'], 'arxiv')

    def test_cached_translation_reused_across_modes_and_heat_changes(self):
        client = client_for_answers()
        p = paper()
        with tempfile.TemporaryDirectory() as directory:
            hf.process_hf_with_ai(client, [p], cache_dir=directory)
            p['upvotes'] = 99
            result = hf.process_hf_with_ai(client, [p], 'weekly', cache_dir=directory)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertIn('99 Upvotes', result)
        self.assertIn('每周', result)

    def test_changed_abstract_invalidates_cache(self):
        client = client_for_answers()
        p = paper()
        with tempfile.TemporaryDirectory() as directory:
            hf.process_hf_with_ai(client, [p], cache_dir=directory)
            p['abstract'] = 'Updated abstract'
            hf.process_hf_with_ai(client, [p], cache_dir=directory)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_bad_or_incomplete_response_retries_then_fails(self):
        client = Mock()
        client.chat.completions.create.return_value = Mock(choices=[Mock(
            message=Mock(content='{"papers": []}'), finish_reason='stop')])
        with patch('http_utils.time.sleep'):
            with self.assertRaises(hf.AIProcessingError):
                hf.process_hf_with_ai(client, [paper()])
        self.assertEqual(client.chat.completions.create.call_count, 3)

    def test_successful_batches_survive_later_failure(self):
        papers = [paper(i) for i in range(6)]
        client = client_for_answers()
        respond = client.chat.completions.create.side_effect
        def fail_second(**kwargs):
            if client.chat.completions.create.call_count > 1:
                raise RuntimeError('temporary outage')
            return respond(**kwargs)
        client.chat.completions.create.side_effect = fail_second
        with tempfile.TemporaryDirectory() as directory, patch('http_utils.time.sleep'):
            with self.assertRaises(hf.AIProcessingError):
                hf.process_hf_with_ai(client, papers, cache_dir=directory)
            self.assertEqual(len(list(Path(directory).glob('*.json'))), 5)
            retry_client = client_for_answers()
            hf.process_hf_with_ai(retry_client, papers, cache_dir=directory)
            self.assertEqual(retry_client.chat.completions.create.call_count, 1)

    def test_source_and_generated_html_are_escaped(self):
        p = paper(abstract='<script>bad()</script>')
        row = answer(p)
        row['translation'] = '<img src=x onerror=bad()>'
        result = hf._render_paper(1, p, row)
        self.assertNotIn('<script>', result)
        self.assertNotIn('<img src=x', result)
        self.assertIn('&lt;script&gt;', result)
        self.assertIn('未读取全文', result)

    def test_duplicate_unknown_and_missing_translation_are_rejected(self):
        row = answer(paper())
        for rows in [[row, row], [{**row, 'id': 'other'}], [{**row, 'translation': ''}]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                hf._validate_summaries(json.dumps({'papers': rows}), [row['id']])


class RuntimeTests(unittest.TestCase):
    def test_client_has_one_retry_layer_and_explicit_timeout(self):
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-key'}), patch('openai.OpenAI') as create:
            main.create_llm_client()
        self.assertEqual(create.call_args.kwargs['max_retries'], 0)
        self.assertEqual(create.call_args.kwargs['timeout'], 180)

    def test_truncated_response_is_not_published(self):
        client = Mock()
        client.chat.completions.create.return_value = Mock(choices=[Mock(
            message=Mock(content='partial'), finish_reason='length')])
        with patch('http_utils.time.sleep'), self.assertRaises(Exception):
            complete(client, 'prompt')
        self.assertEqual(client.chat.completions.create.call_count, 3)


class PublicationTests(unittest.TestCase):
    def test_staging_contains_only_public_assets_and_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            os.chdir(directory)
            try:
                Path('archive').mkdir()
                Path('.run').mkdir()
                Path('archive/report.html').write_text('report')
                Path('archive/report.json').write_text('{}')
                Path('index.html').write_text('<a href="archive/report.html">Report</a>')
                Path('private.env').write_text('not public')
                Path('.run/report.json').write_text(json.dumps({'page_path': 'archive/report.html'}))
                manifest = publication.prepare()
                self.assertFalse(Path('_site/private.env').exists())
                self.assertFalse(Path('_site/.run').exists())
                self.assertEqual(manifest['files']['archive/report.html'], hashlib.sha256(b'report').hexdigest())
                Path('index.html').write_text('missing link')
                with self.assertRaises(ValueError):
                    publication.prepare()
            finally:
                os.chdir(previous)

    def test_verify_waits_for_exact_live_bytes(self):
        manifest = {'run_id': '1', 'run_attempt': '1', 'files': {'archive/a.html': hashlib.sha256(b'new').hexdigest()}}
        client = Mock()
        client.get.side_effect = [Mock(json=lambda: {}), Mock(json=lambda: manifest), Mock(content=b'new')]
        with patch('publication.time.sleep') as sleep:
            publication.verify('https://example.test/', manifest, attempts=2, session=client)
        sleep.assert_called_once()
        self.assertEqual(client.get.call_count, 3)

    def test_mismatched_live_content_fails_without_success(self):
        manifest = {'run_id': '1', 'run_attempt': '1', 'files': {'archive/a.html': 'new-hash'}}
        client = Mock()
        client.get.side_effect = [Mock(json=lambda: manifest), Mock(content=b'old')]
        with self.assertRaises(RuntimeError):
            publication.verify('https://example.test/', manifest, attempts=1, session=client)

    def test_notification_requires_verification_marker(self):
        with patch('sys.argv', ['publication.py', 'notify']), patch('publication.Path.is_file', return_value=False), patch('publication.send_feishu') as send:
            with self.assertRaises(RuntimeError):
                publication.main()
        send.assert_not_called()


if __name__ == '__main__':
    unittest.main()
