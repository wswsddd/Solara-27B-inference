"""HTTP integration checks using a deterministic fake engine."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from decision_index.engines.base import Unsupported, validate
from decision_index.engines.http import HttpSystemOne
from decision_index.systemone_server import make_handler, create_engine


class FakeEngine:
    model_id = 'Solara-27B'
    read_twice = True

    def __call__(self, state, questions):
        if state == 'capacity':
            raise Unsupported('Too many labels')
        return {'model': self.model_id, 'answers': {
            k: {'type': 'noul', 'noul': 0.9} if q['type'] == 'noul' else
            {'type': 'choice', 'choice': 'red', 'probabilities': {'red': 0.9, 'blue': 0.1}}
            for k, q in questions.items()}}, {}


class ServerTests(unittest.TestCase):
    def test_release_fixes_read_twice_on_and_thinking_off(self):
        args = SimpleNamespace(model='Solara-27B', tokenizer='checkpoint', device='cpu')
        with patch('decision_index.systemone_server.UnifiedLMHeadEngine') as factory:
            factory.return_value = FakeEngine()
            engine = create_engine(args)
        factory.assert_called_once_with(model='checkpoint', device='cpu',
                                        read_twice=True, enable_thinking=False)
        self.assertEqual(engine.model_id, 'Solara-27B')

    def setUp(self):
        self.ready = True
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(FakeEngine(), lambda: self.ready))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = 'http://127.0.0.1:' + str(self.server.server_port)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        r = urllib.request.Request(self.url + path, data=data, headers={'Content-Type': 'application/json'})
        try:
            response = self.opener.open(r, timeout=3)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    def payload(self):
        return {'state': 'The kettle is red.', 'questions': {
            'color': {'type': 'choice', 'criteria': {'red': 'Red', 'blue': 'Blue'}},
            'flag': {'type': 'noul', 'instructions': 'The kettle is red.'}}}

    def test_typed_response(self):
        p = self.payload(); status, response = self.request('/v1/systemone', p)
        self.assertEqual(status, 200); validate(p['questions'], response)

    def test_health_follows_backend(self):
        self.assertEqual(self.request('/health')[0], 200)
        self.ready = False
        self.assertEqual(self.request('/health')[0], 503)

    def test_unknown_model(self):
        p = self.payload(); p['model'] = 'wrong'
        self.assertEqual(self.request('/v1/systemone', p)[0], 404)

    def test_invalid_questions(self):
        p = self.payload(); p['questions']['color']['criteria'] = []
        self.assertEqual(self.request('/v1/systemone', p)[0], 422)

    def test_object_instructions_from_official_schema(self):
        p = self.payload()
        p['questions']['color']['instructions'] = {'task': 'Pick the kettle color.'}
        status, response = self.request('/v1/systemone', p)
        self.assertEqual(status, 200)
        validate(p['questions'], response)

    def test_capacity_uses_official_marker(self):
        p = self.payload(); p['state'] = 'capacity'
        status, response = self.request('/v1/systemone', p)
        self.assertEqual(status, 413)
        self.assertIn('options per choice', response['error'])

    def test_do_not_silently_accept_thinking(self):
        p = self.payload(); p['enable_thinking'] = True
        self.assertEqual(self.request('/v1/systemone', p)[0], 422)

    def test_official_http_engine_can_call_frontend(self):
        p = self.payload()
        engine = HttpSystemOne(base_url=self.url, model=FakeEngine.model_id)
        try:
            response, _ = engine(p['state'], p['questions'])
            validate(p['questions'], response)
            with self.assertRaises(Unsupported):
                engine('capacity', p['questions'])
        finally:
            engine.close()

    def test_documented_smoke_command(self):
        result = subprocess.run([sys.executable, str(Path(__file__).parent/'scripts/smoke_test.py'),
                                 '--url', self.url], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['smoke_test'], 'passed')


if __name__ == '__main__':
    unittest.main()
