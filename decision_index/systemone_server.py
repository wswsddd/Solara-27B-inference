"""Thin System One HTTP frontend over the submitted unified LM-head engine."""
import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from decision_index.engines.base import Unsupported, validate
from decision_index.engines.unified_lm_head_engine import UnifiedLMHeadEngine

MAX_BODY_BYTES = 16 * 1024 * 1024


def create_engine(args):
    engine = UnifiedLMHeadEngine(model=args.tokenizer, device=args.device,
                                 read_twice=True, enable_thinking=False)
    engine.model_id = args.model
    return engine


def make_handler(engine, backend_ready):
    # A single decision request at a time, matching the latency-test workload.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                self.reply(404, {'error': 'Unknown endpoint'})
                return
            ready = backend_ready()
            self.reply(200 if ready else 503, {'ready': ready, 'model': engine.model_id,
                                             'enable_thinking': False,
                                             'read_twice': getattr(engine, 'read_twice', None)})

        def do_POST(self):
            if self.path != '/v1/systemone':
                self.reply(404, {'error': 'Unknown endpoint'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= MAX_BODY_BYTES:
                    self.reply(413, {'error': f'context window transport limit: request body capacity is {MAX_BODY_BYTES} bytes'})
                    return
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict) or 'state' not in payload:
                    raise ValueError('Expected state and questions')
                if payload.get('model') not in (None, 'default', engine.model_id):
                    self.reply(404, {'error': 'Unknown model'})
                    return
                if set(payload) - {'model', 'state', 'questions'}:
                    raise ValueError('Unsupported request fields')
                questions = payload.get('questions')
                if not isinstance(questions, dict) or not questions:
                    raise ValueError('questions must be a nonempty object')
                for q in questions.values():
                    if not isinstance(q, dict) or q.get('type') not in ('choice', 'noul'):
                        raise ValueError('Supported question types: choice, noul')
                    if 'instructions' in q and not isinstance(q['instructions'], (str, dict)):
                        raise ValueError('instructions must be a string or object')
                    if q['type'] == 'choice' and (not isinstance(q.get('criteria'), dict) or len(q['criteria']) < 2):
                        raise ValueError('a choice needs at least two options')
                with lock:
                    response, _ = engine(payload['state'], questions)
                    validate(questions, response)
                self.reply(200, response)
            except Unsupported as exc:
                # Official HTTP engine recognizes capacity errors through markers.
                self.reply(413, {'error': 'Unsupported capacity (context window / options per choice): ' + str(exc)})
            except (ValueError, TypeError, KeyError, UnicodeDecodeError) as exc:
                self.reply(422, {'error': str(exc)})
            except Exception:
                logging.exception('Decision request failed')
                self.reply(502, {'error': 'Backend decision request failed; see server log'})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tokenizer', required=True, help='Downloaded checkpoint directory')
    ap.add_argument('--model', default='Solara-27B')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8008)
    args = ap.parse_args()
    engine = create_engine(args)

    def backend_ready():
        return True

    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, backend_ready))
    print(f'System One frontend: http://{args.host}:{args.port}; model: {args.model}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        engine.close()


if __name__ == '__main__':
    main()
