#!/usr/bin/env python3
"""One synthetic typed-decision request; no benchmark inputs needed."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from decision_index.engines.base import validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:8008')
    args = ap.parse_args()
    questions = {
        'color': {'type': 'choice', 'instructions': 'What color is the kettle?',
                  'criteria': {'red': 'Red', 'blue': 'Blue'}},
        'red': {'type': 'noul', 'instructions': 'The kettle is red.'},
    }
    with httpx.Client(base_url=args.url, timeout=180, trust_env=False) as client:
        health = client.get('/health')
        health.raise_for_status()
        result = client.post('/v1/systemone', json={'state': 'The kettle is red.', 'questions': questions})
        result.raise_for_status()
        response = result.json()
    validate(questions, response)
    assert response['answers']['color']['choice'] == 'red', response
    assert response['answers']['red']['noul'] > 0.5, response
    print(json.dumps({'smoke_test': 'passed', 'response': response}, indent=2))


if __name__ == '__main__':
    main()
