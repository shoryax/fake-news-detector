#!/usr/bin/env python3
"""Simple smoke test for the /predict endpoint.

Usage:
  python scripts/smoke_test.py http://127.0.0.1:5000
  python scripts/smoke_test.py https://your-vercel-app.vercel.app
"""
import sys
import requests


def main():
    if len(sys.argv) < 2:
        print('Usage: python scripts/smoke_test.py <base_url>')
        sys.exit(2)
    base = sys.argv[1].rstrip('/')
    url = base + '/predict'
    payload = {'text': 'This is a short test article.'}
    try:
        r = requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print('Request failed:', e)
        sys.exit(1)
    print('HTTP', r.status_code)
    try:
        j = r.json()
    except Exception:
        print('Response is not JSON:')
        print(r.text[:1000])
        sys.exit(1)
    print('JSON:', j)
    if r.status_code != 200:
        print('Smoke test failed: status != 200')
        sys.exit(1)
    if 'prediction' not in j:
        print('Smoke test failed: missing prediction key')
        sys.exit(1)
    print('Smoke test OK')


if __name__ == '__main__':
    main()
