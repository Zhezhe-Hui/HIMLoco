#!/usr/bin/env python3
"""Read-only HTTP preflight. Never starts tasks or sends robot commands."""
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
from urllib.request import urlopen


def inspect(base):
    parts = urlsplit(base)
    if parts.scheme not in ('http', 'https') or not parts.netloc or parts.path not in ('', '/'):
        raise ValueError('请使用网页根地址，例如 http://10.118.86.232:8765')
    base = base.rstrip('/')
    checks = []

    def check(name, ok, detail):
        checks.append(dict(name=name, ok=bool(ok), detail=detail))

    def fetch(route):
        with urlopen(base + route, timeout=8) as response:
            return response.read(), response.headers

    page, headers = fetch('/')
    check('网页响应', bool(page), 'HTTP 200')
    raw, _ = fetch('/api/state')
    first = json.loads(raw)
    time.sleep(1)
    raw, _ = fetch('/api/state')
    state = json.loads(raw)
    camera, real = state.get('camera', {}), state.get('real', {})
    task = real.get('task') or {}
    check('板载模式', state.get('runtime', {}).get('location') == 'robot', state.get('runtime'))
    check('相机持续更新', camera.get('connected') and camera.get('frame', -1) > first.get('camera', {}).get('frame', -1),
          dict(frame=camera.get('frame'), previous=first.get('camera', {}).get('frame'),
               fps=camera.get('received_fps'), error=camera.get('error')))
    check('本体与输入新鲜', real.get('connected') and real.get('inputs_fresh'),
          dict(age_ms=real.get('state_age_ms'), error=real.get('bridge_error')))
    check('当前没有实机任务', task.get('outcome') != 'running',
          dict(kind=task.get('kind'), outcome=task.get('outcome')))
    check('当前没有流程模拟', not state.get('active'), bool(state.get('active')))
    server_date = headers.get('Date')
    skew = None
    if server_date:
        skew = abs((datetime.now(timezone.utc) - parsedate_to_datetime(server_date)).total_seconds())
    check('服务时间与电脑相符', skew is not None and skew < 60,
          dict(server_date=server_date, absolute_offset_seconds=skew))
    cal = real.get('calibration', {})
    return dict(checked_at=datetime.now().astimezone().isoformat(), url=base,
                checks=checks, ok=all(c['ok'] for c in checks),
                calibration=dict(ground_checked_this_session=cal.get('ground_checked_this_session'),
                                 approved_this_session=cal.get('approved_this_session'), needs=cal.get('needs')),
                prior_task_bridge_writes=real.get('bridge_writes'),
                note='只读检查，不代表可行走；需站立地面复核、现场确认和影子规划。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://10.118.86.232:8765')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = inspect(args.url)
    except Exception as exc:
        report = dict(ok=False, url=args.url, error=str(exc),
                      checked_at=datetime.now().astimezone().isoformat())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for item in report.get('checks', []):
        print(('通过' if item['ok'] else '待解决') + '：' + item['name'] + ' ' +
              json.dumps(item['detail'], ensure_ascii=False))
    if report.get('error'):
        print('检查失败：' + report['error'])
    print(report.get('note', '未执行任何运动操作。'))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
