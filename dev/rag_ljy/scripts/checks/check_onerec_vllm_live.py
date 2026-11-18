#!/usr/bin/env python3
"""Real standalone OneRec vLLM generation smoke check (not RAG acceptance)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.request import Request, ProxyHandler, build_opener

ROOT=Path(__file__).resolve().parents[2]


def request(url, payload=None):
    body=json.dumps(payload).encode() if payload is not None else None
    with build_opener(ProxyHandler({})).open(Request(url,data=body,
            headers={'Content-Type':'application/json'}),timeout=180) as response:
        return json.load(response)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8090')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    models=request(args.url+'/v1/models')
    if 'onerec-8b' not in {model['id'] for model in models['data']}:
        raise RuntimeError('Expected onerec-8b service')
    print('PREFLIGHT=passed',flush=True)
    if not args.run:
        return 0
    out=ROOT/'experiment_outputs/checks/onerec_vllm_smoke'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out.mkdir(parents=True)
    print('OUTPUT='+str(out),flush=True)
    (out/'models.json').write_text(json.dumps(models,indent=2))
    (out/'check_onerec_vllm_live.py').write_text(Path(__file__).read_text())
    cases=[('largest planet','Jupiter is the largest planet in the Solar System.'),
           ('capital of France','Paris is the capital of France.'),
           ('chemical formula of water','The chemical formula of water is H2O.'),
           ('number of days in a week','There are seven days in a week.')]
    def send(case):
        question,source=case
        payload={'model':'onerec-8b','messages':[
            {'role':'system','content':'Answer using only the supplied source. Give a short answer and cite [1].'},
            {'role':'user','content':f'Source [1]: {source}\nQuestion: What is the {question}?'}],
            'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}}
        started=time.monotonic()
        row={'request':payload}
        try:
            response=request(args.url+'/v1/chat/completions',payload)
            row['response']=response
            text=response['choices'][0]['message']['content']
            row['passed']=bool(isinstance(text,str) and text.strip() and response['usage']['completion_tokens']>0)
        except Exception as error:
            row.update(passed=False,error=repr(error))
        row['latency_ms']=(time.monotonic()-started)*1000
        return row
    serial=send(cases[0])
    # Four concurrent real HTTP requests; no mocked backend or synthetic allocation.
    with ThreadPoolExecutor(4) as pool:
        concurrent=list(pool.map(send,cases))
    rows=[serial]+concurrent
    (out/'responses.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    (out/'generated_responses.txt').write_text('\n\n'.join(
        row.get('response',{}).get('choices',[{'message':{'content':row.get('error','')}}])[0]['message']['content'] or ''
        for row in rows))
    summary={'status':'passed' if all(row['passed'] for row in rows) else 'failed',
        'requests':5,'serial_requests':1,'concurrent_requests':4,
        'successful':sum(row['passed'] for row in rows),
        'scope':'real standalone vLLM generation; not RAG priority, watermark or quality acceptance'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return 0 if summary['status']=='passed' else 1


if __name__=='__main__':
    sys.exit(main())
