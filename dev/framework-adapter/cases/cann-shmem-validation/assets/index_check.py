"""Read official index and independently check its cp311/aarch64 wheel hashes.

Downloads are diagnostic only. Nothing from this script is installed or executed.
"""
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import urllib.parse
import urllib.request

URL='https://ascend.devcloud.huaweicloud.com/cann/pypi/simple/cann-shmem/'


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.items=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            if 'cp311-cp311-linux_aarch64.whl' in href:
                self.items.append(urllib.parse.urljoin(URL,href))


def main():
    out=Path('/results'); report={'index':URL,'wheels':[]}
    req=urllib.request.Request(URL,headers={'Cache-Control':'no-cache'})
    with urllib.request.urlopen(req,timeout=30) as response:
        body=response.read(); report['index_headers']=dict(response.headers)
    (out/'official-index.html').write_bytes(body)
    links=Links(); links.feed(body.decode())
    report['links']=links.items
    # Bounded download: inspect 1.7.0 and at most two adjacent older wheels.
    for url in sorted(set(links.items),reverse=True)[:3]:
        clean,fragment=urllib.parse.urldefrag(url)
        expected=urllib.parse.parse_qs(fragment).get('sha256',[None])[0]
        req=urllib.request.Request(clean,headers={'Cache-Control':'no-cache'})
        with urllib.request.urlopen(req,timeout=60) as response:
            data=response.read(30*1024*1024+1); headers=dict(response.headers)
        assert len(data)<=30*1024*1024, 'unexpectedly large wheel'
        actual=hashlib.sha256(data).hexdigest()
        item={'url':url,'expected_sha256':expected,'actual_sha256':actual,
              'matches':bool(expected and actual==expected),'bytes':len(data),'headers':headers}
        report['wheels'].append(item)
        print(json.dumps(item),flush=True)
    (out/'index-check.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
