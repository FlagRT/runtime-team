"""Join measured ATen overloads to source registrations; availability is not approval."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import yaml


def registrations(source, gems=False):
    names=set()
    for node in ast.walk(ast.parse(source)):
        if gems and isinstance(node,ast.Tuple) and len(node.elts)>=2:
            if isinstance(node.elts[0],ast.Constant) and isinstance(node.elts[0].value,str) and isinstance(node.elts[1],ast.Name):
                names.add(node.elts[0].value)
        if not gems and isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='impl':
            if node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
                names.add(node.args[0].value)
    return names


def main():
    p=argparse.ArgumentParser()
    for n in ('root','flagop','gems','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    with zipfile.ZipFile(a.gems) as z:gems=registrations(z.read('flag_gems/__init__.py').decode(),True)
    catalog=yaml.safe_load((a.flagop/'ops/catalog.yaml').read_text())
    flagop={}
    for entry in catalog['operators']:
        path=a.flagop/'ops'/entry['directory']/'register.py'
        if not path.exists():continue
        for name in registrations(path.read_text()):
            flagop[name]=dict(directory=entry['directory'],platforms=entry.get('platforms',{}))
    summary=dict(flagop_commit='2591d081188d88c2ef75c2798f263186ec3837e2',
        gems_wheel_sha256=hashlib.sha256(a.gems.read_bytes()).hexdigest(),
        matching='Exact ATen overload; default overload omits .default in registration. Source only, not execution approval.',
        flagop_catalog_note='Incremental subset; absent is not equivalent to unsupported.',models={})
    lines=['# 三模型实际算子清单（2026-11-10）','',
        'PPU eager原生训练各1步；前向、反向、优化器分别记录。Python入口与ATen可能不同。',
        'OneRec开启梯度检查点，其反向阶段包含重算的前向调用。完整形状/输入参数/模块样本见对应原始JSON。',
        'FlagGems列仅匹配固定wheel的注册表；FlagOP列仅匹配固定源码register.py，不自动授予本环境资格。','']
    for model in ('deepfm','qwen','onerec'):
        path=a.root/(model+'.json');r=json.loads(path.read_text())
        assert r['status']=='PASS' and r['optimizer_steps']==1 and not r['inventory']['signatures_truncated']
        rows=r['inventory']['signatures'];counts=Counter()
        for row in rows:
            if row['layer']=='aten':counts[row['phase'],row['operator']]+=row['count']
        details=[]
        for (phase,op),count in sorted(counts.items()):
            key=op.removeprefix('aten.').removesuffix('.default')
            match=flagop.get(key)
            details.append(dict(phase=phase,operator=op,calls=count,flaggems_registered=key in gems,
                flagop=match,signatures=[dict(inputs=x['inputs'],output=x['output'],count=x['count'],modules=x['modules'],
                    module_samples_truncated=x['module_samples_truncated']) for x in rows
                    if x['layer']=='aten' and x['phase']==phase and x['operator']==op]))
        summary['models'][model]=dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            parameter_count=r['parameter_count'],settings=r['settings'],loss=r['loss'],
            phase_counts={phase:dict(unique=sum(p==phase for p,o in counts),calls=sum(n for (p,o),n in counts.items() if p==phase))
                          for phase in ('forward','backward','optimizer')},operators=details)
        lines+=['## '+model,'',f"参数量：{r['parameter_count']}；配置：`{json.dumps(r['settings'],ensure_ascii=False)}`。",'']
        for phase in ('forward','backward','optimizer'):
            lines+=['### '+phase,'','| 实际ATen入口 | 调用次数 | FlagGems注册 | FlagOP目录 |','|---|---:|---|---|']
            for d in details:
                if d['phase']!=phase:continue
                lines.append(f"| `{d['operator']}` | {d['calls']} | {'有' if d['flaggems_registered'] else '未匹配'} | {d['flagop']['directory'] if d['flagop'] else '未匹配'} |")
            lines.append('')
    (a.output/'inventory-comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (a.output/'算子清单-20261110.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v['phase_counts'] for k,v in summary['models'].items()},indent=2))


if __name__=='__main__':main()
