"""Verify host orchestration without Docker, service restarts or accelerator work."""
import json
import os
from pathlib import Path
import subprocess
import sys

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/checks/run_elasticsearch_memory_acceptance.sh'


def invoke(tmp_path, *args, fail=False):
    executable=tmp_path/'docker'
    executable.write_text('#!'+sys.executable+'\n'+'''import os,sys,json
from pathlib import Path
p=Path(os.environ['CALL_LOG'])
with p.open('a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('FAIL_TRIAL')=='1' and any(a.endswith('check_measured_watermarks.py') for a in sys.argv):sys.exit(2)
print('OUTPUT=/workspace/dev/rag_ljy/experiment_outputs/checks/fake/'+str(len(p.read_text().splitlines())))
''')
    executable.chmod(0o755)
    env={**os.environ,'PATH':str(tmp_path)+os.pathsep+os.environ['PATH'],'CALL_LOG':str(tmp_path/'calls'),'FAIL_TRIAL':str(int(fail))}
    result=subprocess.run(['bash',str(SCRIPT),*args],env=env,capture_output=True,text=True)
    calls=[json.loads(l) for l in (tmp_path/'calls').read_text().splitlines()]
    return result,calls


def test_preview_does_not_restart_or_run_load(tmp_path):
    result,calls=invoke(tmp_path)
    assert result.returncode==0 and len(calls)==1
    assert '--run' not in calls[0]
    assert calls[0][-2:]==['--retrieval-backend','elasticsearch']


def test_all_three_trials_and_summary_use_es(tmp_path):
    result,calls=invoke(tmp_path,'--run')
    assert result.returncode==0 and len(calls)==8
    for call in calls:assert 'elasticsearch' in call
    trials=[c for c in calls if 'scripts/checks/check_measured_watermarks.py' in c]
    assert [c[c.index('--modes')+1] for c in trials]==['burst','staggered','continuous']
    assert 'scripts/checks/summarize_measured_acceptance.py' in calls[-1]
    assert '--stale' in calls[-1] and '--deployments' in calls[-1]


def test_incomplete_trial_stops_suite_and_prevents_pass_summary(tmp_path):
    result,calls=invoke(tmp_path,'--run',fail=True)
    assert result.returncode==2 and len(calls)==2
    assert not any('scripts/checks/summarize_measured_acceptance.py' in c for c in calls)


def test_generation_suite_requires_generation_in_preflight_loads_and_summary(tmp_path):
    result,calls=invoke(tmp_path,'--run','--generate','--max-new-tokens','128')
    assert result.returncode==0 and len(calls)==9
    checks=[c for c in calls if 'scripts/checks/check_measured_watermarks.py' in c]
    assert len(checks)==4 and '--run' not in checks[0]
    assert all('--generate' in c and c[c.index('--max-new-tokens')+1]=='128' for c in checks)
    assert '--generate' in calls[-1]
