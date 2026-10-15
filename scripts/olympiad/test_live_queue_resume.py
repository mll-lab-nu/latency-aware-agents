"""Check that resuming two-host results never replays a completed configuration."""
import importlib.util,json,subprocess,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[2]
p=root/'scripts/olympiad/run_round.py'
s=importlib.util.spec_from_file_location('live_queue_check',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
ms=r.matches(1,'tabletennis',10,['astra','luna'])
with tempfile.TemporaryDirectory() as td:
 local=Path(td)/'local';other=Path(td)/'other';local.mkdir();other.mkdir()
 good=ms[:-1];last=ms[-1]
 def row(m,valid):return {k:v for k,v in m.items() if k!='cmd'}|{'reward':{'valid_trial':valid}}
 local_log=local/'tabletennis-round1.jsonl'
 remote_log=other/'tabletennis-round1.jsonl'
 local_log.write_text(''.join(json.dumps(row(m,1))+'\n' for m in good))
 remote_log.write_text(json.dumps(row(last,0))+'\n')
 cmd=[sys.executable,str(p),'1','--sport','tabletennis','--players','astra,luna','--lanes','astra,luna','--missing','--results',str(local),'--completed-results',str(other),'--concurrency','1','--dry']
 out=subprocess.check_output(cmd,text=True)
 assert out.endswith('1 matches\n'),out
 assert f"--a {last['a']} --b {last['b']}" in out
 before=local_log.read_bytes()
 remote_log.write_text(remote_log.read_text()+json.dumps(row(last,1))+'\n')
 assert subprocess.check_output(cmd,text=True)=='0 matches\n'
 assert local_log.read_bytes()==before
 print('PASS: merge host results, retry invalid configuration, skip valid configuration, preserve ledger')
