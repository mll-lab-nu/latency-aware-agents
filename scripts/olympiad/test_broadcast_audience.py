"""Spectator names and facts must stay attached to the right model, not the camera angle."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent/'broadcast'))
import audience as a
names={'A':'Opus 5.5','B':'Astra'}
assert a.referee_reason("A's shot hit its own half",names,'tabletennis') == "Opus 5.5's return bounced on its own side of the table"
assert a.referee_reason('B missed the ball',names,'tennis')=='Astra missed the ball'
assert a.referee_reason('Astra missed the ball',names,'tennis')=='Astra missed the ball'
assert a.names({'A':'Luna','B':'Luna'})=={'A':'Luna · Blue','B':'Luna · Orange'}
assert a.command_activity("echo 'game ready'") is None
assert a.command_activity('sleep 30') is None
assert a.command_activity('unknown-tool --secret foo') is None
assert a.command_activity('cd /app && python3 output/player.py & sleep 1')=='Running a control script'
assert a.command_activity('timeout 10 game status')=='Checking the score and match state'
assert a.command_activity('python3 -m py_compile output/player.py')=='Checking the control program for syntax errors'
assert a.command_activity('game ready')=='Reporting ready for the next point'
assert a.caption_lines({'kind':'Review','text':'The return was too short → aim deeper'})==['Assessment: The return was too short','Next attempt: aim deeper']
assert a.caption_lines({'kind':'Says','text':'I will win'})==['I will win']  # remains model speech, not a match result
assert a.excerpt('A complete sentence. '+('long '*80),70)=='A complete sentence. …'
print('PASS: correct model fault attribution, same-model identity, command summaries, attributed review clauses')

# A stalled language-model request cannot hold up the caller or the rendering clock.
import tempfile,threading,time
from review_editor import ReviewEditor
with tempfile.TemporaryDirectory() as tmp:
    editor=ReviewEditor(tmp,enabled=True,api_key='test-only')
    entered,release=threading.Event(),threading.Event()
    def slow(text,sport):
        entered.set();release.wait(3)
        return {'assessment':'My return landed short.','plan':'Aim deeper next time.'}
    editor.rewrite=slow
    t=time.monotonic();assert editor.request('why → plan','tennis') is None
    assert time.monotonic()-t<.1 and entered.wait(1)
    t=time.monotonic();assert editor.request('why → plan','tennis') is None
    assert time.monotonic()-t<.1
    release.set();editor.jobs.join()
    assert editor.request('why → plan','tennis')['plan']=='Aim deeper next time.'
    offline=ReviewEditor(tmp)
    assert offline.request('why → plan','tennis')['assessment']=='My return landed short.'
    try:editor.validate({'assessment':'x'*111,'plan':'p'})
    except ValueError:pass
    else:raise AssertionError('overlong rewrite accepted')
print('PASS: background rewrite returns immediately, deduplicates, caches and validates output')
with tempfile.TemporaryDirectory() as tmp:
    editor=ReviewEditor(tmp,enabled=True,api_key='test-only')
    def unavailable(*args):raise OSError('offline')
    editor.rewrite=unavailable
    editor.request('review','tennis');editor.jobs.join()
    assert editor.failures==1 and not editor.pending
    assert editor.request('review','tennis') is None and editor.jobs.empty()
print('PASS: failed summaries fall back without blocking or retry storms')
