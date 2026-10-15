import importlib.util,threading,time
from pathlib import Path
from types import SimpleNamespace
p=Path(__file__).resolve().with_name('broadcast.py')
sp=importlib.util.spec_from_file_location('broadcast_smooth_test',p);b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
entered=threading.Event();release=threading.Event()
def slow_read():entered.set();release.wait(3)
steps=[];timestamps=[]
def render(helper,bm,req):
 steps.append(req[2]['k']+(req[3]['k']-req[2]['k'])*req[4] if req[3] else req[2]['k']);timestamps.append(time.monotonic());return b'jpg',.001
bm=SimpleNamespace(id='test',ks=list(range(0,502,2)),recs=[{'k':k} for k in range(0,502,2)],simdir_ok=True,slowdown=2.,dt=.01,hdr={'every':2},gone=None,tail=slow_read,phase_at=lambda k:'run',live_overlay=lambda *a,**kw:{},trail=lambda i:[])
b.helper_call=render;b.target_fps=lambda:24
st=b.Stream(bm);bm.stream=st;st.clients=1;st.start();assert entered.wait(1)
time.sleep(.8)
# A blocked Docker/SSH read must not freeze the live render clock.
assert len(steps)>=12,(len(steps),steps)
assert max(t2-t1 for t1,t2 in zip(timestamps,timestamps[1:]))<.15
speed=(steps[-1]-steps[0])/(timestamps[-1]-timestamps[0]);assert abs(speed-50)<1,speed
st.clients=0;st.last_client=time.time()-20;release.set();st.join(2);assert not st.is_alive()
print('PASS: continuous renders during blocked log read; x2 advances 50 steps/s; clean reader shutdown')
