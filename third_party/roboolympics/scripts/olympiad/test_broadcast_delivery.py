"""Bandwidth tiers share bounded work, preserve the original and never upscale."""
import io
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent / 'broadcast'))
from delivery import Delivery, Rendition

def jpeg(size):
    b=io.BytesIO();Image.new('RGB',size,(34,100,156)).save(b,'JPEG');return b.getvalue()

original=jpeg((1920,1080));d=Delivery()
assert d.frame('original',1,original)==(1,original)
for profile,size in [('720p',(1280,720)),('540p',(960,540)),('360p',(640,360))]:
    seq,out=d.frame(profile,1,original)
    assert Image.open(io.BytesIO(out)).size==size
    assert len(out)<len(original)
    with ThreadPoolExecutor(max_workers=8) as pool:
        copies=list(pool.map(lambda _:d.frame(profile,1,original),range(16)))
    assert all(x[1] is out for x in copies), 'viewers did not share cached bytes'
    assert d.frame(profile,0,b'invalid')[1] is out, 'stale source replaced latest frame'
with patch('delivery.time.monotonic',return_value=100) as clock:
    r=Rendition((640,360),12,60);first=r.frame(1,original)
    clock.return_value=100.03
    assert r.frame(2,original)==first, 'tier frame cap ignored'
    clock.return_value=100.10
    assert r.frame(3,original)[0]==3
small=jpeg((480,270));out=Delivery().frame('720p',1,small)[1]
assert Image.open(io.BytesIO(out)).size==(480,270)
with patch('delivery.time.monotonic') as clock:
    r=Rendition((640,360),24,60);seen=set()
    for i in range(60):
        clock.return_value=100+i/30
        seen.add(r.frame(i,original)[0])
    assert 47<=len(seen)<=49, ('24 FPS cadence drifted on a 30 FPS source',len(seen))
print('PASS: shared per-tier JPEGs, monotonic latest frame, FPS cap, untouched original, no upscale')
