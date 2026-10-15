"""Renderer scheduling and EGL failure fallback, without a GPU or Docker daemon."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import tempfile

spec = importlib.util.spec_from_file_location('broadcast_quality_test', Path(__file__).with_name('broadcast.py'))
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

def match(sport, streams=1, programs=1):
    return NS(sport=sport, turn_based=sport == 'billiards',
              stream=NS(clients=streams), program=NS(clients=programs))

with patch.object(b.os, 'getloadavg', return_value=(0, 0, 0)):
    b.STATE['matches'] = {'tt': match('tabletennis', 0), 'soc': match('soccer'), 'bill': match('billiards', 0)}
    b.LIVE.rt = b.SOC.rt = b.BILL.rt = .04
    assert abs(b.target_fps() - 21.25) < .001  # soccer/billiards don't consume this pool
    assert abs(b.target_fps(b.SOC) - 10.625) < .001  # plain + program soccer BOTH count
    b.STATE['matches']['tt'].program.clients = 12
    assert abs(b.target_fps() - 21.25) < .001  # multiple viewers share one render
    b.STATE['matches']['tt'].stream.clients = 1
    assert abs(b.target_fps() - 10.625) < .001

# A missing/broken EGL image must use the original CPU image without --gpus.
with tempfile.TemporaryDirectory() as tmp, \
     patch.object(b, 'CACHE', Path(tmp)), patch.object(b, 'RENDER_BACKEND', 'egl'), \
     patch.object(b, 'probe_egl', return_value=''), patch.object(b.subprocess, 'run'), \
     patch.object(b.subprocess, 'Popen') as popen, patch.object(b.Helper, '_recv', return_value=('ok', 'ready')):
    h = b.Helper('quality-test', 2)
    h.start()
    args = popen.call_args.args[0]
    assert '--gpus' not in args and b.IMAGE in args and 'MUJOCO_GL=osmesa' in args
    assert h.backend == 'osmesa'

print('PASS: isolated per-helper budgets, soccer plain stream, shared viewers, CPU fallback')

# An unread MJPEG socket must release its viewer count instead of queuing video indefinitely.
import socket
import threading
import time
st = NS(cond=threading.Condition(), clients=0, seq=1, jpeg=b'x' * (2 * 1024 * 1024), last_client=0)
m = NS(sport='tabletennis', mode='duel', hdr=True, turn_based=False, gone=False, program=st)
with patch.object(b.H, 'match', return_value=m), patch.object(b, 'get_program', return_value=st):
    server = b.Server(('127.0.0.1', 0), b.H)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.socket() as client:
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
            client.connect(server.server_address)
            client.sendall(b'GET /program/test HTTP/1.0\r\n\r\n')
            deadline = time.monotonic() + 2
            while st.clients == 0 and time.monotonic() < deadline:
                time.sleep(.01)
            assert st.clients == 1, 'stream never subscribed'
            deadline = time.monotonic() + 4
            while st.clients and time.monotonic() < deadline:
                time.sleep(.02)
            assert st.clients == 0, 'stalled socket kept a renderer subscription'
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
print('PASS: stalled MJPEG client is disconnected and unsubscribed')
