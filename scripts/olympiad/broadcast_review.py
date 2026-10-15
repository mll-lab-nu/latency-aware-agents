#!/usr/bin/env python3
"""Serve exported broadcast review videos with byte ranges for immediate chapter seeking.

python3 scripts/olympiad/broadcast_review.py --directory /path/to/public/review --port 8058
The directory must contain only files intended for public review.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import re


class ReviewHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        self.remaining = None
        value = self.headers.get('Range')
        if not value:
            return super().send_head()
        path = self.translate_path(self.path)
        try:
            f = open(path, 'rb')
        except (OSError, ValueError):
            return super().send_head()
        f.seek(0, 2)
        size = f.tell()
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', value.strip())
        if not match or not any(match.groups()):
            # Ignore unsupported/multiple ranges as permitted by HTTP.
            f.close()
            return super().send_head()
        left, right = match.groups()
        start = int(left) if left else max(0, size - int(right))
        end = min(size - 1, int(right)) if left and right else size - 1
        if start >= size or start > end:
            f.close()
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        f.seek(start)
        self.remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(path))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.remaining))
        self.end_headers()
        return f

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def copyfile(self, source, outputfile):
        try:
            if self.remaining is None:
                return super().copyfile(source, outputfile)
            while self.remaining:
                chunk = source.read(min(self.remaining, 256 * 1024))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # normal when a viewer seeks elsewhere or closes the page


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', required=True)
    ap.add_argument('--port', type=int, default=8058)
    args = ap.parse_args()
    ThreadingHTTPServer(('0.0.0.0', args.port), partial(ReviewHandler, directory=args.directory)).serve_forever()
