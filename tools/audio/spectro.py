#!/usr/bin/env python3
"""Spectrogram PNG from a 16-bit WAV (numpy + PIL only).
usage: python3 -I tools/audio/spectro.py in.wav out.png [fmax=6000] [log=0|1] [width=1400]
"""
import sys, struct
import numpy as np
from PIL import Image, ImageDraw


def read_wav(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'RIFF' and data[8:12] == b'WAVE'
    pos = 12
    fmt = None
    pcm = None
    while pos + 8 <= len(data):
        cid = data[pos:pos + 4]
        size = struct.unpack('<I', data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if cid == b'fmt ':
            fmt = struct.unpack('<HHIIHH', body[:16])
        elif cid == b'data':
            pcm = body
        pos += 8 + size + (size & 1)
    _, ch, fs, _, _, bits = fmt
    assert bits == 16
    x = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
    x = x.reshape(-1, ch)
    return x, fs


def colormap(v):
    # v in 0..1 -> magma-like
    stops = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=np.float32)
    t = np.clip(v, 0, 1) * (len(stops) - 1)
    i = np.clip(t.astype(int), 0, len(stops) - 2)
    f = (t - i)[..., None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    fmax = float(sys.argv[3]) if len(sys.argv) > 3 else 6000.0
    logf = len(sys.argv) > 4 and sys.argv[4] == '1'
    width = int(sys.argv[5]) if len(sys.argv) > 5 else 1400
    x, fs = read_wav(src)
    mono = x.mean(axis=1)
    N = 4096
    hop = max(64, int(len(mono) / width))
    win = np.hanning(N).astype(np.float32)
    frames = max(1, (len(mono) - N) // hop)
    spec = np.empty((frames, N // 2 + 1), dtype=np.float32)
    for i in range(frames):
        seg = mono[i * hop:i * hop + N] * win
        spec[i] = np.abs(np.fft.rfft(seg)) * (4.0 / N)
    db = 20 * np.log10(spec + 1e-9)
    H = 520
    freqs = np.fft.rfftfreq(N, 1 / fs)
    if logf:
        fy = np.geomspace(30, fmax, H)
    else:
        fy = np.linspace(0, fmax, H)
    idx = np.clip(np.searchsorted(freqs, fy), 0, len(freqs) - 1)
    img = db[:, idx].T[::-1]
    top = max(-20.0, float(np.percentile(db, 99.9)))
    v = (img - (top - 70)) / 70.0
    rgb = colormap(v)
    im = Image.fromarray(rgb, 'RGB')
    d = ImageDraw.Draw(im)
    dur = len(mono) / fs
    for s in range(0, int(dur) + 1):
        xpx = int(s / dur * im.width)
        d.line([(xpx, H - 8), (xpx, H)], fill=(255, 255, 255))
        d.text((xpx + 2, H - 12), f'{s}s', fill=(255, 255, 255))
    marks = [50, 100, 200, 500, 1000, 2000, 4000, 8000] if logf else list(range(0, int(fmax) + 1, 1000))
    for m in marks:
        if m > fmax or (logf and m < 30):
            continue
        y = H - 1 - int(np.searchsorted(fy, m))
        d.line([(0, y), (6, y)], fill=(255, 255, 255))
        d.text((8, max(0, y - 5)), f'{m}', fill=(255, 255, 255))
    im.save(dst)
    print(f'{dst}: {im.width}x{im.height}, {dur:.2f}s, peak {np.abs(x).max():.3f}')


if __name__ == '__main__':
    main()
