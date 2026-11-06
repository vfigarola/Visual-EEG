"""Minimal MATLAB v5 .mat reader (numpy only) for environments without scipy.
Returns nested dicts for structs, numpy arrays for numeric/char, lists for cells."""
import zlib, struct, numpy as np

_DT = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 9: 'f8', 12: 'i8', 13: 'u8', 16: 'u1', 17: 'u2', 18: 'u4'}
_CLASS = {1: 'cell', 2: 'struct', 3: 'object', 4: 'char', 5: 'sparse', 6: 'double', 7: 'single', 8: 'int8', 9: 'uint8',
          10: 'int16', 11: 'uint16', 12: 'int32', 13: 'uint32', 14: 'int64', 15: 'uint64'}


def _tag(buf, off, endian):
    t, n = struct.unpack_from(endian + 'II', buf, off)
    if t >> 16:  # small data element
        return t & 0xFFFF, t >> 16, off + 4, off + 8
    return t, n, off + 8, off + 8 + ((n + 7) // 8) * 8


def _elem(buf, off, endian):
    t, n, d, nxt = _tag(buf, off, endian)
    return t, buf[d:d + n], nxt


def _num(t, raw, endian):
    return np.frombuffer(raw, dtype=endian + _DT[t]) if t in _DT else np.frombuffer(raw, 'u1')


def _matrix(buf, endian, skip_big=None):
    off = 0
    _, flags, off = _elem(buf, off, endian)
    f = _num(5 if len(flags) == 8 else 6, flags, endian)
    cls = int(f[0]) & 0xFF
    complex_ = bool(int(f[0]) & 0x800)
    t, dims, off = _elem(buf, off, endian)
    dims = tuple(int(x) for x in _num(t, dims, endian))
    t, name, off = _elem(buf, off, endian)
    name = bytes(name).decode('latin1')
    c = _CLASS.get(cls, cls)
    if c == 'struct' or c == 'object':
        if c == 'object':
            _, _, off = _elem(buf, off, endian)
        t, fl, off = _elem(buf, off, endian)
        flen = int(_num(t, fl, endian)[0])
        t, fn, off = _elem(buf, off, endian)
        names = [bytes(fn[i:i + flen]).split(b'\0')[0].decode() for i in range(0, len(fn), flen)]
        n = int(np.prod(dims)) if dims else 0
        out = []
        for _ in range(n):
            rec = {}
            for k in names:
                t, sub, off = _elem(buf, off, endian)
                rec[k] = _matrix(sub, endian, skip_big)[1] if len(sub) else None
            out.append(rec)
        return name, (out[0] if n == 1 else {'__structarray__': out, 'dims': dims})
    if c == 'cell':
        n = int(np.prod(dims))
        out = []
        for _ in range(n):
            t, sub, off = _elem(buf, off, endian)
            out.append(_matrix(sub, endian, skip_big)[1] if len(sub) else None)
        return name, {'__cell__': out, 'dims': dims}
    if c == 'sparse':
        return name, {'__sparse__': dims}
    t, re_, off = _elem(buf, off, endian)
    if skip_big and len(re_) > skip_big:
        return name, {'__skipped__': True, 'class': c, 'dims': dims}
    a = _num(t, re_, endian)
    if complex_:
        t, im, off = _elem(buf, off, endian)
        a = a + 1j * _num(t, im, endian)
    if c == 'char':
        s = ''.join(chr(x) for x in a)
        if len(dims) == 2 and dims[0] > 1:
            return name, [s[i::dims[0]] for i in range(dims[0])]
        return name, s
    return name, a.reshape(dims, order='F') if dims and np.prod(dims) == a.size else a


def loadmat(path, skip_big=None, only=None):
    with open(path, 'rb') as fh:
        data = fh.read()
    endian = '<' if data[126:128] == b'IM' else '>'
    off, out = 128, {}
    while off + 8 <= len(data):
        t, n, d, nxt = _tag(data, off, endian)
        raw = data[d:d + n]
        if t == 15:
            nxt = d + n  # compressed elements are not 8-byte padded
            raw = zlib.decompress(raw)
            t, n2, d2, _ = _tag(raw, 0, endian)
            raw = raw[d2:d2 + n2]
        if t == 14:
            name, val = _matrix(raw, endian, skip_big)
            if only is None or name in only:
                out[name] = val
        off = nxt
    return out


def describe(v, ind='', depth=0, maxd=3):
    if isinstance(v, dict) and '__structarray__' in v:
        print(f"{ind}struct array dims={v['dims']}")
        if v['__structarray__'] and depth < maxd:
            describe(v['__structarray__'][0], ind + '  ', depth + 1, maxd)
    elif isinstance(v, dict) and '__cell__' in v:
        print(f"{ind}cell dims={v['dims']}")
    elif isinstance(v, dict) and ('__skipped__' in v or '__sparse__' in v):
        print(f"{ind}{v}")
    elif isinstance(v, dict):
        for k, x in v.items():
            if isinstance(x, np.ndarray):
                print(f"{ind}{k}: {x.dtype} {x.shape} {x.ravel()[:6] if x.size else ''}")
            elif isinstance(x, (str, list)):
                print(f"{ind}{k}: {repr(x)[:100]}")
            else:
                print(f"{ind}{k}:")
                if depth < maxd:
                    describe(x, ind + '  ', depth + 1, maxd)
    else:
        print(ind, type(v), repr(v)[:100])
