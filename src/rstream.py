"""Streaming reader for R's XDR serialisation (RData / RDS), keeping memory low.

Big vectors are skipped unless the path to them matches KEEP_BIG; what is kept is
returned as a light tree of RObj nodes and pickled.
"""
import gzip, struct, sys, pickle, re, time
import numpy as np

NILVALUE, GLOBALENV, UNBOUND, MISSINGARG, BASENS = 254, 253, 252, 251, 250
NAMESPACESXP, PACKAGESXP, PERSISTSXP = 249, 248, 247
CLASSREF, GENERICREF, BCREPDEF, EMPTYENV, BASEENV = 246, 245, 244, 242, 241
ATTRLANG, ATTRLIST, ALTREP, REFSXP, BCREPREF = 240, 239, 238, 255, 243
NILSXP, SYMSXP, LISTSXP, CLOSXP, ENVSXP, PROMSXP, LANGSXP = 0, 1, 2, 3, 4, 5, 6
SPECIALSXP, BUILTINSXP, CHARSXP, LGLSXP, INTSXP, REALSXP, CPLXSXP = 7, 8, 9, 10, 13, 14, 15
STRSXP, DOTSXP, VECSXP, EXPRSXP, BCODESXP, EXTPTRSXP, WEAKREFSXP = 16, 17, 19, 20, 21, 22, 23
RAWSXP, S4SXP = 24, 25


class RObj:
    __slots__ = ("type", "value", "attr")

    def __init__(self, t, v=None, attr=None):
        self.type, self.value, self.attr = t, v, attr

    def a(self, name):
        return (self.attr or {}).get(name)

    def __repr__(self):
        v = self.value
        if isinstance(v, np.ndarray):
            v = f"ndarray{v.shape}"
        elif isinstance(v, list):
            v = f"list[{len(v)}]"
        return f"RObj({self.type}, {v}, attr={list((self.attr or {}).keys())})"


class Skipped:
    __slots__ = ("type", "n")

    def __init__(self, t, n):
        self.type, self.n = t, n

    def __repr__(self):
        return f"Skipped({self.type}, n={self.n})"


class Reader:
    def __init__(self, path, keep_big, big=2_000_000):
        self.f = gzip.open(path, "rb")
        self.f = __import__("io").BufferedReader(self.f, buffer_size=1 << 24)
        self.refs = []
        self.path = []
        self.keep_big = keep_big
        self.big = big
        self.nread = 0
        self.t0 = time.time()
        self.last = 0

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError
        self.nread += n
        if self.nread - self.last > 2e9:
            self.last = self.nread
            print(f"  {self.nread/1e9:.0f} GB  {time.time()-self.t0:.0f}s  at {'/'.join(map(str,self.path[-8:]))}",
                  flush=True)
        return b

    def skip(self, n):
        while n > 0:
            k = min(n, 1 << 26)
            self.read(k)
            n -= k

    def int(self):
        return struct.unpack(">i", self.read(4))[0]

    def length(self):
        n = self.int()
        if n == -1:
            hi, lo = self.int(), self.int()
            n = (hi << 32) + lo
        return n

    def string(self, n):
        return self.read(n).decode("utf-8", "replace")

    def keep(self):
        p = "/".join(map(str, self.path))
        return bool(re.search(self.keep_big, p))

    # ------------------------------------------------------------------
    def header(self):
        magic = self.read(5)
        assert magic in (b"RDX3\n", b"RDX2\n"), magic
        fmt = self.read(2)
        assert fmt == b"X\n", fmt
        ver = self.int(); self.int(); self.int()
        if ver == 3:
            self.string(self.int())

    def attrs(self):
        a = self.item()
        out = {}
        if isinstance(a, RObj) and a.type == LISTSXP:
            for k, v in a.value:
                out[k] = v
        return out

    def item(self):
        return self.dispatch(self.int())

    def dispatch(self, flags):
        t = flags & 0xFF
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if t == NILVALUE:
            return None
        if t in (EMPTYENV, BASEENV, GLOBALENV, UNBOUND, MISSINGARG, BASENS):
            return RObj(t)
        if t == REFSXP:
            i = flags >> 8
            if i == 0:
                i = self.int()
            return self.refs[i - 1]
        if t == PERSISTSXP:
            self.int(); n = self.int()
            v = RObj(t, [self.item() for _ in range(n)])
            self.refs.append(v)
            return v
        if t == SYMSXP:
            c = self.item()
            self.refs.append(c)
            return c
        if t in (PACKAGESXP, NAMESPACESXP):
            self.int(); n = self.int()
            v = RObj(t, [self.item() for _ in range(n)])
            self.refs.append(v)
            return v
        if t == ENVSXP:
            locked = self.int()
            env = RObj(t, {})
            self.refs.append(env)
            enclos = self.item(); frame = self.item(); hashtab = self.item(); attr = self.item()
            env.value = {"frame": frame, "hashtab": hashtab}
            return env
        if t in (LISTSXP, LANGSXP, CLOSXP, PROMSXP, DOTSXP, ATTRLANG, ATTRLIST):
            items = []
            first_attr = None
            while True:
                attr = self.attrs() if has_attr else None
                if first_attr is None:
                    first_attr = attr
                tag = self.item() if has_tag else None
                tagname = tag if isinstance(tag, str) else None
                self.path.append(tagname if tagname else "@")
                car = self.item()
                self.path.pop()
                items.append((tagname, car))
                # CDR: iterate instead of recursing when it is another pairlist cell
                flags = self.int()
                nt = flags & 0xFF
                if nt in (LISTSXP, LANGSXP, CLOSXP, PROMSXP, DOTSXP, ATTRLANG, ATTRLIST):
                    has_attr = bool(flags & (1 << 9)); has_tag = bool(flags & (1 << 10))
                    continue
                if nt == NILVALUE:
                    break
                rest = self.dispatch(flags)
                items.append(("..cdr", rest))
                break
            return RObj(LISTSXP if t in (LISTSXP, ATTRLIST) else t, items, first_attr)
        return self._item_with_flags(flags)

    def _item_with_flags(self, flags):
        t = flags & 0xFF
        has_attr = bool(flags & (1 << 9))
        has_tag = bool(flags & (1 << 10))
        if t == NILVALUE:
            return None
        if t == EXTPTRSXP:
            o = RObj(t)
            self.refs.append(o)
            self.item(); self.item()
        elif t == WEAKREFSXP:
            o = RObj(t)
            self.refs.append(o)
        elif t in (SPECIALSXP, BUILTINSXP):
            o = RObj(t, self.string(self.int()))
        elif t == CHARSXP:
            n = self.int()
            return None if n == -1 else self.string(n)
        elif t in (LGLSXP, INTSXP):
            n = self.length()
            if n > self.big and not self.keep():
                self.skip(4 * n); o = Skipped(t, n)
            else:
                o = RObj(t, np.frombuffer(self.read(4 * n), ">i4").astype(np.int32))
        elif t == REALSXP:
            n = self.length()
            if n > self.big and not self.keep():
                self.skip(8 * n); o = Skipped(t, n)
            else:
                o = RObj(t, np.frombuffer(self.read(8 * n), ">f8").astype(np.float64))
        elif t == CPLXSXP:
            n = self.length(); self.skip(16 * n); o = Skipped(t, n)
        elif t == RAWSXP:
            n = self.length()
            o = RObj(t, self.read(n)) if n < self.big else (self.skip(n) or Skipped(t, n))
        elif t == STRSXP:
            n = self.length()
            o = RObj(t, [self.item() for _ in range(n)])
        elif t in (VECSXP, EXPRSXP):
            n = self.length()
            vals = []
            for i in range(n):
                self.path.append(f"[{i}]")
                vals.append(self.item())
                self.path.pop()
            o = RObj(t, vals)
        elif t == S4SXP:
            o = RObj(t)
        elif t == BCODESXP:
            nreps = self.int()
            reps = [None] * nreps
            o = RObj(t, self._bc1(reps))
        elif t == ALTREP:
            info = self.item(); state = self.item(); attr = self.item()
            o = RObj(ALTREP, {"info": info, "state": state})
            o.attr = attr.value and dict(attr.value) if isinstance(attr, RObj) else None
            return self._altrep(o)
        else:
            raise ValueError(f"unknown SEXP type {t} at {'/'.join(map(str, self.path))}")
        if has_attr:
            if isinstance(o, RObj):
                # attributes are read with the S4 slot names on the path
                o.attr = self._attr_with_path()
            else:
                self._attr_with_path()
        return o

    def _attr_with_path(self):
        a = self.item()
        out = {}
        if isinstance(a, RObj) and a.type == LISTSXP:
            for k, v in a.value:
                out[k] = v
        return out

    def _altrep(self, o):
        info = o.value["info"]
        name = None
        try:
            name = info.value[0][1]
        except Exception:
            pass
        st = o.value["state"]
        if name == "compact_intseq" and isinstance(st, RObj):
            n, s, inc = st.value[:3]
            return RObj(INTSXP, (s + inc * np.arange(int(n))).astype(np.int32), o.attr)
        if name == "compact_realseq" and isinstance(st, RObj):
            n, s, inc = st.value[:3]
            return RObj(REALSXP, s + inc * np.arange(int(n)), o.attr)
        if name in ("wrap_integer", "wrap_real", "wrap_string", "wrap_logical", "wrap_list",
                    "deferred_string") and isinstance(st, RObj):
            inner = st.value[0][1] if st.type == LISTSXP else st.value[0]
            if isinstance(inner, RObj) and o.attr:
                inner.attr = {**(inner.attr or {}), **o.attr}
            if name == "deferred_string" and isinstance(inner, RObj) and inner.type in (INTSXP, REALSXP):
                inner = RObj(STRSXP, [str(int(x)) if float(x).is_integer() else repr(float(x))
                                      for x in inner.value], inner.attr)
            return inner
        return o

    def _bc1(self, reps):
        code = self.item()
        n = self.int()
        consts = []
        for _ in range(n):
            t = self.int()
            if t == BCODESXP:
                consts.append(self._bc1(reps))
            elif t in (LANGSXP, LISTSXP, BCREPDEF, BCREPREF, ATTRLANG, ATTRLIST):
                consts.append(self._bclang(t, reps))
            else:
                consts.append(self.item())
        return {"code": code, "consts": len(consts)}

    def _bclang(self, t, reps):
        if t == BCREPREF:
            return reps[self.int()]
        if t in (BCREPDEF, LANGSXP, LISTSXP, ATTRLANG, ATTRLIST):
            pos = -1
            if t == BCREPDEF:
                pos = self.int()
                t = self.int()
            o = RObj(t, [])
            if pos >= 0:
                reps[pos] = o
            if t in (ATTRLANG, ATTRLIST):
                self.item()
            self.item()  # tag
            car = self._bclang(self.int(), reps)
            cdr = self._bclang(self.int(), reps)
            o.value = [car, cdr]
            return o
        return self.item()


def main(path, out, keep_big):
    r = Reader(path, keep_big)
    r.header()
    top = r.item()
    print("done in", round(time.time() - r.t0), "s,", r.nread / 1e9, "GB")
    with open(out, "wb") as fh:
        pickle.dump(top, fh, protocol=5)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else r"^seurat_object_whole_clustering/assays/\[\d+\]/counts|/image$")
