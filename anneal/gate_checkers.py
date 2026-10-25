"""Validity checkers for the fixed 200's problems where more than one output is right (25 Oct 2026).

CHECKERS maps a Codeforces id (e.g. 1594C) to check(input_text, output_text, expected_text) -> bool: True only when
the output is a right answer to the input by the statement. Where the statement asks for the best value (fewest steps,
largest set, ...), the best value is read from the expected output, so a checker is used only after
scripts/build_gate200.py --check found it fit: it accepts the expected output of every kept test, and on no test does
it reject 2 or more of the problem's accepted solutions. A checker reads the whole output and checks all of it at every
input size, in linear or near-linear time; anything it can't parse is a wrong answer (anneal/evalsets.case_match counts
an exception as one).

- Integers in the program's output are read as Codeforces' testlib reads them: an optional minus sign and ASCII
  digits, no leading zeros, no "-0". Python's default 4,300-digit limit on int() stays: no answer here needs more,
  and a longer number is a wrong answer.
- YES and NO match in any letter case; extra tokens after the answer make it wrong, as on Codeforces.
- Problems that say "print any" but have exactly one right answer (1552B, 1560B), and the real-number problems
  (1555B, 1557A, 1575B, compared within the statement's tolerance), are judged by evalsets.case_match's plain compare,
  not here.
- Standard library only. checker_sha256 (module_sha256) in each item's judge spec pins this file: an item built with
  one version of it is never judged with another.
"""
import decimal
import functools
import hashlib
from bisect import bisect_right
from collections import Counter
from itertools import compress


@functools.lru_cache(maxsize=1)
def module_sha256():
    """sha256 of this file's bytes; scripts/build_gate200.py records it in every checker item."""
    with open(__file__, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------- reading

def sint(x):
    """testlib-style integer: optional '-', ASCII digits, no leading zeros, no -0."""
    neg = x[:1] == '-'
    d = x[1:] if neg else x
    if not d or not d.isascii() or not d.isdigit() or (d[0] == '0' and (len(d) > 1 or neg)):
        raise ValueError(x)
    v = int(d)
    return -v if neg else v


class R:
    """Token reader. strict=True (program output) uses testlib integer rules."""
    __slots__ = ('t', 'i', 'cv')

    def __init__(self, text, strict=False):
        self.t = text.split()
        self.i = 0
        self.cv = sint if strict else int

    def s(self):
        x = self.t[self.i]
        self.i += 1
        return x

    def int(self):
        x = self.t[self.i]
        self.i += 1
        return self.cv(x)

    def ints(self, k):
        if k < 0 or self.i + k > len(self.t):
            raise IndexError('short')
        cv = self.cv
        out = [cv(x) for x in self.t[self.i:self.i + k]]
        self.i += k
        return out

    def skip(self, k):
        if k < 0 or self.i + k > len(self.t):
            raise IndexError('short')
        self.i += k

    def word(self):
        return self.s().upper()

    def done(self):
        return self.i == len(self.t)


def _ioe(i, o, e):
    return R(i), R(o, True), R(e)


def _is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71):
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


# ---------------------------------------------------------------- checkers

def c1549A(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        p = I.int()
        a, b = O.int(), O.int()
        if not (2 <= a < b <= p and p % a == p % b):
            return False
    return O.done()


def c1551A(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        c1, c2 = O.int(), O.int()
        e1, e2 = E.int(), E.int()
        if c1 < 0 or c2 < 0 or c1 + 2 * c2 != n or abs(c1 - c2) != abs(e1 - e2):
            return False
    return O.done()


def c1551B2(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, k = I.int(), I.int()
        a = I.ints(n)
        c = O.ints(n)
        best = n - E.ints(n).count(0)
        cnt = [0] * (k + 1)
        seen = set()
        for x, v in zip(c, a):
            if x < 0 or x > k:
                return False
            if x:
                cnt[x] += 1
                key = v * (k + 1) + x
                if key in seen:
                    return False
                seen.add(key)
        painted = sum(cnt)
        if painted != best or (painted and min(cnt[1:]) != max(cnt[1:])):
            return False
    return O.done()


def c1552E(i, o, e):
    I, O, _ = _ioe(i, o, e)
    n, k = I.int(), I.int()
    N = n * k
    c = [0] + I.ints(N)
    lim = -(-n // (k - 1))
    d = [0] * (N + 2)
    for col in range(1, n + 1):
        a, b = O.int(), O.int()
        if not (1 <= a < b <= N) or c[a] != col or c[b] != col:
            return False
        d[a] += 1
        d[b + 1] -= 1
    run = 0
    for x in d:
        run += x
        if run > lim:
            return False
    return O.done()


def c1554D(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        s = O.s()
        if len(s) != n or not s.isascii() or not s.isalpha() or not s.islower():
            return False
        if not _sam_all_odd(s):
            return False
    return O.done()


def _sam_all_odd(s):
    """Suffix automaton: every distinct substring occurs an odd number of times."""
    ln = [0]
    link = [-1]
    nxt = [{}]
    cnt = [0]
    last = 0
    for ch in s:
        cur = len(ln)
        ln.append(ln[last] + 1)
        link.append(0)
        nxt.append({})
        cnt.append(1)
        p = last
        while p != -1 and ch not in nxt[p]:
            nxt[p][ch] = cur
            p = link[p]
        if p != -1:
            q = nxt[p][ch]
            if ln[p] + 1 == ln[q]:
                link[cur] = q
            else:
                cl = len(ln)
                ln.append(ln[p] + 1)
                link.append(link[q])
                nxt.append(dict(nxt[q]))
                cnt.append(0)
                while p != -1 and nxt[p].get(ch) == q:
                    nxt[p][ch] = cl
                    p = link[p]
                link[q] = cl
                link[cur] = cl
        last = cur
    buckets = [[] for _ in range(len(s) + 1)]
    for v in range(1, len(ln)):
        buckets[ln[v]].append(v)
    for L in range(len(s), 0, -1):
        for v in buckets[L]:
            if not cnt[v] & 1:
                return False
            cnt[link[v]] += cnt[v]
    return True


def c1558C(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        a = I.ints(n)
        em = E.int()
        if em != -1:
            E.skip(em)
        m = O.int()
        if em == -1 or m == -1:
            if m != em:
                return False
            continue
        if not 0 <= m <= 5 * n // 2:
            return False
        for p in O.ints(m):
            if not (1 <= p <= n and p & 1):
                return False
            a[:p] = a[p - 1::-1]
        if any(a[j] != j + 1 for j in range(n)):
            return False
    return O.done()


def c1559B(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        s = I.s()
        x = O.s()
        y = E.s()
        if len(x) != n or any(ch not in 'BR' for ch in x):
            return False
        if any(c != '?' and c != d for c, d in zip(s, x)):
            return False
        bad = sum(x[j] == x[j + 1] for j in range(n - 1))
        if bad != sum(y[j] == y[j + 1] for j in range(len(y) - 1)):
            return False
    return O.done()


def c1559C(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        a = I.ints(n)
        ev = E.int()
        if ev != -1:
            E.skip(n)
        x = O.int()
        if ev == -1 or x == -1:
            if x != ev:
                return False
            continue
        path = [x] + O.ints(n)
        seen = bytearray(n + 2)
        for v in path:
            if not 1 <= v <= n + 1 or seen[v]:
                return False
            seen[v] = 1
        for u, v in zip(path, path[1:]):
            if u < n and v == u + 1:
                continue
            if u <= n and v == n + 1 and a[u - 1] == 0:
                continue
            if u == n + 1 and v <= n and a[v - 1] == 1:
                continue
            return False
    return O.done()


def _find(p, x):
    r = x
    while p[r] != r:
        r = p[r]
    while p[x] != r:
        p[x], x = r, p[x]
    return r


def c1559D(i, o, e):
    I, O, E = _ioe(i, o, e)
    n, m1, m2 = I.int(), I.int(), I.int()
    p1 = list(range(n + 1))
    p2 = list(range(n + 1))
    for _ in range(m1):
        a, b = I.int(), I.int()
        p1[_find(p1, a)] = _find(p1, b)
    for _ in range(m2):
        a, b = I.int(), I.int()
        p2[_find(p2, a)] = _find(p2, b)
    h = O.int()
    if h != E.int():
        return False
    for _ in range(h):
        u, v = O.int(), O.int()
        if not (1 <= u <= n and 1 <= v <= n):
            return False
        a1, b1, a2, b2 = _find(p1, u), _find(p1, v), _find(p2, u), _find(p2, v)
        if a1 == b1 or a2 == b2:
            return False
        p1[a1] = b1
        p2[a2] = b2
    return O.done()


def c1560E(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        t = I.s()
        ev = E.s()
        if ev != '-1':
            E.s()
        x = O.s()
        if ev == '-1' or x == '-1':
            if x != ev:
                return False
            continue
        order = O.s()
        if len(x) > len(t) or len(set(order)) != len(order) or set(order) != set(x):
            return False
        if not x.isascii() or not x.isalpha() or not x.islower():
            return False
        parts = []
        cur = x
        for ch in order:
            parts.append(cur)
            cur = cur.replace(ch, '')
        if ''.join(parts) != t:
            return False
    return O.done()


def c1562B(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        k = I.int()
        digits = I.s()
        L = O.int()
        x = O.s()
        le = E.int()
        E.s()
        if L != le or not 1 <= L <= k or len(x) != L or not x.isascii() or not x.isdigit():
            return False
        it = iter(digits)
        if not all(ch in it for ch in x) or _is_prime(int(x)):
            return False
    return O.done()


def c1562C(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        w = I.s()
        l1, r1, l2, r2 = O.ints(4)
        m = n // 2
        if not (1 <= l1 <= r1 <= n and 1 <= l2 <= r2 <= n):
            return False
        if r1 - l1 + 1 < m or r2 - l2 + 1 < m or (l1, r1) == (l2, r2):
            return False
        ft = int(w[l1 - 1:r1], 2)
        fw = int(w[l2 - 1:r2], 2)
        if (ft != 0) if fw == 0 else (ft % fw != 0):
            return False
    return O.done()


def c1562D2(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, q = I.int(), I.int()
        s = I.s()
        pf = [0] * (n + 1)
        acc = 0
        for j in range(1, n + 1):
            v = 1 if s[j - 1] == '+' else -1
            acc += v if j & 1 else -v
            pf[j] = acc
        for _ in range(q):
            l, r = I.int(), I.int()
            ke = E.int()
            E.skip(ke)
            k = O.int()
            if k != ke:
                return False
            pos = O.ints(k)
            if len(set(pos)) != k or any(not l <= x <= r for x in pos):
                return False
            tot = 0
            prev = l
            removed = 0
            for x in sorted(pos) + [r + 1]:
                a, b = prev, x - 1
                if a <= b:
                    seg = pf[b] - pf[a - 1]
                    if not a & 1:
                        seg = -seg
                    tot += seg if ((a - l) - removed) % 2 == 0 else -seg
                prev = x + 1
                removed += 1
            if tot != 0:
                return False
    return O.done()


def c1567D(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        s, n = I.int(), I.int()
        x = O.ints(n)
        y = E.ints(n)
        if any(v < 1 for v in x) or sum(x) != s:
            return False
        if sum(int(str(v), 11) for v in x) != sum(int(str(v), 11) for v in y):
            return False
    return O.done()


def c1567F(i, o, e):
    I, O, E = _ioe(i, o, e)
    n, m = I.int(), I.int()
    g = [I.s() for _ in range(n)]
    ev = E.word()
    ov = O.word()
    if ov != ev or ov not in ('YES', 'NO'):
        return False
    if ov == 'NO':
        return O.done()
    v = O.ints(n * m)
    for r in range(n):
        row = g[r]
        for c in range(m):
            x = v[r * m + c]
            if row[c] == '.':
                if x != 1 and x != 4:
                    return False
            else:
                s = 0
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < n and 0 <= cc < m and g[rr][cc] == '.':
                        s += v[rr * m + cc]
                if x != s or s % 5:
                    return False
    return O.done()


def c1569A(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        s = I.s()
        l, r = O.int(), O.int()
        el, er = E.int(), E.int()
        if el == -1:
            if (l, r) != (-1, -1):
                return False
            continue
        if not 1 <= l <= r <= n:
            return False
        sub = s[l - 1:r]
        if 2 * sub.count('a') != len(sub):
            return False
    return O.done()


def c1569B(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        ty = I.s()
        ev = E.word()
        if ev == 'YES':
            E.skip(n)
        ov = O.word()
        if ov != ev or ov not in ('YES', 'NO'):
            return False
        if ov == 'NO':
            continue
        G = [O.s() for _ in range(n)]
        if any(len(row) != n for row in G):
            return False
        for a in range(n):
            ra = G[a]
            if ra[a] != 'X':
                return False
            for b in range(a + 1, n):
                if (ra[b], G[b][a]) not in (('+', '-'), ('-', '+'), ('=', '=')):
                    return False
            if ty[a] == '1' and '-' in ra:
                return False
            if ty[a] == '2' and '+' not in ra:
                return False
    return O.done()


def c1572B(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        a = I.ints(n)
        ev = E.word()
        if ev == 'YES':
            E.skip(E.int())
        ov = O.word()
        if ov != ev or ov not in ('YES', 'NO'):
            return False
        if ov == 'NO':
            continue
        k = O.int()
        if not 0 <= k <= n:
            return False
        for b in O.ints(k):
            if not 1 <= b <= n - 2:
                return False
            x = a[b - 1] ^ a[b] ^ a[b + 1]
            a[b - 1] = a[b] = a[b + 1] = x
        if any(a):
            return False
    return O.done()


def c1574D(i, o, e):
    I, O, E = _ioe(i, o, e)
    n = I.int()
    vals = []
    for _ in range(n):
        vals.append(I.ints(I.int()))
    m = I.int()
    ban = set(tuple(I.ints(n)) for _ in range(m))
    b = O.ints(n)
    be = E.ints(n)
    if any(not 1 <= b[j] <= len(vals[j]) for j in range(n)) or tuple(b) in ban:
        return False
    if sum(vals[j][b[j] - 1] for j in range(n)) != sum(vals[j][be[j] - 1] for j in range(n)):
        return False
    return O.done()


def c1579B(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        a = I.ints(n)
        k = O.int()
        if not 0 <= k <= n:
            return False
        for _ in range(k):
            l, r, d = O.int(), O.int(), O.int()
            if not (1 <= l < r <= n and 1 <= d <= r - l):
                return False
            seg = a[l - 1:r]
            a[l - 1:r] = seg[d:] + seg[:d]
        if any(a[j] > a[j + 1] for j in range(n - 1)):
            return False
    return O.done()


def _composite(x):
    return x >= 4 and not _is_prime(x)


def c1586A(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        a = I.ints(n)
        xe = E.int()
        E.skip(xe)
        x = O.int()
        if x != xe:
            return False
        idx = O.ints(x)
        if len(set(idx)) != x or any(not 1 <= j <= n for j in idx):
            return False
        if not _composite(sum(a[j - 1] for j in idx)):
            return False
    return O.done()


def c1586B(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n, m = I.int(), I.int()
        req = [(I.int(), I.int(), I.int()) for _ in range(m)]
        E2 = O.ints(2 * (n - 1))
        adj = [[] for _ in range(n + 1)]
        for j in range(0, 2 * (n - 1), 2):
            a, b = E2[j], E2[j + 1]
            if not (1 <= a <= n and 1 <= b <= n) or a == b:
                return False
            adj[a].append(b)
            adj[b].append(a)
        # iterative DFS from 1: entry/exit times, children in entry order
        tin = [-1] * (n + 1)
        tout = [0] * (n + 1)
        kids = [[] for _ in range(n + 1)]
        timer = 0
        tin[1] = 0
        stack = [(1, 0)]
        while stack:
            v, j = stack[-1]
            if j < len(adj[v]):
                stack[-1] = (v, j + 1)
                w = adj[v][j]
                if tin[w] < 0:
                    timer += 1
                    tin[w] = timer
                    kids[v].append(tin[w])
                    stack.append((w, 0))
            else:
                tout[v] = timer
                stack.pop()
        if timer != n - 1:
            return False   # not connected, so not a tree

        def anc(x, y):
            return tin[x] <= tin[y] <= tout[x]

        for a, b, c in req:
            ia, ic = anc(b, a), anc(b, c)
            if ia != ic:
                return False
            if ia and bisect_right(kids[b], tin[a]) != bisect_right(kids[b], tin[c]):
                return False
    return O.done()


def c1586E(i, o, e):
    I, O, E = _ioe(i, o, e)
    n, m = I.int(), I.int()
    N = n + 1
    edges = set()
    for _ in range(m):
        u, v = I.int(), I.int()
        if u > v:
            u, v = v, u
        edges.add(u * N + v)
    q = I.int()
    Q = [(I.int(), I.int()) for _ in range(q)]
    ev = E.word()
    ov = O.word()
    if ov != ev or ov not in ('YES', 'NO'):
        return False
    if ov == 'NO':
        return O.int() == E.int() and O.done()
    par = Counter()
    for a, b in Q:
        k = O.int()
        path = O.ints(k)
        if k < 2 or path[0] != a or path[-1] != b or len(set(path)) != k:
            return False
        for u, v in zip(path, path[1:]):
            if u > v:
                u, v = v, u
            key = u * N + v
            if key not in edges:
                return False
            par[key] ^= 1
    return not any(par.values()) and O.done()


def c1586F(i, o, e):
    I, O, E = _ioe(i, o, e)
    n, k = I.int(), I.int()
    c = O.int()
    if c != E.int() or c < 1:
        return False
    col = O.ints(n * (n - 1) // 2)
    L = [[0] * (c + 1) for _ in range(n)]
    p = 0
    for a in range(n):
        La = L[a]
        for b in range(a + 1, n):
            x = col[p]
            p += 1
            if not 1 <= x <= c:
                return False
            if La[x] + 1 > L[b][x]:
                L[b][x] = La[x] + 1
                if L[b][x] >= k:
                    return False
    return O.done()


def c1591E(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, q = I.int(), I.int()
        a = [0] + I.ints(n)
        kids = [[] for _ in range(n + 1)]
        for v in range(2, n + 1):
            kids[I.int()].append(v)
        at = [[] for _ in range(n + 1)]
        for j in range(q):
            v, _l, _k = I.int(), I.int(), I.int()
            at[v].append(j)
        ans = O.ints(q)
        exp = E.ints(q)
        cnt = [0] * (n + 1)
        stack = [1]
        entered = bytearray(n + 1)
        while stack:
            v = stack[-1]
            if not entered[v]:
                entered[v] = 1
                cnt[a[v]] += 1
                for j in at[v]:
                    y, z = ans[j], exp[j]
                    if z == -1 or y == -1:
                        if y != z:
                            return False
                    elif not 1 <= y <= n or cnt[y] != cnt[z]:
                        return False
                stack.extend(kids[v])
            else:
                cnt[a[v]] -= 1
                stack.pop()
    return O.done()


def c1594A(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        l, r = O.int(), O.int()
        if not (-10 ** 18 <= l < r <= 10 ** 18) or (l + r) * (r - l + 1) != 2 * n:
            return False
    return O.done()


def c1594C(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        c = I.s()
        s = I.s()
        me = E.int()
        E.skip(me)
        m = O.int()
        if m != me:
            return False
        xs = O.ints(m)
        if any(not 1 <= x <= n for x in xs):
            return False
        for j in range(1, n + 1):
            if s[j - 1] != c and all(j % x == 0 for x in xs):
                return False
    return O.done()


def c1598G(i, o, e):
    I, O, _ = _ioe(i, o, e)
    s = I.s()
    x = I.s()
    n = len(s)
    l1, r1, l2, r2 = O.ints(4)
    if not (1 <= l1 <= r1 and r1 + 1 == l2 and l2 <= r2 <= n):
        return False
    A, B = s[l1 - 1:r1], s[l2 - 1:r2]
    if len(A) > len(x) or len(B) > len(x):
        return False
    ctx = decimal.Context(prec=len(x) + 5, Emax=decimal.MAX_EMAX, Emin=decimal.MIN_EMIN)
    D = decimal.Decimal
    if ctx.add(D(A), D(B)) != D(x):
        return False
    return O.done()


def c1599A(i, o, e):
    I, O, E = _ioe(i, o, e)
    n = I.int()
    a = I.ints(n)
    S = I.s()
    if E.s() == '-1':
        return O.s() == '-1' and O.done()
    left = set(a)
    L = R_ = 0
    for j in range(n):
        w = O.int()
        side = O.s()
        if w not in left or side not in ('L', 'R'):
            return False
        left.discard(w)
        if side == 'L':
            L += w
        else:
            R_ += w
        if (L <= R_) if S[j] == 'L' else (R_ <= L):
            return False
    return O.done()


def c1600F(i, o, e):
    I, O, E = _ioe(i, o, e)
    n, m = I.int(), I.int()
    N = n + 1
    edges = set()
    for _ in range(m):
        u, v = I.int(), I.int()
        if u > v:
            u, v = v, u
        edges.add(u * N + v)
    if E.int() == -1:
        return O.int() == -1 and O.done()
    S = O.ints(5)
    if len(set(S)) != 5 or any(not 1 <= x <= n for x in S):
        return False
    S.sort()
    k = sum(S[a] * N + S[b] in edges for a in range(5) for b in range(a + 1, 5))
    return k in (0, 10) and O.done()


def c1601B(i, o, e):
    I, O, E = _ioe(i, o, e)
    n = I.int()
    a = I.ints(n)
    b = I.ints(n)
    ke = E.int()
    k = O.int()
    if k != ke:
        return False
    if k == -1:
        return O.done()
    x = n
    path = O.ints(k)
    for j, y in enumerate(path):
        if not (0 <= y <= x and x - a[x - 1] <= y):
            return False
        if y == 0:
            return j == k - 1 and O.done()
        x = y + b[y - 1]
    return False


def c1604D(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        x, y = I.int(), I.int()
        n = O.int()
        if not (1 <= n <= 2 * 10 ** 18) or n % x != y % n:
            return False
    return O.done()


def c1606A(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        s = I.s()
        x = O.s()
        y = E.s()
        if len(x) != len(s) or x.count('a') + x.count('b') != len(x):
            return False
        if x.count('ab') != x.count('ba'):
            return False
        if sum(c != d for c, d in zip(s, x)) != sum(c != d for c, d in zip(s, y)):
            return False
    return O.done()


_MV = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}


def _robot_moves(n, m, s, r, c):
    k = 0
    for ch in s:
        dr, dc = _MV[ch]
        r += dr
        c += dc
        if not (1 <= r <= n and 1 <= c <= m):
            break
        k += 1
    return k


def c1607E(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, m = I.int(), I.int()
        s = I.s()
        r, c = O.int(), O.int()
        er, ec = E.int(), E.int()
        if not (1 <= r <= n and 1 <= c <= m):
            return False
        if _robot_moves(n, m, s, r, c) != _robot_moves(n, m, s, er, ec):
            return False
    return O.done()


def c1607F(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, m = I.int(), I.int()
        g = [I.s() for _ in range(n)]
        r, c, d = O.int(), O.int(), O.int()
        E.skip(2)
        if d != E.int() or not (1 <= r <= n and 1 <= c <= m):
            return False
        vis = bytearray(n * m)
        r -= 1
        c -= 1
        cnt = 0
        while 0 <= r < n and 0 <= c < m and not vis[r * m + c]:
            vis[r * m + c] = 1
            cnt += 1
            ch = g[r][c]
            if ch == 'L':
                c -= 1
            elif ch == 'R':
                c += 1
            elif ch == 'U':
                r -= 1
            else:
                r += 1
        if cnt != d:
            return False
    return O.done()


def c1607G(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, m = I.int(), I.int()
        d = [(I.int(), I.int()) for _ in range(n)]
        be = E.int()
        E.skip(2 * n)
        bal = O.int()
        if bal != be:
            return False
        xs = O.ints(2 * n)
        F = M = 0
        for j, (a, b) in enumerate(d):
            x, y = xs[2 * j], xs[2 * j + 1]
            if x < 0 or y < 0 or x > a or y > b or x + y != m:
                return False
            F += a - x
            M += b - y
        if abs(F - M) != bal:
            return False
    return O.done()


def c1607H(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        d = [(I.int(), I.int(), I.int()) for _ in range(n)]
        ve = E.int()
        E.skip(2 * n)
        v = O.int()
        if v != ve:
            return False
        xs = O.ints(2 * n)
        res = set()
        for j, (a, b, m) in enumerate(d):
            x, y = xs[2 * j], xs[2 * j + 1]
            if not (0 <= x <= a and 0 <= y <= b and x + y == m):
                return False
            res.add((a - x, b - y))
        if len(res) != v:
            return False
    return O.done()


def c1608A(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        a = O.ints(n)
        if any(not 1 <= x <= 10 ** 9 for x in a):
            return False
        if any(a[j] >= a[j + 1] or a[j + 1] % a[j] == 0 for j in range(n - 1)):
            return False
    return O.done()


def c1608B(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, a, b = I.int(), I.int(), I.int()
        ev = E.int()
        if ev != -1:
            E.skip(n - 1)
        x = O.int()
        if ev == -1 or x == -1:
            if x != ev:
                return False
            continue
        p = [x] + O.ints(n - 1)
        if sorted(p) != list(range(1, n + 1)):
            return False
        mx = sum(1 for j in range(1, n - 1) if p[j - 1] < p[j] > p[j + 1])
        mn = sum(1 for j in range(1, n - 1) if p[j - 1] > p[j] < p[j + 1])
        if (mx, mn) != (a, b):
            return False
    return O.done()


def c1613B(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        A = set(I.ints(n))
        seen = set()
        for _ in range(n // 2):
            x, y = O.int(), O.int()
            if x == y or x not in A or y not in A or x % y in A or (x, y) in seen:
                return False
            seen.add((x, y))
    return O.done()


def c1615D(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, m = I.int(), I.int()
        N = n + 1
        given = {}
        for _ in range(n - 1):
            x, y, v = I.int(), I.int(), I.int()
            if x > y:
                x, y = y, x
            given[x * N + y] = v
        Q = [(I.int(), I.int(), I.int()) for _ in range(m)]
        ev = E.word()
        if ev == 'YES':
            E.skip(3 * (n - 1))
        ov = O.word()
        if ov != ev or ov not in ('YES', 'NO'):
            return False
        if ov == 'NO':
            continue
        adj = [[] for _ in range(n + 1)]
        used = set()
        for _ in range(n - 1):
            x, y, v = O.int(), O.int(), O.int()
            if not (1 <= x <= n and 1 <= y <= n and 0 <= v < 1 << 30):
                return False
            key = min(x, y) * N + max(x, y)
            if key not in given or key in used or (given[key] != -1 and given[key] != v):
                return False
            used.add(key)
            b = bin(v).count('1') & 1
            adj[x].append((y, b))
            adj[y].append((x, b))
        par = [-1] * (n + 1)
        par[1] = 0
        st = [1]
        while st:
            u = st.pop()
            for w, b in adj[u]:
                if par[w] < 0:
                    par[w] = par[u] ^ b
                    st.append(w)
        if any(par[a] ^ par[b] != p for a, b, p in Q):
            return False
    return O.done()


def c1615H(i, o, e):
    I, O, E = _ioe(i, o, e)
    n, m = I.int(), I.int()
    a = I.ints(n)
    req = [(I.int(), I.int()) for _ in range(m)]
    b = O.ints(n)
    be = E.ints(n)
    if any(abs(x) > 10 ** 15 for x in b):
        return False
    if any(b[u - 1] > b[v - 1] for u, v in req):
        return False
    if sum(abs(x - y) for x, y in zip(a, b)) != sum(abs(x - y) for x, y in zip(a, be)):
        return False
    return O.done()


def c1618A(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        b = I.ints(7)
        x, y, z = O.ints(3)
        if min(x, y, z) < 1:
            return False
        if sorted([x, y, z, x + y, x + z, y + z, x + y + z]) != b:
            return False
    return O.done()


def c1618B(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        bg = [I.s() for _ in range(n - 2)]
        w = O.s()
        if len(w) != n or w.count('a') + w.count('b') != n:
            return False
        full = [w[j:j + 2] for j in range(n - 1)]
        j = 0
        while j < n - 2 and full[j] == bg[j]:
            j += 1
        if full[:j] + full[j + 1:] != bg:
            return False
    return O.done()


def _alternates(a, d):
    red = [x % d == 0 for x in a]
    return all(red[j] != red[j + 1] for j in range(len(a) - 1))


def c1618C(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        a = I.ints(n)
        d = O.int()
        de = E.int()
        if de == 0 or d == 0:
            if d != de:
                return False
            continue
        if not (1 <= d <= 10 ** 18) or not _alternates(a, d):
            return False
    return O.done()


def c1618E(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        b = I.ints(n)
        ev = E.word()
        if ev == 'YES':
            E.skip(n)
        ov = O.word()
        if ov != ev or ov not in ('YES', 'NO'):
            return False
        if ov == 'NO':
            continue
        a = O.ints(n)
        if any(not 1 <= x <= 10 ** 9 for x in a):
            return False
        S = sum(a)
        cur = sum(a[j] * ((-j) % n + 1) for j in range(n))
        if cur != b[0]:
            return False
        for j in range(1, n):
            cur += S - n * a[j]
            if cur != b[j]:
                return False
    return O.done()


def _tanya(a, b):
    sa, sb = str(a), str(b)
    L = max(len(sa), len(sb))
    sa, sb = sa.zfill(L), sb.zfill(L)
    return int(''.join(str(int(x) + int(y)) for x, y in zip(sa, sb)))


def c1619C(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        a, s = I.int(), I.int()
        b = O.int()
        be = E.int()
        if be == -1 or b == -1:
            if b != be:
                return False
            continue
        if b < 1 or _tanya(a, b) != s:
            return False
    return O.done()


def c1619F(i, o, e):
    I, O, _ = _ioe(i, o, e)
    for _ in range(I.int()):
        n, m, k = I.int(), I.int(), I.int()
        lo, hi = n // m, -(-n // m)
        big = [0] * (n + 1)
        for _ in range(k):
            seen = bytearray(n + 1)
            cnt = 0
            for _ in range(m):
                p = O.int()
                if p != lo and p != hi:
                    return False
                for x in O.ints(p):
                    if not 1 <= x <= n or seen[x]:
                        return False
                    seen[x] = 1
                    if p == hi:
                        big[x] += 1
                cnt += p
            if cnt != n:
                return False
        if max(big[1:]) - min(big[1:]) > 1:
            return False
    return O.done()


def c1620F(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        p = I.ints(n)
        ev = E.word()
        if ev == 'YES':
            E.skip(n)
        ov = O.word()
        if ov != ev or ov not in ('YES', 'NO'):
            return False
        if ov == 'NO':
            continue
        a = O.ints(n)
        if any(abs(x) != v for x, v in zip(a, p)):
            return False
        # no i < j < k with a_i > a_j > a_k
        suf = [0] * (n + 1)
        suf[n] = float('inf')
        for j in range(n - 1, -1, -1):
            suf[j] = a[j] if a[j] < suf[j + 1] else suf[j + 1]
        pre = -float('inf')
        for j in range(n):
            if pre > a[j] > suf[j + 1]:
                return False
            if a[j] > pre:
                pre = a[j]
    return O.done()


def c1622B(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n = I.int()
        p = I.ints(n)
        s = I.s()
        q = O.ints(n)
        qe = E.ints(n)
        if sorted(q) != list(range(1, n + 1)):
            return False
        liked = [q[j] for j in range(n) if s[j] == '1']
        disliked = [q[j] for j in range(n) if s[j] == '0']
        if liked and disliked and min(liked) < max(disliked):
            return False
        if sum(abs(x - y) for x, y in zip(p, q)) != sum(abs(x - y) for x, y in zip(p, qe)):
            return False
    return O.done()


def c1622E(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, m = I.int(), I.int()
        x = I.ints(n)
        S = [I.s() for _ in range(n)]
        q = O.ints(m)
        qe = E.ints(m)
        if sorted(q) != list(range(1, m + 1)):
            return False

        def surprise(pp):
            return sum(abs(x[r] - sum(pp[j] for j in range(m) if S[r][j] == '1')) for r in range(n))

        if surprise(q) != surprise(qe):
            return False
    return O.done()


def _primes_upto(n):
    if n < 2:
        return []
    s = bytearray([1]) * (n + 1)
    s[0] = s[1] = 0
    p = 2
    while p * p <= n:
        if s[p]:
            s[p * p::p] = bytes(len(range(p * p, n + 1, p)))
        p += 1
    return list(compress(range(n + 1), s))


def _fact_odd_primes(x, primes):
    """Exact set of primes whose exponent in x! is odd (Legendre's formula)."""
    out = set()
    for p in primes:
        if p > x:
            break
        e, q = 0, x
        while q:
            q //= p
            e += q
        if e & 1:
            out.add(p)
    return out


def c1622F(i, o, e):
    # The product of x! over x in S must be a square. With P(n) = 1! 2! ... n!, pairing (2j-1)! and (2j)!
    # gives P(2m) = square * 2^m * m!, and P(2m+1) = P(2m) * (2m+1)!. The answer S is {1..n} minus a set D
    # (|D| = n - k, at most 3 at the optimum k, which comes from the expected output), so S is a square
    # exactly when the odd-exponent prime sets of P(n) and of the x! for x in D cancel. Exact, no hashing.
    I, O, E = _ioe(i, o, e)
    n = I.int()
    k = O.int()
    if k != E.int() or not 0 <= k <= n:
        return False
    a = O.ints(k)
    seen = bytearray(n + 1)
    for x in a:
        if not 1 <= x <= n or seen[x]:
            return False
        seen[x] = 1
    missing = [x for x in range(1, n + 1) if not seen[x]]
    if len(missing) > 3:
        return False      # cannot happen at the optimum for n >= 1 (the optimum drops at most 3 numbers)
    primes = _primes_upto(n)
    m = n // 2
    odd = _fact_odd_primes(m, primes)
    if m & 1:
        odd ^= {2}
    if n & 1:
        odd ^= _fact_odd_primes(n, primes)
    for x in missing:
        odd ^= _fact_odd_primes(x, primes)
    return not odd and O.done()


def c1606D(i, o, e):
    I, O, E = _ioe(i, o, e)
    for _ in range(I.int()):
        n, m = I.int(), I.int()
        rows = [I.ints(m) for _ in range(n)]
        ev = E.word()
        if ev == 'YES':
            E.skip(2)
        ov = O.word()
        if ov != ev or ov not in ('YES', 'NO'):
            return False
        if ov == 'NO':
            continue
        col = O.s()
        k = O.int()
        if len(col) != n or set(col) != {'R', 'B'} or not 1 <= k < m:
            return False
        red_left_min = blue_right_min = float('inf')
        blue_left_max = red_right_max = -float('inf')
        for row, c in zip(rows, col):
            left, right = row[:k], row[k:]
            if c == 'R':
                red_left_min = min(red_left_min, min(left))
                red_right_max = max(red_right_max, max(right))
            else:
                blue_left_max = max(blue_left_max, max(left))
                blue_right_min = min(blue_right_min, min(right))
        if not (red_left_min > blue_left_max and blue_right_min > red_right_max):
            return False
    return O.done()


CHECKERS = {
    '1549A': c1549A, '1551A': c1551A, '1551B2': c1551B2, '1552E': c1552E, '1554D': c1554D, '1558C': c1558C,
    '1559B': c1559B, '1559C': c1559C, '1559D1': c1559D, '1559D2': c1559D, '1560E': c1560E, '1562B': c1562B,
    '1562C': c1562C, '1562D2': c1562D2, '1567D': c1567D, '1567F': c1567F, '1569A': c1569A, '1569B': c1569B,
    '1572B': c1572B, '1574D': c1574D, '1579B': c1579B, '1586A': c1586A, '1586B': c1586B, '1586E': c1586E,
    '1586F': c1586F, '1591E': c1591E, '1594A': c1594A, '1594C': c1594C, '1598G': c1598G, '1599A': c1599A,
    '1600F': c1600F, '1601B': c1601B, '1604D': c1604D, '1606A': c1606A, '1606D': c1606D, '1607E': c1607E,
    '1607F': c1607F, '1607G': c1607G, '1607H': c1607H, '1608A': c1608A, '1608B': c1608B, '1613B': c1613B,
    '1615D': c1615D, '1615H': c1615H, '1618A': c1618A, '1618B': c1618B, '1618C': c1618C, '1618E': c1618E,
    '1619C': c1619C, '1619F': c1619F, '1620F': c1620F, '1622B': c1622B, '1622E': c1622E, '1622F': c1622F,
}
