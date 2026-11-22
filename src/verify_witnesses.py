#!/usr/bin/env python3
"""
Zero-dependency pure-Python mathematical verifier for TII McEliece Track-2
equivalent private keys and harvested codeword datasets.

Independently fetches official TII public keys `pk_McEliece_<sec>.txt` into the
OS temporary directory (or reads from --pk-dir) and verifies from first principles:
  1. SHA-256 digests of both public key and witness files, plus field-defining polynomial.
  2. Exact parsing of g(X) (monic, degree t) and support L (n distinct elements in GF(2^m)).
  3. Square-freeness (gcd(g, g') == 1), irreducibility (Rabin's test over GF(2^m)),
     and non-vanishing g(L_i) != 0 for all i.
  4. Exact parity-check matrix reconstruction H_raw(L, g) over GF(2) and
     bit-for-bit RREF equality: RREF(H_raw(L, g)) == H_pub.
  5. Optional verification (`--check-words`) of every codeword in `codewords/words_*.txt.gz`
     against the shortened parity-check matrix H_A = H_pub[:, :n'], plus exact SHA-256 checks.

Usage:
  python3 src/verify_witnesses.py [--check-words]
"""
import argparse
import gzip
import hashlib
import os
import re
import sys
import tempfile
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/ElenaKirshanova/tii_decoding_challenge/main/public_keyRec"
TARGETS = [
    (
        245, 493, 9, 12, 326, "words_245_np326.txt.gz", 24500,
        "ea029e86d34d13334bb9439f87333a8e87ebd902379ec515331d2bed78e973a8",
        "967d2f4e934ca7617bdd71cfb5a0fcd98ec0f54a2c50db5170899ab3debac523",
        "b54b11b6c695b32a08aedb545cfb68dac9e0e3e7e1184175474344bfc59b4a11",
    ),
    (
        237, 500, 9, 15, 378, "words_237_np378.txt.gz", 29742,
        "b3cba630833e00d8e225d1943be87c31ddd74f270bd73f61c373aa70b1356f3b",
        "128e26137b2b2e8128316cbc5e1c7766073d790e26a308a9766a120f05770e0f",
        "7793b74438d76235c8f5048a6b5644113b8372d999f837e1c09c274b2a472531",
    ),
]


def gf_mul(a: int, b: int, mod_poly: int, m: int) -> int:
    res = 0
    while b:
        if b & 1:
            res ^= a
        b >>= 1
        a <<= 1
        if a & (1 << m):
            a ^= mod_poly
    return res


def gf_pow(a: int, e: int, mod_poly: int, m: int) -> int:
    res = 1
    while e:
        if e & 1:
            res = gf_mul(res, a, mod_poly, m)
        a = gf_mul(a, a, mod_poly, m)
        e >>= 1
    return res


def gf_inv(a: int, mod_poly: int, m: int) -> int:
    if a == 0:
        raise ZeroDivisionError("Inverse of 0 in GF(2^m)")
    return gf_pow(a, (1 << m) - 2, mod_poly, m)


def parse_gf_elem(s: str) -> int:
    s = s.strip()
    if s == "0":
        return 0
    val = 0
    for term in s.split("+"):
        term = term.strip()
        if not term:
            continue
        if term == "1":
            val ^= 1
        elif term == "a":
            val ^= 2
        else:
            m_obj = re.fullmatch(r"a\^(\d+)", term)
            if not m_obj:
                raise ValueError(f"Unexpected GF term: {term!r}")
            val ^= 1 << int(m_obj.group(1))
    return val


def format_mod_poly(flist) -> str:
    terms = []
    for deg in range(len(flist) - 1, -1, -1):
        if flist[deg]:
            if deg == 0:
                terms.append("1")
            elif deg == 1:
                terms.append("a")
            else:
                terms.append(f"a^{deg}")
    return " + ".join(terms)


def poly_trim(p):
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


def poly_mul(a, b, mod_poly, m):
    res = [0] * (len(a) + len(b) - 1)
    for i, ca in enumerate(a):
        if ca == 0:
            continue
        for j, cb in enumerate(b):
            if cb:
                res[i + j] ^= gf_mul(ca, cb, mod_poly, m)
    return poly_trim(res)


def poly_mod(a, b, mod_poly, m):
    a = list(a)
    poly_trim(a)
    b = list(b)
    poly_trim(b)
    db = len(b) - 1
    inv_lb = gf_inv(b[-1], mod_poly, m)
    while len(a) - 1 >= db and not (len(a) == 1 and a[0] == 0):
        shift = len(a) - 1 - db
        coef = gf_mul(a[-1], inv_lb, mod_poly, m)
        for i in range(db + 1):
            a[shift + i] ^= gf_mul(coef, b[i], mod_poly, m)
        poly_trim(a)
    return a


def poly_gcd(a, b, mod_poly, m):
    a = poly_trim(list(a))
    b = poly_trim(list(b))
    while not (len(b) == 1 and b[0] == 0):
        a, b = b, poly_mod(a, b, mod_poly, m)
    inv_l = gf_inv(a[-1], mod_poly, m)
    return [gf_mul(c, inv_l, mod_poly, m) for c in a]


def poly_powmod(base, exp, modulus, mod_poly, m):
    res = [1]
    b = poly_mod(base, modulus, mod_poly, m)
    while exp > 0:
        if exp & 1:
            res = poly_mod(poly_mul(res, b, mod_poly, m), modulus, mod_poly, m)
        b = poly_mod(poly_mul(b, b, mod_poly, m), modulus, mod_poly, m)
        exp >>= 1
    return res


def poly_is_irreducible(g, mod_poly, m):
    """Rabin's irreducibility test over GF(q), q = 2^m."""
    t = len(g) - 1
    q = 1 << m
    primes = []
    tmp = t
    d = 2
    while d * d <= tmp:
        if tmp % d == 0:
            primes.append(d)
            while tmp % d == 0:
                tmp //= d
        d += 1
    if tmp > 1:
        primes.append(tmp)

    x_poly = [0, 1]
    h = x_poly
    for _ in range(t):
        h = poly_powmod(h, q, g, mod_poly, m)
    if h != x_poly:
        return False

    for p in primes:
        h = x_poly
        for _ in range(t // p):
            h = poly_powmod(h, q, g, mod_poly, m)
        diff = list(h)
        while len(diff) < 2:
            diff.append(0)
        diff[1] ^= 1
        poly_trim(diff)
        if len(poly_gcd(diff, g, mod_poly, m)) != 1:
            return False
    return True


def poly_eval(g, x, mod_poly, m):
    acc = 0
    for c in reversed(g):
        acc = gf_mul(acc, x, mod_poly, m) ^ c
    return acc


def rref_gf2_rows(rows_int, ncols):
    """Compute exact RREF of binary matrix stored as list of Python ints (bit j = col j)."""
    rows = list(rows_int)
    nrows = len(rows)
    r = 0
    for c in range(ncols):
        piv = -1
        for i in range(r, nrows):
            if (rows[i] >> c) & 1:
                piv = i
                break
        if piv == -1:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        pr = rows[r]
        for i in range(nrows):
            if i != r and ((rows[i] >> c) & 1):
                rows[i] ^= pr
        r += 1
        if r == nrows:
            break
    return rows


def verify_words_gz(words_path, h_pub_rows, mt, nprime, w_target):
    col_masks = [0] * nprime
    for r in range(mt):
        row_r = h_pub_rows[r]
        bit = 1 << r
        for j in range(nprime):
            if (row_r >> j) & 1:
                col_masks[j] |= bit
    seen = set()
    with gzip.open(words_path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            coords = [int(x) for x in s.split()]
            if len(coords) != w_target or any(c < 0 or c >= nprime for c in coords):
                raise RuntimeError(f"{words_path}:{lineno}: invalid weight or out-of-range coordinate")
            sup = tuple(sorted(coords))
            if len(set(sup)) != w_target:
                raise RuntimeError(f"{words_path}:{lineno}: duplicate coordinate in support")
            syn = 0
            for c in sup:
                syn ^= col_masks[c]
            if syn != 0:
                raise RuntimeError(f"{words_path}:{lineno}: nonzero syndrome in H_A")
            seen.add(sup)
    if not seen:
        raise RuntimeError(f"{words_path}: empty codeword dataset")
    return len(seen)


def verify_instance(sec, n, m, t, nprime, words_fname, expected_words,
                    expected_pk_sha, expected_sk_sha, expected_gz_sha,
                    repo_root, pk_dir, check_words):
    sk_path = os.path.join(repo_root, "witnesses", f"sk_McEliece_{sec}.txt")
    pk_path = os.path.join(pk_dir, f"pk_McEliece_{sec}.txt")
    if not os.path.isfile(pk_path):
        url = f"{BASE_URL}/pk_McEliece_{sec}.txt"
        urllib.request.urlretrieve(url, pk_path)

    with open(pk_path, "rb") as f:
        pk_bytes = f.read()
    with open(sk_path, "rb") as f:
        sk_bytes = f.read()
    pk_sha = hashlib.sha256(pk_bytes).hexdigest()
    sk_sha = hashlib.sha256(sk_bytes).hexdigest()
    if pk_sha != expected_pk_sha:
        raise RuntimeError(f"TII-{sec} pk SHA-256 mismatch: {pk_sha} != {expected_pk_sha}")
    if sk_sha != expected_sk_sha:
        raise RuntimeError(f"TII-{sec} sk SHA-256 mismatch: {sk_sha} != {expected_sk_sha}")

    mt = m * t
    pk_lines = pk_bytes.decode("utf-8").splitlines()
    h_pub_rows = []
    for i in range(mt):
        vals = [int(x) for x in pk_lines[i].strip("[] ").split() if x]
        if len(vals) != n:
            raise RuntimeError(f"Row {i} len {len(vals)} != {n}")
        v = 0
        for j, b in enumerate(vals):
            if b:
                v |= 1 << j
        h_pub_rows.append(v)

    flist = [int(x) for x in pk_lines[mt].strip("[] ").split(",")]
    mod_poly = 0
    for i, b in enumerate(flist):
        if b:
            mod_poly |= 1 << i

    sk_lines = sk_bytes.decode("utf-8").splitlines()
    g_strs = sk_lines[0].strip("[] ").split(",")
    l_strs = sk_lines[1].strip("[] ").split(",")
    g = [parse_gf_elem(s) for s in g_strs]
    L = [parse_gf_elem(s) for s in l_strs]

    if len(g) != t + 1 or g[-1] != 1:
        raise RuntimeError(f"g is not monic of degree {t}")
    if len(L) != n or len(set(L)) != n:
        raise RuntimeError(f"L does not have {n} distinct elements")

    g_deriv = [g[i] if (i % 2 == 1) else 0 for i in range(1, len(g))]
    poly_trim(g_deriv)
    if len(poly_gcd(g, g_deriv, mod_poly, m)) != 1:
        raise RuntimeError("g is not square-free")
    irred = poly_is_irreducible(g, mod_poly, m)
    if not irred:
        raise RuntimeError("g is not irreducible over GF(2^m)")

    inv_gL = []
    for x in L:
        gx = poly_eval(g, x, mod_poly, m)
        if gx == 0:
            raise RuntimeError("g(L_i) == 0")
        inv_gL.append(gf_inv(gx, mod_poly, m))

    h_raw_rows = [0] * mt
    for j in range(n):
        xj = L[j]
        val = inv_gL[j]
        for i in range(t):
            for b in range(m):
                if (val >> b) & 1:
                    h_raw_rows[m * i + b] |= 1 << j
            val = gf_mul(val, xj, mod_poly, m)

    h_rref = rref_gf2_rows(h_raw_rows, n)
    if h_rref != h_pub_rows:
        raise RuntimeError(f"RREF(H_raw(L, g)) != H_pub for TII-{sec}!")

    print(f"[PASS] TII-{sec} (n={n}, m={m}, t={t}, mt={mt})")
    print(f"       GF(2^{m})  : {format_mod_poly(flist)}")
    print(f"       pk SHA-256 : {pk_sha}")
    print(f"       sk SHA-256 : {sk_sha}")
    print(f"       g monic deg={t}, square-free=True, irreducible={irred}, |L|={len(set(L))}, RREF_EQ=True")

    if check_words:
        words_path = os.path.join(repo_root, "codewords", words_fname)
        with open(words_path, "rb") as f:
            w_sha = hashlib.sha256(f.read()).hexdigest()
        if w_sha != expected_gz_sha:
            raise RuntimeError(f"TII-{sec} words .gz SHA-256 mismatch: {w_sha} != {expected_gz_sha}")
        n_ok = verify_words_gz(words_path, h_pub_rows, mt, nprime, 2 * t + 1)
        if n_ok != expected_words:
            raise RuntimeError(f"TII-{sec} unique codeword count mismatch: {n_ok} != {expected_words}")
        print(f"       words .gz  : {n_ok} unique weight-{2*t+1} codewords verified in ker(H_A) (n'={nprime})")
        print(f"       gz SHA-256 : {w_sha}")


def main():
    parser = argparse.ArgumentParser(description="Verify TII Track-2 equivalent private keys in pure Python.")
    parser.add_argument("--pk-dir", default=os.path.join(tempfile.gettempdir(), "tii_pk_cache"),
                        help="Directory to cache downloaded official public keys (default: OS temp dir)")
    parser.add_argument("--check-words", action="store_true",
                        help="Also verify every harvested codeword in codewords/words_*.txt.gz against H_A")
    args = parser.parse_args()

    os.makedirs(args.pk_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for row in TARGETS:
        verify_instance(*row, repo_root, args.pk_dir, args.check_words)
    print("\nALL WITNESSES VERIFIED BIT-FOR-BIT IN PURE PYTHON (ZERO EXTERNAL DEPS).")


if __name__ == "__main__":
    main()
