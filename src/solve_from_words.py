#!/usr/bin/env python3
"""
Algebraic Key Recovery & Official TII Track-2 Verifier from Pre-Harvested Codewords.

Attribution & License Notice:
  Derived from MiniMax's `tii_runner.py` / `goppa_attack.py` (Apache License 2.0,
  https://github.com/MiniMax-AI/Research/tree/main/Cryptography/TII-McEliece,
  released Nov 20, 2026). All algebraic lifting logic (`affine_polars`, `polar_forms`,
  `low_rank_components`, `lift_all`, `make_field`, `goppa_from_word`, `extend_support`)
  is imported directly from MiniMax's repository.

Given a shortened-window parity-check matrix H_A = H_pub[:, :nprime] and a file of
minimum odd-weight (w = 2t + 1) codewords (plain text or `.gz`), this script:
  1. Strictly validates every input codeword (exact weight, distinct indices in [0, n'),
     and exact parity H_A * v == 0 over GF(2)).
  2. Runs MiniMax's Route-A quadratic polar lift to recover an equivalent private key (g, L).
  3. Enforces degree, monicity, square-freeness, irreducibility, support distinctness,
     non-vanishing g(L_i) != 0, and bit-for-bit RREF equality with H_pub.
  4. Invokes the unmodified official TII `check_solution_track2.sage` harness and verifies
     both process exit code 0 and output == "1".

Usage:
  sage -python solve_from_words.py <sec> <nprime> <words_file[.gz]> <mm_goppa_dir> <out_dir>
"""
import gzip
import hashlib
import os
import subprocess
import sys
import time
import urllib.request

TII_PK_BASE_URL = "https://raw.githubusercontent.com/ElenaKirshanova/tii_decoding_challenge/main/public_keyRec"
KNOWN_PK_SHA256 = {
    245: "ea029e86d34d13334bb9439f87333a8e87ebd902379ec515331d2bed78e973a8",
    228: "f98ddc98e6f54a60308b49ec36e16f70b6249ba4b7eac79f72cdf7fe70649f5a",
    237: "b3cba630833e00d8e225d1943be87c31ddd74f270bd73f61c373aa70b1356f3b",
    235: "f56256c7b6f253ff7748207ee05545d92644f6f8c5a7189fcaff9442eedd87a5",
}

if len(sys.argv) < 6:
    print("Usage: sage -python solve_from_words.py <sec> <nprime> <words_file[.gz]> <mm_goppa_dir> <out_dir>",
          file=sys.stderr)
    sys.exit(2)

sec = int(sys.argv[1])
nprime = int(sys.argv[2])
words_file = os.path.abspath(sys.argv[3])
mm_dir = os.path.abspath(sys.argv[4])
out_dir = os.path.abspath(sys.argv[5])

os.makedirs(out_dir, exist_ok=True)
sys.path.insert(0, mm_dir)
sys.path.insert(0, os.path.join(mm_dir, "route_a"))
os.environ["TII_OUTPUT_DIR"] = out_dir

# Ensure pk_McEliece_<sec>.txt exists in tii_repo/public_keyRec/ (MiniMax's fetch_third_party.sh
# only extracts its 94 solved keys and omits unsolved instances such as TII-237).
tii_pk_dir = os.path.join(mm_dir, "tii_repo", "public_keyRec")
pk_path = os.path.join(tii_pk_dir, f"pk_McEliece_{sec}.txt")
if not os.path.isfile(pk_path):
    os.makedirs(tii_pk_dir, exist_ok=True)
    urllib.request.urlretrieve(f"{TII_PK_BASE_URL}/pk_McEliece_{sec}.txt", pk_path)
if sec in KNOWN_PK_SHA256:
    with open(pk_path, "rb") as f:
        got_sha = hashlib.sha256(f.read()).hexdigest()
    if got_sha != KNOWN_PK_SHA256[sec]:
        raise RuntimeError(f"pk_McEliece_{sec}.txt SHA-256 mismatch: {got_sha} != {KNOWN_PK_SHA256[sec]}")

import goppa_attack
import tii_runner
from sage.all import GF, matrix, vector

t_load0 = time.time()
meta, H_pub, flist, ff = tii_runner.load_tii_public_key(sec)
n, k, t, m = meta["n"], meta["k"], meta["t"], meta["m"]
mt = m * t
if nprime <= mt or nprime > n:
    raise ValueError(f"Invalid n'={nprime} for mt={mt}, n={n}")

A = list(range(nprime))
HA = H_pub.matrix_from_columns(A)
if HA.rank() != mt:
    raise RuntimeError(f"Shortened matrix rank {HA.rank()} != {mt}")
print(f"[TII-{sec}] n={n}, m={m}, t={t}, mt={mt}, n'={nprime}, rank(HA)={HA.rank()}", flush=True)

# Precompute column bitmasks of HA over GF(2) for fast exact parity check
ha_cols = [0] * nprime
for r in range(mt):
    row_r = HA.row(r)
    bit = 1 << r
    for j in range(nprime):
        if row_r[j]:
            ha_cols[j] |= bit

w_target = 2 * t + 1
words = []
seen = set()
opener = gzip.open if words_file.endswith(".gz") else open
with opener(words_file, "rt", encoding="utf-8") as f:
    for lineno, raw_line in enumerate(f, 1):
        s = raw_line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) != w_target:
            raise ValueError(f"{words_file}:{lineno}: expected {w_target} indices, got {len(parts)}")
        try:
            coords = [int(x) for x in parts]
        except ValueError as exc:
            raise ValueError(f"{words_file}:{lineno}: non-integer coordinate") from exc
        sup = tuple(sorted(coords))
        if len(set(sup)) != w_target or sup[0] < 0 or sup[-1] >= nprime:
            raise ValueError(f"{words_file}:{lineno}: duplicate or out-of-range coordinate in [0, {nprime})")
        syn = 0
        for c in sup:
            syn ^= ha_cols[c]
        if syn != 0:
            raise ValueError(f"{words_file}:{lineno}: vector is not in ker(H_A) (nonzero syndrome)")
        if sup not in seen:
            seen.add(sup)
            words.append(sup)

if not words:
    raise RuntimeError(f"No valid weight-{w_target} codewords found in {words_file}")
print(f"[TII-{sec}] Verified {len(words)} unique weight-{w_target} codewords in ker(H_A) "
      f"({time.time() - t_load0:.2f}s)", flush=True)

t_lift0 = time.time()
GA_ = HA.right_kernel().basis_matrix()
E, relations, aff_dim, pair_index, _ = goppa_attack.affine_polars(GA_, words)
q_dim = relations.nrows()
print(f"[TII-{sec}] Relation quotient dim = {q_dim} (expected {m + 1}) in {time.time() - t_lift0:.2f}s", flush=True)

G0 = E.transpose() * GA_
mats = goppa_attack.polar_forms(relations, aff_dim, pair_index)
t_form = goppa_attack.chart_weight_form(G0)
mask = goppa_attack.express_in_basis(mats, t_form)
divided = None
if mask:
    drop = (mask & -mask).bit_length() - 1
    divided = mats[:drop] + mats[drop + 1:]

components = None
if divided is not None and len(divided) == m:
    components = [divided]
    print(f"[TII-{sec}] Quotient mod T_pub has exact dim {m} -> direct lift!", flush=True)
elif divided is not None:
    tii_runner._ensure_route_a_binaries()
    components, _, _ = goppa_attack.low_rank_components(
        divided, aff_dim, nprime=nprime, m=m, finder=tii_runner.RANKMASK_BIN,
        restricted=(len(divided) > 16), target=m
    )
if components is None:
    tii_runner._ensure_route_a_binaries()
    components, _, _ = goppa_attack.low_rank_components(
        mats, aff_dim, nprime=nprime, m=m, finder=tii_runner.RANKMASK_BIN,
        restricted=(len(mats) > 17), target=m + 1
    )

pivots = G0.pivots()
U = G0[:, pivots].inverse()
Gs = U * G0
pivot_set = set(pivots)
nonpivots = [j for j in range(nprime) if j not in pivot_set]

for raw in components:
    Bs = [U * B * U.transpose() for B in raw]
    W, blocks, I = goppa_attack.lift_all(Bs, Gs, nonpivots, pivots, m=m)
    projections = range(1 << m) if len(Bs) == m + 1 else range(1)
    for proj in projections:
        if len(Bs) == m + 1:
            Wp = matrix(GF(2), [W.row(i) + (((proj >> i) & 1) * W.row(m)) for i in range(m)])
            bp = [
                [
                    [int(blocks[tp][i][j]) ^ ((proj >> tp & 1) * int(blocks[m][i][j])) for j in range(m)]
                    for i in range(m)
                ]
                for tp in range(m)
            ]
        else:
            Wp, bp = W, blocks[:m]
        try:
            labels_A = goppa_attack.make_field(Wp, bp, I, ff, m=m)
        except (ValueError, ZeroDivisionError, ArithmeticError):
            continue
        if len(set(labels_A)) != nprime:
            continue
        poly = goppa_attack.goppa_from_word(ff, labels_A, words[0], t=t)
        if poly.degree() != t or poly.gcd(poly.derivative()) != 1:
            continue
        if any(poly(x) == 0 for x in labels_A):
            continue
        if goppa_attack.parity_matrix(ff, labels_A, poly, m, t).row_space() != HA.row_space():
            continue

        labels_full = goppa_attack.extend_support(H_pub, ff, A, labels_A, poly, m=m, t=t)
        poly = poly / poly.leading_coefficient()
        lift_elapsed = time.time() - t_lift0

        # Explicit non-removable checks on the recovered equivalent private key
        if poly.degree() != t or poly.leading_coefficient() != 1:
            raise RuntimeError("Recovered g(X) is not monic of degree t")
        if poly.gcd(poly.derivative()) != 1:
            raise RuntimeError("Recovered g(X) is not square-free")
        if not poly.is_irreducible():
            raise RuntimeError("Recovered g(X) is not irreducible over GF(2^m)")
        if len(labels_full) != n or len(set(labels_full)) != n:
            raise RuntimeError(f"Recovered support L does not have {n} distinct elements")
        if any(poly(x) == 0 for x in labels_full):
            raise RuntimeError("Recovered g(X) vanishes on a support element")

        H_cand = tii_runner.tii_gen_parity_full(labels_full, poly, m, n, ff, echelonForm=True)
        if H_cand != H_pub:
            raise RuntimeError("RREF bit-for-bit equality H_cand == H_pub failed!")

        sk_path = os.path.join(out_dir, f"sk_McEliece_{sec}.txt")
        g_list = [poly[i] for i in range(t + 1)]
        with open(sk_path, "w", encoding="utf-8") as f:
            f.write(str(g_list) + "\n")
            f.write(str(list(labels_full)) + "\n")

        # Run official TII Track-2 checker
        t_chk0 = time.time()
        sage_bin = os.environ.get("SAGE_BIN", "sage")
        tii_repo = os.path.join(mm_dir, "tii_repo")
        chk = subprocess.run(
            [sage_bin, "check_solution_track2.sage", str(sec), sk_path],
            cwd=tii_repo, capture_output=True, text=True
        )
        chk_elapsed = time.time() - t_chk0
        out_val = chk.stdout.strip().splitlines()[-1].strip() if chk.stdout.strip() else ""
        if chk.returncode != 0 or out_val != "1":
            print(f"[FAIL] Official checker failed (rc={chk.returncode}, out={out_val!r})", file=sys.stderr)
            if chk.stderr:
                print(chk.stderr, file=sys.stderr)
            sys.exit(1)

        print(f"[SUCCESS] TII-{sec} equivalent key recovered | lift={lift_elapsed:.2f}s | "
              f"checker={chk_elapsed:.2f}s | RREF_EQ=True | g_irreducible=True | OFFICIAL_CHECK=1", flush=True)
        print(f"Saved equivalent private key to: {sk_path}", flush=True)
        sys.exit(0)

print(f"[FAIL] Could not lift key for TII-{sec}", file=sys.stderr)
sys.exit(1)
