#!/usr/bin/env python3
"""
Export the shortened parity-check matrix H_A = H_pub[:, :nprime] of a TII Track-2
instance into the LSB-first hexadecimal row format expected by `minwords_gpu_v4`
(matching MiniMax's `_write_parity` convention).

Usage:
  python3 src/export_ha.py <sec> <nprime> <out_path> [--pk-file <local_pk.txt>]

Example:
  python3 src/export_ha.py 237 378 ha_237.txt
  python3 src/export_ha.py 245 326 ha_245.txt
"""
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/ElenaKirshanova/tii_decoding_challenge/main/public_keyRec"


def gf2_rank(rows_int, ncols):
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
    return r


def main():
    parser = argparse.ArgumentParser(description="Export shortened TII parity-check matrix H_A to hex.")
    parser.add_argument("sec", type=int, help="TII challenge number (e.g., 245 or 237)")
    parser.add_argument("nprime", type=int, help="Shortened column count n' (prefix columns 0..n'-1)")
    parser.add_argument("out_path", help="Output hex matrix path (e.g., ha_237.txt)")
    parser.add_argument("--pk-file", default=None, help="Optional local pk_McEliece_<sec>.txt path")
    args = parser.parse_args()

    if args.pk_file:
        pk_path = args.pk_file
    else:
        cache_dir = os.path.join(tempfile.gettempdir(), "tii_pk_cache")
        os.makedirs(cache_dir, exist_ok=True)
        pk_path = os.path.join(cache_dir, f"pk_McEliece_{args.sec}.txt")
        if not os.path.isfile(pk_path):
            urllib.request.urlretrieve(f"{BASE_URL}/pk_McEliece_{args.sec}.txt", pk_path)

    with open(pk_path, "rb") as f:
        pk_bytes = f.read()
    pk_sha = hashlib.sha256(pk_bytes).hexdigest()
    lines = pk_bytes.decode("utf-8").splitlines()

    # Matrix rows precede the final field-defining polynomial line
    mat_lines = [ln for ln in lines[:-1] if ln.strip()]
    mt = len(mat_lines)
    rows_int = []
    n_full = None
    for idx, ln in enumerate(mat_lines):
        bits = [int(x) for x in ln.strip("[] ").split() if x]
        if n_full is None:
            n_full = len(bits)
        elif len(bits) != n_full:
            raise RuntimeError(f"Inconsistent row length at row {idx}")
        if args.nprime > n_full:
            raise ValueError(f"n'={args.nprime} exceeds n={n_full}")
        v = 0
        for j in range(args.nprime):
            if bits[j]:
                v |= 1 << j
        rows_int.append(v)

    rank = gf2_rank(rows_int, args.nprime)
    if rank != mt:
        raise RuntimeError(f"Shortened H_A has rank {rank} != {mt}")

    hex_digits = (args.nprime + 3) // 4
    with open(args.out_path, "w", encoding="utf-8") as out:
        for v in rows_int:
            out.write(f"{v:0{hex_digits}x}\n")

    print(f"Exported TII-{args.sec} H_A ({mt} x {args.nprime}, rank={rank}) -> {args.out_path}")
    print(f"Source pk SHA-256: {pk_sha}")


if __name__ == "__main__":
    main()
