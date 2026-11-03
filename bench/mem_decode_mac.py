"""Decode-step KV-cache MEMORY benchmark on a MacBook (MPS) — capacity vs bandwidth.

Answers the reviewer point (Shaowen Peng, AMD Dev Community): the compression ratio is not
the benchmark, the dequant path is. A per-channel int8 dequant that materialises the fp16
key/value adds a transient copy; what matters is PEAK live memory across a decode step, not
the steady-state packed size.

This measures, with REAL MPS allocations and the real vramkit quantiser (per-channel K /
per-token V), for a chosen attention config at a chosen context length:
  - resident KV bytes, fp16 vs int8 (exact, from tensor sizes)  -> the capacity ceiling
  - peak live bytes across one decode step, three strategies:
      fp16            : read the cache directly (baseline)
      int8 streamed   : dequant ONE layer at a time, attend, free  (realistic)
      int8 full-dequant: dequant ALL layers up front (the anti-pattern Shaowen warns about)
  - per-step wall time for each (a tok/s proxy; MPS, not the ROCm number)

Honesty notes printed by the script:
  - int4 `codes` are stored uint8 (not nibble-packed) in this codebase, so int4 resident == int8
    resident here; int4's 4x is on-wire only until the codes are bit-packed.
  - dequantize_kv returns fp32; we cast the transient to fp16 (what attention actually runs in).
"""
import argparse, time
import torch
import torch.nn.functional as Fn
from vramkit.kvquant import quantize_kv, dequantize_kv

GiB = 1024 ** 3
MiB = 1024 ** 2


def exact_bytes(tensors):
    return sum(t.element_size() * t.nelement() for t in tensors)


def packed_resident(packs):
    tot = 0
    for p in packs:
        tot += p.codes.element_size() * p.codes.nelement()   # uint8 codes
        tot += p.scale.element_size() * p.scale.nelement()   # fp16 scales
    return tot


def live(dev):
    if dev == "mps":
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory()
    if dev == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()
    return 0


def attn(q, Kd, Vd):
    """Real one-layer attention via SDPA (GQA handled internally, no q-head expansion).
    q: (B,QH,1,D)  Kd,Vd: (B,H,S,D) -> out (B,QH,1,D)."""
    return Fn.scaled_dot_product_attention(q, Kd, Vd, enable_gqa=(q.shape[1] != Kd.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=28)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--q-heads", type=int, default=28)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seq", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--name", default="Qwen2.5-7B-like (GQA)")
    a = ap.parse_args()

    dev, dt = a.device, torch.float16
    B, H, QH, D, S, L = a.batch, a.kv_heads, a.q_heads, a.head_dim, a.seq, a.layers
    rep = QH // H
    scale = D ** -0.5
    torch.manual_seed(0)
    print(f"\n=== {a.name}  |  {L} layers, {QH}q/{H}kv heads, d={D}, ctx={S}, batch={B}, "
          f"int{a.bits}, device={dev} ===")

    if dev == "mps":
        torch.mps.empty_cache()
    base0 = live(dev)

    # ---- build fp16 KV cache with persistent per-channel key outliers ----
    K_fp, V_fp = [], []
    for _ in range(L):
        k = torch.randn(B, H, S, D, device=dev, dtype=dt)
        oc = torch.randperm(D)[: max(1, D // 32)]
        k[..., oc] *= 12.0                        # massive-activation channels on keys
        v = torch.randn(B, H, S, D, device=dev, dtype=dt)
        K_fp.append(k); V_fp.append(v)
    resident_fp16 = exact_bytes(K_fp) + exact_bytes(V_fp)

    # ---- quantize: per-channel K (over seq axis=2), per-token V ----
    PK = [quantize_kv(k, bits=a.bits, per="channel", token_dim=2) for k in K_fp]
    PV = [quantize_kv(v, bits=a.bits, per="token") for v in V_fp]
    resident_int = packed_resident(PK) + packed_resident(PV)

    q = torch.randn(B, QH, 1, D, device=dev, dtype=dt)

    # ---- PEAK: fp16 baseline (read cache directly, per layer) ----
    if dev == "mps":
        torch.mps.empty_cache()
    peak_fp16 = live(dev)
    t0 = time.perf_counter()
    for _ in range(a.steps):
        for l in range(L):
            peak_fp16 = max(peak_fp16, live(dev))        # K,V in hand (fp16, no transient)
            o = attn(q, K_fp[l], V_fp[l])
            peak_fp16 = max(peak_fp16, live(dev)); del o
    t_fp16 = (time.perf_counter() - t0) / a.steps

    # free fp16, keep only int8 cache resident
    del K_fp, V_fp
    if dev == "mps":
        torch.mps.empty_cache()

    # ---- PEAK: int8 streamed (dequant one layer at a time) ----
    peak_stream = live(dev)
    t0 = time.perf_counter()
    for _ in range(a.steps):
        for l in range(L):
            Kd = dequantize_kv(PK[l]).to(dt)
            Vd = dequantize_kv(PV[l]).to(dt)
            peak_stream = max(peak_stream, live(dev))     # dequant transient alive
            o = attn(q, Kd, Vd)
            peak_stream = max(peak_stream, live(dev))
            del Kd, Vd, o
    t_stream = (time.perf_counter() - t0) / a.steps

    # ---- PEAK: int8 full-dequant (anti-pattern: all layers up front) ----
    if dev == "mps":
        torch.mps.empty_cache()
    t0 = time.perf_counter()
    Kd_all = [dequantize_kv(p).to(dt) for p in PK]
    Vd_all = [dequantize_kv(p).to(dt) for p in PV]
    peak_full = live(dev)                                 # all transients alive
    for l in range(L):
        o = attn(q, Kd_all[l], Vd_all[l]); peak_full = max(peak_full, live(dev)); del o
    t_full = time.perf_counter() - t0
    del Kd_all, Vd_all
    if dev == "mps":
        torch.mps.empty_cache()

    # ---- report ----
    def gb(x):
        return x / GiB
    cap_ratio = resident_fp16 / resident_int
    layer_fp16 = 2 * B * H * S * D * 2                 # one layer's K+V in fp16 (stream transient)
    foot_fp16 = resident_fp16                          # exact decode-step KV footprint
    foot_stream = resident_int + layer_fp16
    foot_full = resident_int + resident_fp16
    print("\nRESIDENT KV cache (exact, from tensor sizes):")
    print(f"  fp16        : {gb(resident_fp16):6.3f} GiB")
    print(f"  int{a.bits} packed : {gb(resident_int):6.3f} GiB   -> {cap_ratio:.2f}x smaller")
    print("\nDECODE-STEP KV FOOTPRINT (exact: resident cache + dequant transient):")
    print(f"  fp16 baseline        : {gb(foot_fp16):6.3f} GiB")
    print(f"  int{a.bits} streamed/layer  : {gb(foot_stream):6.3f} GiB   -> {foot_fp16/foot_stream:.2f}x less than fp16"
          f"  (transient = 1 layer = {layer_fp16/MiB:.0f} MiB)")
    print(f"  int{a.bits} full-dequant    : {gb(foot_full):6.3f} GiB   (>= fp16: ERASES the win — the anti-pattern)")
    print(f"  MPS-allocator cross-check (incl. attention scratch): "
          f"fp16 {gb(peak_fp16):.2f} / stream {gb(peak_stream):.2f} / full {gb(peak_full):.2f} GiB")
    print(f"\nper-step wall time (MPS, {a.steps} steps; dequant COMPUTE overhead, NOT the ROCm bandwidth number):")
    print(f"  fp16 {t_fp16*1000:7.1f} ms   int{a.bits} streamed {t_stream*1000:7.1f} ms   "
          f"(int{a.bits} {t_stream/max(t_fp16,1e-9):.2f}x fp16 time)")
    print(f"\nVERDICT: streaming int{a.bits} per layer keeps a real {foot_fp16/foot_stream:.2f}x capacity win "
          f"(peak = cache + one dequantised layer, {layer_fp16/MiB:.0f} MiB). Dequantising the whole cache at "
          f"once erases it. On MPS the dequant adds {t_stream/max(t_fp16,1e-9):.1f}x compute; the bandwidth "
          f"win needs a memory-bound GPU (MI300X).")
    if a.bits == 4:
        print("NOTE: int4 codes are uint8 here (not nibble-packed) -> int4 resident == int8; "
              "the 4x is on-wire only until codes are bit-packed.")


if __name__ == "__main__":
    main()
