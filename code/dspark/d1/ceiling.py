"""D1-DRAFTADAPT task 1: how much acceptance is achievable at all at T=1 on the captured legal-v2a trajectories.

  python3 -m d1.ceiling --v1 <this-repo>/code/dspark --out d1/results/ceiling.json
  (run from offload/dspark of this worktree; CPU only; reads the proprietary captures, writes aggregates only)

The captures hold no target logits. What they do hold, for every committed decode row, is the mean of the four mHC
streams at the INPUT of target layers 37/38/39 (official model.py: main_hiddens.append(h.mean(dim=2))). Layer 39 is the
last layer, and head.weight + norm.weight are on this host, so a "logit lens" softmax(head(rmsnorm(tap39)*norm)) is a
proxy for the target's next-token distribution that skips only layer 39 and the learned mHC collapse.

Two kinds of number come out:
  * BOUNDS that hold whatever the proxy's quality: for any predictor g computed from the context, the committed token
    (a T=1 sample of p_target) equals g with probability E[p_target(g)] <= E[max p_target]. So the lens' top-1 hit
    rate bounds the per-position ceiling from below, and the lens-greedy chain's committed-prefix match count bounds the
    perfect greedy drafter's accepted/round from below (the chain is itself a greedy drafter: if its first k-1 tokens
    were accepted, the committed prefix IS its draft prefix, so its lens at the next position is the right one).
  * ESTIMATES that assume the temperature-calibrated lens has the target's confidence: E[max p] per position and the
    smoothed perfect-greedy accepted/round sum_k E[prod_{i<k} 1(x_i = g_i) * max p_k].

Every sample is a (context, committed next token) pair, so request-clustered bootstrap intervals are reported; with
four requests the clustered intervals are wide, and say so.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import torch

BLOCK = 5
CKPT = "/data/dsv41-offload/v1/native-1017"
CAPS = "/data/dsv41-offload/v1/captures/A-capture-20261022"
REF_DUMPS = "/data/dsv41-offload/v1/dumps/ref"
NORM_EPS = 1e-20  # config text_config.rms_norm_eps


def load_tensors(names: list[str]) -> dict[str, torch.Tensor]:
    from safetensors import safe_open
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for n in names:
        with safe_open(os.path.join(CKPT, idx[n]), framework="pt") as f:
            out[n] = f.get_tensor(n).float()
    return out


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + NORM_EPS) * w


def boot(vals: list[float], groups: list[str], iters: int = 2000, seed: int = 0) -> dict:
    """Mean with a sample-level and a request-clustered 95% bootstrap interval."""
    rng = random.Random(seed)
    n = len(vals)
    if not n:
        return {"n": 0}
    m = sum(vals) / n
    bs = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    by: dict[str, list[float]] = {}
    for v, g in zip(vals, groups):
        by.setdefault(g, []).append(v)
    gl = list(by.values())
    cs = []
    for _ in range(iters):
        pick = [gl[rng.randrange(len(gl))] for _ in gl]
        tot = sum(len(p) for p in pick)
        cs.append(sum(sum(p) for p in pick) / tot)
    cs.sort()
    return {"n": n, "mean": round(m, 4), "ci95": [round(bs[int(.025 * iters)], 4), round(bs[int(.975 * iters)], 4)],
            "ci95_req_cluster": [round(cs[int(.025 * iters)], 4), round(cs[int(.975 * iters)], 4)]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True, help="V1-VERIFY offload/dspark dir (for verify.captures)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.v1)
    from verify.captures import CaptureSet  # V1's asserted capture semantics, unchanged

    torch.set_num_threads(min(32, os.cpu_count() or 8))
    cs = CaptureSet(CAPS)
    if cs.plumbing:
        raise RuntimeError(f"capture plumbing errors: {cs.plumbing[:3]}")
    reqs = sorted(r for r in cs.rows if "warmup" not in r)  # warmup requests are synthetic: excluded everywhere
    rid = {r: f"req{i}" for i, r in enumerate(reqs)}  # anonymised workload labels
    W = load_tensors(["head.weight", "norm.weight", "mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight"])
    head, normw = W["head.weight"], W["norm.weight"]
    memb, mhead = W["mtp.2.markov_head.embed.weight"], W["mtp.2.markov_head.head.weight"]

    # ---- lens at every committed row with taps whose next position is committed (teacher-forced pairs)
    lens: dict[tuple[str, int], dict] = {}
    keys, X, Y = [], [], []
    for r in reqs:
        rows = cs.rows[r]
        for p, row in rows.items():
            if row.taps is not None and (p + 1) in rows:
                keys.append((r, p))
                X.append(row.taps[2 * 5120:3 * 5120])  # tap of layer 39 input
                Y.append(rows[p + 1].tok)
    Xt = rmsnorm(torch.stack(X), normw)
    Yt = torch.tensor(Y)
    logits = torch.empty(len(keys), head.shape[0])
    for i in range(0, len(keys), 128):
        logits[i:i + 128] = Xt[i:i + 128] @ head.t()
    groups_all = [rid[k[0]] for k in keys]

    def stats_at(tau: float, sel: list[int]) -> dict:
        lg = logits[sel] / tau
        lp = torch.log_softmax(lg.double(), -1)
        pr = lp.exp()
        mx, am = pr.max(-1)
        y = Yt[sel]
        lpy = lp.gather(1, y[:, None]).squeeze(1)
        ent = -(pr * lp).sum(-1)
        return {"top1": (am == y).double(), "maxp": mx, "nll": -lpy, "ent": ent, "argmax": am}

    def fit_tau(sel: list[int]) -> float:
        best = None
        for tau in [0.5 + 0.05 * j for j in range(41)]:
            nll = float(stats_at(tau, sel)["nll"].mean())
            if best is None or nll < best[1]:
                best = (tau, nll)
        return best[0]

    # 2-fold by request: fit tau on two requests, apply to the other two (and vice versa)
    fold = {r: (i % 2) for i, r in enumerate(reqs)}
    taus = {}
    for f in (0, 1):
        taus[f] = fit_tau([i for i, k in enumerate(keys) if fold[k[0]] != f])
    tau_of = [taus[fold[k[0]]] for k in keys]
    allsel = list(range(len(keys)))
    raw = stats_at(1.0, allsel)
    cal_parts = {}
    for f in (0, 1):
        sel = [i for i in allsel if fold[keys[i][0]] == f]
        cal_parts[f] = (sel, stats_at(taus[f], sel))
    cal = {k2: torch.empty(len(keys), dtype=torch.float64) for k2 in ("top1", "maxp", "nll", "ent")}
    for f, (sel, st) in cal_parts.items():
        for k2 in cal:
            cal[k2][sel] = st[k2].double()
    for i, k in enumerate(keys):
        lens[k] = {"argmax": int(raw["argmax"][i]), "maxp_raw": float(raw["maxp"][i]), "maxp_cal": float(cal["maxp"][i]),
                   "top1": int(raw["top1"][i]), "nll_raw": float(raw["nll"][i]), "ent_raw": float(raw["ent"][i]),
                   "nll_cal": float(cal["nll"][i]), "ent_cal": float(cal["ent"][i]), "P": k[1]}

    rep: dict = {"n_lens_pairs": len(keys), "warmup_requests_excluded": sum(1 for r in cs.rows if "warmup" in r), "requests": {rid[r]: {"rows": len(cs.rows[r]),
                 "lens_pairs": sum(1 for k in keys if k[0] == r),
                 "pos_range": [min(cs.rows[r]), max(cs.rows[r])]} for r in reqs},
                 "tau_fit_2fold": {f"fold{f}": round(t, 3) for f, t in taus.items()}}

    def summ(st: dict, sel: list[int] | None = None) -> dict:
        idx = sel if sel is not None else allsel
        g = [groups_all[i] for i in idx]
        return {k2: boot([float(st[k2][i]) for i in idx], g) for k2 in ("top1", "maxp", "nll", "ent")}

    rep["lens_all_positions"] = {"raw_tau1": summ(raw), "calibrated_2fold": summ(cal)}

    # ---- per round: lens-greedy chain vs engine K2 drafts vs native ref drafter (teacher-forced), same trajectories
    samples = []
    for s in cs.samples:
        if any(t is None for t in s.realized):
            continue  # V1's scored set requires all five committed tokens (match_len None otherwise)
        samples.append(s)
    per = []
    for s in samples:
        rec = {"g": rid[s.req], "P": s.P}
        k2 = 0
        while k2 < BLOCK and s.engine_drafts[k2] == s.realized[k2]:
            k2 += 1
        rec["k2"] = k2
        L = [lens.get((s.req, s.P + 1 + i)) for i in range(BLOCK)]
        rec["lens_ok"] = all(x is not None for x in L)
        if rec["lens_ok"]:
            n = 0
            while n < BLOCK and L[n]["argmax"] == s.realized[n]:
                n += 1
            rec["lens_chain"] = n
            surv, sm = 1.0, 0.0
            for i in range(BLOCK):  # smoothed: prob of reaching i (indicator) times calibrated max p at i
                sm += surv * L[i]["maxp_cal"]
                surv *= float(L[i]["argmax"] == s.realized[i])
            rec["lens_smoothed_cal"] = sm
            rec["lens_maxp_cal"] = [x["maxp_cal"] for x in L]
            rec["lens_maxp_raw"] = [x["maxp_raw"] for x in L]
            rec["lens_top1"] = [x["top1"] for x in L]
        # native FP32 reference drafter, teacher-forced on the committed prefix (V1 dumps)
        fp = os.path.join(REF_DUMPS, s.key + ".pt")
        if os.path.exists(fp):
            d = torch.load(fp, map_location="cpu", weights_only=False)
            prev = d["variants"]["realized"]["prev"]
            ql = d["base_logits"] + memb[torch.tensor(prev)] @ mhead.t()
            lq = torch.log_softmax(ql.double(), -1)
            y = torch.tensor(s.realized)
            rec["ref_tf_top1"] = (lq.argmax(-1) == y).int().tolist()
            rec["ref_tf_nll"] = (-lq.gather(1, y[:, None]).squeeze(1)).tolist()
            rec["ref_tf_maxq"] = lq.exp().max(-1).values.tolist()
            rec["ref_tf_ent"] = (-(lq.exp() * lq).sum(-1)).tolist()
            rec["ref_own_chain"] = 0
            while rec["ref_own_chain"] < BLOCK and d["drafts"][rec["ref_own_chain"]] == s.realized[rec["ref_own_chain"]]:
                rec["ref_own_chain"] += 1
            if rec["lens_ok"]:
                # where drafter and lens disagree, which one the target wrote (diagnostic of headroom)
                rec["disagree"] = []
                for i in range(BLOCK):
                    qa, la, t = int(lq[i].argmax()), L[i]["argmax"], s.realized[i]
                    if qa != la:
                        rec["disagree"].append((int(qa == t), int(la == t)))
                rec["kl_lenscal_ref"] = None  # filled below only if cheap; lens full logits not kept per sample
        per.append(rec)

    def acc_table(lens_list: list[int]) -> dict:
        n = len(lens_list)
        surv = [sum(1 for x in lens_list if x >= k) / n for k in range(1, BLOCK + 1)]
        cond = []
        for k in range(BLOCK):
            reach = sum(1 for x in lens_list if x >= k)
            cond.append(round(sum(1 for x in lens_list if x >= k + 1) / reach, 4) if reach else None)
        return {"n": n, "mean_accepted": round(sum(lens_list) / n, 4), "prefix_survival": [round(x, 4) for x in surv],
                "conditional": cond}

    both = [r for r in per if r["lens_ok"] and "ref_tf_top1" in r]
    g = [r["g"] for r in both]
    rep["rounds"] = {"n_scored": len(per), "n_with_lens_and_ref": len(both)}
    rep["rounds"]["engine_k2_all_scored"] = acc_table([r["k2"] for r in per])
    rep["rounds"]["engine_k2"] = acc_table([r["k2"] for r in both])
    rep["rounds"]["ref_own_chain"] = acc_table([r["ref_own_chain"] for r in both])
    rep["rounds"]["lens_chain_LB"] = acc_table([r["lens_chain"] for r in both])
    rep["rounds"]["lens_smoothed_calibrated_EST"] = boot([r["lens_smoothed_cal"] for r in both], g)
    rep["rounds"]["paired_lens_chain_minus_k2"] = boot([r["lens_chain"] - r["k2"] for r in both], g)
    rep["rounds"]["paired_lens_smoothed_minus_k2"] = boot([r["lens_smoothed_cal"] - r["k2"] for r in both], g)
    # teacher-forced per position (every round contributes at every position: unconditional on survival)
    tf = {}
    for name, key in (("ref_top1", "ref_tf_top1"), ("lens_top1", "lens_top1"), ("ref_nll", "ref_tf_nll"),
                      ("ref_maxq", "ref_tf_maxq"), ("ref_entropy", "ref_tf_ent"), ("lens_maxp_cal", "lens_maxp_cal"),
                      ("lens_maxp_raw", "lens_maxp_raw")):
        tf[name] = [round(sum(r[key][i] for r in both) / len(both), 4) for i in range(BLOCK)]
    dis = [x for r in both for x in r["disagree"]]
    tf["disagreements"] = {"n": len(dis), "ref_right": sum(x[0] for x in dis), "lens_right": sum(x[1] for x in dis)}
    rep["teacher_forced_per_position"] = tf
    # union headroom at position 1: either predictor right
    # ---- splits by context length (P) and request
    edges = [0, 2048, 8192, 16384, 32768, 10 ** 9]
    by_len = {}
    for lo, hi in zip(edges, edges[1:]):
        sub = [r for r in both if lo <= r["P"] < hi]
        if len(sub) < 20:
            continue
        by_len[f"P[{lo},{hi})"] = {"n": len(sub), "k2": round(sum(r["k2"] for r in sub) / len(sub), 3),
                                   "lens_chain_LB": round(sum(r["lens_chain"] for r in sub) / len(sub), 3),
                                   "lens_smoothed_EST": round(sum(r["lens_smoothed_cal"] for r in sub) / len(sub), 3),
                                   "lens_maxp_cal_pos1": round(sum(r["lens_maxp_cal"][0] for r in sub) / len(sub), 3)}
    rep["by_context_length"] = by_len
    by_req = {}
    for rq in sorted(set(r["g"] for r in both)):
        sub = [r for r in both if r["g"] == rq]
        by_req[rq] = {"n": len(sub), "k2": round(sum(r["k2"] for r in sub) / len(sub), 3),
                      "lens_chain_LB": round(sum(r["lens_chain"] for r in sub) / len(sub), 3),
                      "lens_smoothed_EST": round(sum(r["lens_smoothed_cal"] for r in sub) / len(sub), 3),
                      "P_median": sorted(r["P"] for r in sub)[len(sub) // 2]}
    rep["by_request"] = by_req
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
