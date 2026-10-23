"""D1 Phase 0: the EXACT T=1 acceptance ceiling on the captured trajectories, from the target's own top-20 logprobs.

The captures hold every committed token of each request (prompt rows from prefill rounds, generated rows from decode
rounds), so replaying each full sequence once through the served model with vLLM `prompt_logprobs=20` returns, at every
position, the target's top-20 next-token logprobs on the committed prefix -- exactly the distributions the drafter is
verified against. Nothing in the engine changes; one prefill per request (~26K tokens in all).

  build  (the controller host, CPU):  python3 -m d1.exact_ceiling build --v1 <v1 offload/dspark> --dir <private dir, mode 700>
         writes <dir>/reqN.request.json (token-id prompts: PROPRIETARY, never leaves x86host/the controller host/Sparks)
  replay (coordinator, Spark window; see the progress file "needs GPU/Spark"):
         curl -s http://<serve>/v1/completions -H 'Content-Type: application/json' -d @reqN.request.json > reqN.response.json
  score  (the controller host, CPU):  python3 -m d1.exact_ceiling score --v1 <v1 offload/dspark> --dir <dir> --out <aggregates.json>

Scored quantities (per round sample s, draft position i predicts the token at P+2+i from the prefix through P+1+i):
  * E[max p] per position (unconditional) and conditional on a perfect greedy drafter surviving to it;
  * perfect greedy drafter accepted/round, smoothed (unbiased): sum_k E[prod_{i<k} 1(x_i = g_i) * max p_k], g = argmax p;
  * the K2 drafts' expected accepted/round with the same smoothing: sum_k E[prod_{i<k} 1(x_i = d_i) * p(d_k)]
    (p(d_k) = 0 when d_k is outside the top-20, so this is a lower bound; the residual mass is reported);
  * KL(p_target || q_native) per position on the top-20 support plus one tail bucket (q from V1's FP32 ref dumps).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

BLOCK = 5
CAPS = "/data/dsv41-offload/v1/captures/A-capture-20261022"
REF_DUMPS = "/data/dsv41-offload/v1/dumps/ref"
CKPT = "/data/dsv41-offload/v1/native-1017"
TOPK = 20


def load_caps(v1: str):
    sys.path.insert(0, v1)
    from verify.captures import CaptureSet
    cs = CaptureSet(CAPS)
    if cs.plumbing:
        raise RuntimeError(f"capture plumbing errors: {cs.plumbing[:3]}")
    reqs = sorted(r for r in cs.rows if "warmup" not in r)
    return cs, reqs


def build(a) -> None:
    cs, reqs = load_caps(a.v1)
    os.makedirs(a.dir, mode=0o700, exist_ok=True)
    os.chmod(a.dir, 0o700)
    for i, r in enumerate(reqs):
        rows = cs.rows[r]
        n = max(rows) + 1
        if sorted(rows) != list(range(n)):
            raise RuntimeError(f"req{i}: committed positions are not contiguous 0..{n - 1}")
        body = {"model": a.model, "prompt": [rows[p].tok for p in range(n)], "max_tokens": 1, "temperature": 1.0,
                "top_p": 1.0, "prompt_logprobs": TOPK, "cache_salt": f"d1-exact-{i}", "priority": 5}
        fp = os.path.join(a.dir, f"req{i}.request.json")
        with open(fp, "w") as fh:
            json.dump(body, fh)
        os.chmod(fp, 0o600)
        print(f"req{i}: {n} tokens -> {fp}")


def parse_response(fp: str) -> list[dict[int, float] | None]:
    r = json.load(open(fp))
    pl = r["choices"][0]["prompt_logprobs"]
    out = []
    for e in pl:
        out.append(None if e is None else {int(k): float(v["logprob"]) for k, v in e.items()})
    return out


def score(a) -> None:
    cs, reqs = load_caps(a.v1)
    from safetensors import safe_open
    idx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
    mt = {}
    for n in ("mtp.2.markov_head.embed.weight", "mtp.2.markov_head.head.weight"):
        with safe_open(os.path.join(CKPT, idx[n]), framework="pt") as f:
            mt[n.split(".")[-2]] = f.get_tensor(n).float()
    dist = {}
    for i, r in enumerate(reqs):
        lp = parse_response(os.path.join(a.dir, f"req{i}.response.json"))
        if len(lp) != max(cs.rows[r]) + 1:
            raise RuntimeError(f"req{i}: {len(lp)} prompt_logprobs entries for {max(cs.rows[r]) + 1} tokens")
        for p, e in enumerate(lp):
            if e is not None:
                if cs.rows[r][p].tok not in e:
                    raise RuntimeError(f"req{i} pos {p}: response omits the prompt token's own logprob")
                dist[(r, p)] = e  # p_target(. | tokens < p), top-20 plus the actual token
    per = []
    for s in cs.samples:
        if any(t is None for t in s.realized):
            continue
        D = [dist.get((s.req, s.P + 2 + i)) for i in range(BLOCK)]
        if any(d is None for d in D):
            raise RuntimeError(f"{s.key}: missing target distribution")
        rec = {"g": reqs.index(s.req), "P": s.P}
        top = [max(d.items(), key=lambda kv: kv[1]) for d in D]
        mx = [math.exp(v) for _, v in top]
        hit = [int(top[i][0] == s.realized[i]) for i in range(BLOCK)]
        dhit = [int(s.engine_drafts[i] == s.realized[i]) for i in range(BLOCK)]
        pd = [math.exp(D[i][s.engine_drafts[i]]) if s.engine_drafts[i] in D[i] else 0.0 for i in range(BLOCK)]
        sm_g = sm_d = 0.0
        sg = sd = 1.0
        rec["reach_g"] = []
        for i in range(BLOCK):
            rec["reach_g"].append(sg)
            sm_g += sg * mx[i]
            sm_d += sd * pd[i]
            sg *= hit[i]
            sd *= dhit[i]
        rec.update(maxp=mx, perfect_smoothed=sm_g, k2_smoothed=sm_d,
                   resid=[1.0 - sum(math.exp(v) for v in d.values()) for d in D])
        fp = os.path.join(REF_DUMPS, s.key + ".pt")
        if os.path.exists(fp):
            dm = torch.load(fp, map_location="cpu", weights_only=False)
            prev = dm["variants"]["realized"]["prev"]
            lq = torch.log_softmax((dm["base_logits"] + mt["embed"][torch.tensor(prev)] @ mt["head"].t()).double(), -1)
            kls = []
            for i in range(BLOCK):
                ids = list(D[i])
                lpt = torch.tensor([D[i][t] for t in ids], dtype=torch.float64)
                lqt = lq[i, ids]
                tail_p = max(1e-12, 1.0 - float(lpt.exp().sum()))
                tail_q = max(1e-12, 1.0 - float(lqt.exp().sum()))
                kls.append(float((lpt.exp() * (lpt - lqt)).sum()) + tail_p * math.log(tail_p / tail_q))
            rec["kl_target_native"] = kls
        per.append(rec)
    n = len(per)

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    rep = {"n_rounds": n,
           "maxp_unconditional": [mean([r["maxp"][i] for r in per]) for i in range(BLOCK)],
           "maxp_given_perfect_survival": [mean([r["maxp"][i] for r in per if r["reach_g"][i]]) for i in range(BLOCK)],
           "perfect_greedy_accepted_per_round": mean([r["perfect_smoothed"] for r in per]),
           "k2_expected_accepted_per_round_lb": mean([r["k2_smoothed"] for r in per]),
           "residual_mass_outside_top20_mean": [mean([r["resid"][i] for r in per]) for i in range(BLOCK)],
           "kl_target_vs_native_per_position": [mean([r["kl_target_native"][i] for r in per if "kl_target_native" in r])
                                                for i in range(BLOCK)]}
    rep["gap_perfect_minus_k2"] = round(rep["perfect_greedy_accepted_per_round"] - rep["k2_expected_accepted_per_round_lb"], 4)
    rep["by_request"] = {f"req{g}": {"n": len(sub), "perfect": mean([r["perfect_smoothed"] for r in sub]),
                                     "k2": mean([r["k2_smoothed"] for r in sub])}
                         for g in sorted(set(r["g"] for r in per)) for sub in [[r for r in per if r["g"] == g]]}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--v1", required=True)
    b.add_argument("--dir", required=True)
    b.add_argument("--model", default="DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2", help="served model name (RUNBOOK; GET /v1/models)")
    s = sp.add_parser("score")
    s.add_argument("--v1", required=True)
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a) if a.cmd == "build" else score(a)


if __name__ == "__main__":
    main()
