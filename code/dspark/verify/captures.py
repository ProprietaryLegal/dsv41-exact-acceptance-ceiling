"""Turn the rank-0 DSpark capture rounds (offload/dspark/capture/speculator.py overlay) into replayable samples.

Proprietary input (legal-v2a taps, token ids, drafts): read only on x86host or the controller host, never copied elsewhere.

Field semantics, re-derived here and asserted on every round (a violation is a plumbing error, never skipped quietly):
  * a round's target rows for request b are target_*[qsl[b]:qsl[b+1]]; the first nv = rows - num_rejected[b] are
    verified (committed) and the token fed at position p is the committed token at p;
  * the query block is [anchor, noise x 4] at positions P+1..P+5, P = the last committed position, anchor = the target's
    token at P+1 (it reappears as the first target row of the request's next round);
  * draft_tokens[b, i] predicts the token at P+2+i;
  * main_x rows are the engine drafter's main_norm(main_proj(taps)) in BF16; taps (three [rows, 5120] BF16 tensors for
    target layers 37/38/39) exist only for rounds with num_tokens <= 64.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import torch

NOISE_TOKEN = 128799
BLOCK = 5
WINDOW = 128


@dataclass
class Row:
    tok: int
    round: int
    mx: torch.Tensor                  # engine main_x [5120] bf16
    taps: torch.Tensor | None         # [15360] bf16 (cat 37/38/39) or None


@dataclass
class Sample:
    round: int
    b: int
    req: str
    B: int
    num_tokens: int
    P: int
    anchor: int
    engine_drafts: list[int]
    engine_conf: list[float] | None
    temperature: float
    realized: list[int | None] = field(default_factory=list)   # committed tokens at P+2..P+6 (None = beyond capture)
    engine_scheduled: int | None = None                        # drafts the next round verified (1..5)
    engine_accepted: int | None = None                         # of those, accepted (from next round's num_rejected)

    @property
    def key(self) -> str:
        return f"r{self.round:06d}_b{self.b}"


class CaptureSet:
    def __init__(self, cap_dir: str):
        self.files = sorted(glob.glob(os.path.join(cap_dir, "round*.pt")))
        if not self.files:
            raise FileNotFoundError(f"no round*.pt under {cap_dir}")
        self.rows: dict[str, dict[int, Row]] = {}
        self.samples: list[Sample] = []
        self.plumbing: list[str] = []
        self.unverified_tail: list[str] = []
        self.next_round_rows: dict[tuple[str, int], tuple[int, int]] = {}  # (req, first pos) -> (rows, nrej)
        self._load()

    def _err(self, msg: str) -> None:
        self.plumbing.append(msg)

    def _load(self) -> None:
        for f in self.files:
            r = torch.load(f, map_location="cpu", weights_only=False)
            rnd = int(r["round"])
            reqs = list(r["req_ids"])
            nr = len(reqs)
            k, nq = int(r["num_speculative_steps"]), int(r["num_query_per_req"])
            if not r["sample_from_anchor"] or nq != k or k != BLOCK:
                raise RuntimeError(f"round {rnd}: sample_from_anchor {r['sample_from_anchor']} nq {nq} k {k}")
            qsl = [int(x) for x in r["query_start_loc"]]
            if qsl[-1] != int(r["num_tokens"]):
                raise RuntimeError(f"round {rnd}: query_start_loc end {qsl[-1]} != num_tokens {r['num_tokens']}")
            tids, tpos = r["target_input_ids"].long(), r["target_positions"].long()
            mx = r["main_x"]
            taps = torch.cat(list(r["taps"]), dim=-1) if r["taps"] is not None else None
            nrej, nsam = r["num_rejected"].tolist(), r["num_sampled"].tolist()
            qids = r["query_input_ids"].view(nr, nq).long()
            qpos = r["query_positions"].view(nr, nq).long()
            eng = r["draft_tokens"].long()
            conf = r["confidence"]
            temp = r["temperature"].reshape(-1).tolist()
            for b, req in enumerate(reqs):
                st, en = qsl[b], qsl[b + 1]
                nv = en - st - nrej[b]
                if nv < 0:
                    raise RuntimeError(f"round {rnd} req {b}: {en - st} rows, {nrej[b]} rejected")
                self.next_round_rows[(req, int(tpos[st]))] = (en - st, nrej[b])
                rows = self.rows.setdefault(req, {})
                for j in range(st, st + nv):
                    p, t = int(tpos[j]), int(tids[j])
                    old = rows.get(p)
                    if old is not None and old.tok != t:
                        self._err(f"round {rnd} {req[-12:]} pos {p}: committed token {t} != earlier {old.tok}")
                    if old is None:
                        rows[p] = Row(t, rnd, mx[j].clone(), taps[j].clone() if taps is not None else None)
                    elif old.taps is None and taps is not None:
                        old.taps = taps[j].clone()
                P = int(qpos[b, 0]) - 1
                exp_q = torch.cat((qids[b, :1], torch.full((nq - 1,), NOISE_TOKEN)))
                if not torch.equal(qids[b], exp_q):
                    self._err(f"round {rnd} req {b}: query ids {qids[b].tolist()}")
                if not torch.equal(qpos[b], P + 1 + torch.arange(nq)):
                    self._err(f"round {rnd} req {b}: query positions {qpos[b].tolist()}")
                if nv and int(tpos[st + nv - 1]) != P:
                    self._err(f"round {rnd} req {b}: last committed pos {int(tpos[st + nv - 1])} != P {P}")
                if "warmup" in req or nsam[b] <= 0:
                    continue
                self.samples.append(Sample(rnd, b, req, nr, int(r["num_tokens"]), P, int(qids[b, 0]),
                                           eng[b].tolist(), conf[b].tolist() if conf is not None else None,
                                           float(temp[b]) if b < len(temp) else float("nan")))
        for s in self.samples:
            rows = self.rows[s.req]
            s.realized = [rows[p].tok if p in rows else None for p in range(s.P + 2, s.P + 2 + BLOCK)]
            a = rows.get(s.P + 1)
            if a is not None and a.tok != s.anchor:
                self._err(f"{s.key} {s.req[-12:]}: anchor {s.anchor} != committed {a.tok} at P+1")
            nxt = self.next_round_rows.get((s.req, s.P + 1))
            if nxt is not None:
                n_rows, n_rej = nxt
                s.engine_scheduled = n_rows - 1
                s.engine_accepted = n_rows - 1 - n_rej
                acc = 0
                while acc < min(s.engine_scheduled, BLOCK) and s.realized[acc] is not None \
                        and s.realized[acc] == s.engine_drafts[acc]:
                    acc += 1
                if acc < s.engine_accepted:  # an accepted draft must be the committed token
                    self._err(f"{s.key} {s.req[-12:]}: next round accepted {s.engine_accepted} of "
                              f"{s.engine_scheduled}, drafts vs committed give only {acc}")
                elif acc > s.engine_accepted:
                    # The committed token at the first non-accepted position equals the draft: legal only when that
                    # draft was not verified (adaptive verification turns it into a -1 placeholder and the target
                    # samples from its full distribution); a verified rejection resamples with the draft excluded.
                    self.unverified_tail.append(s.key)

    def window(self, s: Sample) -> list[int]:
        return list(range(max(0, s.P - WINDOW + 1), s.P + 1))

    def window_status(self, s: Sample) -> dict:
        rows = self.rows[s.req]
        pos = self.window(s)
        missing = [p for p in pos if p not in rows or rows[p].round > s.round]
        no_taps = [p for p in pos if p in rows and rows[p].taps is None]
        return {"n": len(pos), "missing": missing, "no_taps": no_taps}

    def arm_inputs(self, s: Sample) -> dict:
        """The exact inputs every implementation under test must use for sample s (the fidelity harness contract):
        window rows for positions max(0,P-127)..P in order, each ("taps", [15360] bf16: compute main_x yourself) or
        ("engine_mx", [5120] bf16: inject as main_x unchanged); the anchor token at P+1; block positions P+1..P+5."""
        st = self.window_status(s)
        if st["missing"]:
            raise RuntimeError(f"{s.key}: window positions without a verified row: {st['missing'][:5]}")
        rows = self.rows[s.req]
        win = [(p, "taps", rows[p].taps) if rows[p].taps is not None else (p, "engine_mx", rows[p].mx)
               for p in self.window(s)]
        return {"key": s.key, "req": s.req, "P": s.P, "anchor": s.anchor, "window": win,
                "block_positions": list(range(s.P + 1, s.P + 1 + BLOCK)), "noise_token": NOISE_TOKEN}
