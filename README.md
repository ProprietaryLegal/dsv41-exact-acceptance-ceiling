# DeepSeek-V4.1-Flash exact speculative acceptance ceiling: DGX Spark / GB10 target and x86 RTX 3090 drafter

Measure the temperature-one acceptance headroom of a greedy speculative drafter using the target's own probabilities on captured committed trajectories. This repository contains the method and aggregate results only. Captures, prompts, tokens, reference dumps, and weights are not included.

**Hugging Face:** [Card and files](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling) · **Stack overview:** [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · **Collection:** [All entries](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e)

<!-- pli:blurb:start -->

> Published by [PLI Labs](https://proprietarylegal.ai/labs.html), the open research arm of [Proprietary Legal Intelligence](https://proprietarylegal.ai). PLI builds AI systems around a legal organization's own work product, combining retrieval, staged workflows, validation, and attorney review. Visit [proprietarylegal.ai](https://proprietarylegal.ai) to explore our work and discuss what your organization needs.

<!-- pli:blurb:end -->

## Status

2026-10-23 — research-only: target-logprob replay and aggregate scoring are complete; drafter adaptation and a resulting serving speedup are unqualified.

The word “exact” refers to probabilities returned by the served target on the recorded prefixes. The reported ceiling is a finite-sample estimate for an oracle target-greedy drafter, not a population guarantee or a measured throughput gain.

## What it is

The deployed drafter proposes greedy tokens. At `T = 1`, verification accepts a proposed token `d` with probability `p_target(d | prefix)`. A target-greedy oracle proposes `argmax p_target`, whose acceptance probability at that position is `max p_target`.

For each captured request, `d1/exact_ceiling.py build` reconstructs its contiguous committed token sequence and creates a completion request with `prompt_logprobs: 20`. Replay it against the same served target, tokenizer, and sampling contract. One prefill per sequence returns next-token logprobs on the committed prefixes, including the observed token's logprob.

For position `k`, let `x_k` be the committed token, `g_k` the target argmax and `d_k` the captured draft. The scorer averages these smoothed sums across rounds:

```text
oracle:   sum_k [ product_{i<k} 1(x_i = g_i) ] * max p_k
captured: sum_k [ product_{i<k} 1(x_i = d_i) ] * p_k(d_k)
```

The preceding-match indicator restricts each term to prefixes the corresponding draft chain would have reached. Committed trajectories sampled from the target provide the survival observations. This avoids multiplying marginal mean probabilities as though positions were independent. A captured draft absent from the returned support receives probability zero, making its score a lower bound.

The KL diagnostic compares the target with a teacher-forced FP32 native reference drafter. It uses the returned support plus one residual-mass bucket, rather than claiming full-vocabulary KL. It is not KL against the captured quantized drafter.

### The cheaper logit lens

`d1/ceiling.py` applies the target's final normalization and output head to the input of its last layer. That skips the last layer and learned stream collapse, so it is a proxy. Its committed-token hit rate and greedy-chain score are the reported empirical floors; its temperature-calibrated confidence produces an assumption-dependent estimate. Calibration holds out whole requests. Exact target replay supersedes that estimate, rather than validating its confidence assumption.

### Request-clustered uncertainty

The lens tool bootstraps both individual observations and whole requests. For the clustered interval it resamples requests with replacement, keeps every observation of each selected request, then computes the pooled, round-weighted mean. Use paired per-round differences before resampling to compare predictors. Only four requests were available, so clustered intervals are wide. `exact_ceiling.py` reports means and per-request aggregates; it does not compute an exact-replay confidence interval.

## Why

Low acceptance can come from target sampling entropy or drafter mismatch. Measuring target probabilities separates recoverable drafting headroom from the acceptance lost even by a target-greedy oracle. The later draft positions show the larger mismatch, where the drafter predicts beyond its last observed target state.

## When to use it

Use it to decide whether workload adaptation is worth investigating, compare a drafter with its target on identical committed prefixes, and identify positional mismatch before spending time on training. Bring your own captures and target access.

## When not to use it

Do not use these aggregates as evidence that adaptation improves serving, as a general ceiling for arbitrary probabilistic draft policies, or as a benchmark across unrelated targets. The estimator assumes greedy drafts and the stated target-sampling contract. Free-running text agreement alone does not certify numerical fidelity.

## Requirements

CPU Python with PyTorch and safetensors; private trusted capture files; access to the original target's completion API for replay; matching tokenizer and weights; target normalization/head tensors for the lens; native Markov tensors and reference dumps for KL.

The supplied scripts have module-level `CAPS`, `CKPT`, and `REF_DUMPS` paths. Set them explicitly for your data, as below. The `--v1` option selects the directory containing `verify/`; it does not select captures or weights. No installation manifest or private reference generator is bundled.

### Capture format

`verify/captures.py` loads trusted PyTorch dictionaries named `round*.pt` on CPU. It uses `torch.load(..., weights_only=False)`, so only load files you created or trust. The required fields are:

| field | shape or meaning |
|---|---|
| `round`, `req_ids` | round number and ordered request identifiers |
| `num_speculative_steps`, `num_query_per_req`, `sample_from_anchor` | both counts must equal 5; anchor sampling must be enabled |
| `num_tokens`, `query_start_loc` | target row count and request boundaries, length batch size + 1; last boundary equals row count |
| `target_input_ids`, `target_positions` | flattened target tokens and absolute positions |
| `main_x` | target-row-aligned `[rows, 5120]` BF16 conditioning |
| `taps` | three `[rows, 5120]` BF16 tensors from layer inputs 37/38/39, or `None` |
| `num_rejected`, `num_sampled` | per-request rejection and sampling counts |
| `query_input_ids`, `query_positions` | reshape to `[batch, 5]`; anchor followed by four noise tokens; positions `P+1..P+5` |
| `draft_tokens`, `confidence` | `[batch, 5]` drafts; optional matching confidences |
| `temperature` | per-request temperatures |

The noise token is `128799`. Draft index `i` predicts the committed token at `P+2+i`, where `P` is the last committed target position. Verified rows exclude `num_rejected` tail rows. Repeated committed positions must agree. Requests containing `warmup` in their identifier are excluded from samples. Scoring excludes rounds without the complete committed future block. Replay building requires every committed sequence to start at position zero without holes.

Reference dumps are named from `Sample.key`, for example `r000000_b0.pt`. The exact scorer reads `base_logits` and `variants.realized.prev`; the lens additionally reads `drafts`. KL is omitted for a round with no reference dump. Check coverage yourself before comparing KL summaries.

## Measured effect

Exact replay: captured 2026-10-22, replay results reported 2026-10-23; public [EXL3 2.75 bpw target](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-1017-EXL3-2.75bpw-LegalWeighted-DGX-Spark) on two DGX Spark / GB10 nodes, EXL3 drafter captured on the Sparks in the stock baseline (the x86 RTX 3090 offload was not the capture source); `T = 1`, five drafts per round, four requests. The captured sequences contain 523–8,402 tokens, with positions indexed from zero. Concurrency was not stratified in these aggregates. Each recorded round is one observation; these are not repeated independent serving A/B trials.

| result | value | observations |
|---|---:|---|
| target-greedy oracle expected accepted drafts/round | 3.151 | 529 scored rounds |
| captured baseline drafter expected accepted drafts/round, lower bound | 1.831 | same 529 rounds |
| reported oracle minus lower-bound gap | 1.32 | same 529 rounds; upper estimate of the gap to the captured drafter expectation |
| request A: oracle / captured lower bound | 3.4017 / 1.9051 | 148 rounds |
| request B: oracle / captured lower bound | 3.029 / 1.9048 | 107 rounds |
| request C: oracle / captured lower bound | 3.4315 / 2.3764 | 96 rounds |
| request D: oracle / captured lower bound | 2.8645 / 1.431 | 178 rounds |

All positional rows use the same 529 scored rounds. KL uses the reference dumps that are available; its coverage count is not reported.

| draft position | mean target max probability | residual mass outside returned support | KL(target ‖ native reference), nats |
|---|---:|---:|---:|
| 1 | 0.8275 | 0.0018 | 0.6505 |
| 2 | 0.8386 | 0.0011 | 1.4726 |
| 3 | 0.8363 | 0.0015 | 1.9825 |
| 4 | 0.833 | 0.0019 | 2.8272 |
| 5 | 0.8353 | 0.0017 | 3.3583 |

Lens analysis: the same baseline captures and target hardware; CPU analysis reported 2026-10-23; 521 rounds with both lens and reference, four requests, five draft positions; concurrency not stratified.

| lens result | value | scope differing from the shared conditions |
|---|---:|---|
| greedy-chain empirical floor, accepted/round | 2.0499 | — |
| calibrated smoothed estimate, accepted/round | 2.1479 | confidence assumption, superseded by exact replay |
| floor minus captured drafter proxy, accepted/round | 0.2303 | paired comparison |
| paired difference, sample-level 95% interval | [0.0461, 0.4165] | bootstrap |
| paired difference, request-clustered 95% interval | [-0.1744, 0.4733] | bootstrap |
| lens top-token hit rate | 0.7197 | 1,370 committed-position pairs |
| top-token hit rate, request-clustered 95% interval | [0.6835, 0.7523] | 1,370 pairs |

The earlier lens subset and exact replay have different round counts. Their captured drafter proxy scores must not be substituted for one another. These measurements support investigating later-position workload adaptation and cost-aware draft depth. They do not show that the drafter can recover the full gap: it cannot see future target states.

## Upstream claims

Not measured here. DeepSeek supplies the target and native drafter architecture; vLLM supplies speculative verification. No external performance claim is used in the tables above. Upstream sources: [DeepSeek model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), [vLLM](https://github.com/vllm-project/vllm).

## Deploy

This is offline analysis; it installs no serving overlay. From the repository root, first check the CPU entry points:

```bash
PYTHONPATH="$PWD/code/dspark" python3 -B -m d1.exact_ceiling build --help
PYTHONPATH="$PWD/code/dspark" python3 -B -m d1.exact_ceiling score --help
PYTHONPATH="$PWD/code/dspark" python3 -B -m d1.ceiling --help
```

Configure your own private paths without editing the shipped scripts:

```python
# Run from the repository root with your CPU Python environment.
import sys
sys.path.insert(0, "code/dspark")
from d1 import exact_ceiling as method
method.CAPS = "/data/private/captures"
method.CKPT = "/data/private/native-checkpoint"
method.REF_DUMPS = "/data/private/reference-dumps"
sys.argv = ["exact_ceiling", "build", "--v1", "code/dspark",
            "--dir", "/data/private/replay", "--model", "<served-model-id>"]
method.main()
```

Send each generated `reqN.request.json` to your target's `/v1/completions` endpoint and save its response as `reqN.response.json` in that same private directory. Then run the wrapper again with `score` replacing `build`, remove `--model`, and add `--out /data/private/aggregates.json`. Supply matching reference dumps for KL. For lens analysis, configure the same three constants in `d1.ceiling` and call its `main()` with `--v1 code/dspark --out /data/private/lens-aggregates.json`.

Inspect request completeness, capture plumbing errors, returned logprob alignment, reference coverage, and target identity before interpreting results. Keep generated requests and responses private.

## Rollback

Stop the offline analysis and retain or remove your own generated files according to your data policy. No engine configuration changes are required. A future trained drafter would need separate fidelity, identity, acceptance, latency, and rollback qualification.

## Limits and known issues

Only a small request sample is represented. Context and workload are confounded with request identity. Finite-sample uncertainty remains even with exact target probabilities. The exact aggregate contains no bootstrap interval or KL coverage count. Tail-bucket KL can understate full-distribution KL. Unreturned draft probabilities make deployed acceptance a lower bound.

The scripts retain fixed default data paths; use the explicit wrapper above. No captures, reference generator, unit-test suite, weights, or trained adapter are included. Self-distillation data/training work is in progress; it has no qualified acceptance or throughput result.

## Repository layout

| path | purpose |
|---|---|
| [code/dspark/d1/exact_ceiling.py](code/dspark/d1/exact_ceiling.py) | replay request builder and target-logprob scorer |
| [code/dspark/d1/ceiling.py](code/dspark/d1/ceiling.py) | logit lens, calibration, paired/request-clustered bootstrap |
| [code/dspark/verify/captures.py](code/dspark/verify/captures.py) | asserted capture parsing and replay windows |
| [code/dspark/d1/results/exact_ceiling.json](code/dspark/d1/results/exact_ceiling.json) | exact-replay aggregates only |
| [code/dspark/d1/results/ceiling.json](code/dspark/d1/results/ceiling.json) | lens aggregates only |
| [NOTICE](NOTICE) | component attribution and licence boundary |
| [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt) | complete licence for the shipped PLI Labs code |

## Tests

No standalone CPU unit suite is bundled for this method. The help commands above exercise imports and CLI parsing on CPU; validate the aggregate files with `python3 -m json.tool <file>`. With your own captures, require an empty `CaptureSet.plumbing`, contiguous replay sequences, and aligned logprob response lengths. These checks cannot reproduce the private measurements from aggregates alone. Previously reported runtime test counts apply to the serving components, not this method.

## Credits and licences

PLI Labs method code: Apache-2.0. DeepSeek supplies the target and native drafter; HeNryous supplies the Spark serving image; sudoingX supplies the two-node launcher; turboderp supplies [exllamav3](https://github.com/turboderp-org/exllamav3); [vLLM](https://github.com/vllm-project/vllm) supplies the engine. Their software and weights keep their own licences. Nothing here grants rights to private captures or bundles model weights. See [NOTICE](NOTICE).

## Related

- `dsv41-async-hostflag-handoff`: [GitHub](https://github.com/ProprietaryLegal/dsv41-async-hostflag-handoff) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-async-hostflag-handoff).
- `dsv41-remote-dspark-drafter-3090`: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-dspark-drafter-3090) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-dspark-drafter-3090).
- `dsv41-dspark-drafter-tp2-nvlink`: [GitHub](https://github.com/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-dspark-drafter-tp2-nvlink).
- `dsv41-remote-engram-rdma`: [GitHub](https://github.com/ProprietaryLegal/dsv41-remote-engram-rdma) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-remote-engram-rdma).
- `ofl-rdma-transport`: [GitHub](https://github.com/ProprietaryLegal/ofl-rdma-transport) · [Hugging Face](https://huggingface.co/ProprietaryLegal/ofl-rdma-transport).
- `dsv41-native-1017-drafter-sm86`: [GitHub](https://github.com/ProprietaryLegal/dsv41-native-1017-drafter-sm86) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-native-1017-drafter-sm86).
- `dsv41-longctx-bench-tools`: [GitHub](https://github.com/ProprietaryLegal/dsv41-longctx-bench-tools) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-longctx-bench-tools).
- `dsv41-exact-acceptance-ceiling`: [GitHub](https://github.com/ProprietaryLegal/dsv41-exact-acceptance-ceiling) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-exact-acceptance-ceiling).
- `DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload`: [GitHub](https://github.com/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload) · [Hugging Face](https://huggingface.co/ProprietaryLegal/DeepSeek-V4.1-Flash-2xDGX-Spark-x86-offload).
- `dsv41-spark-offload-findings`: [GitHub](https://github.com/ProprietaryLegal/dsv41-spark-offload-findings) · [Hugging Face](https://huggingface.co/ProprietaryLegal/dsv41-spark-offload-findings).

Upstream forks: [Spark image and offload integration](https://github.com/ProprietaryLegal/deepseek-v41-1017-dgx-spark) · [two-node launcher](https://github.com/ProprietaryLegal/dgx-spark-deepseek-v41). [HF collection](https://huggingface.co/collections/ProprietaryLegal/deepseek-v41-flash-on-2x-dgx-spark-x86-offload-6adab604a517b426ded3916e) · [hub snapshot](https://github.com/ProprietaryLegal/pli-labs/tree/main/research/dgx-spark-offload).

<!-- pli:footer:start -->

---

## About PLI Labs

“Your proprietary work product is your proprietary intelligence.” PLI Labs is the open research arm of Proprietary Legal Intelligence. We publish legal model prunes, serving recipes for owned hardware, and open-source tools so readers can inspect our methods and test our claims before hiring us.

Founded by a practicing trial lawyer, PLI builds AI systems around a legal organization's own work product: retrieval, staged workflows, validation, and attorney review. The approach grew from systems running on live matters in a litigation practice. Our consulting work supports deployment through approved frontier APIs, private cloud, or open-weight models on hardware you own. Explore the research or contact us about your organization's work.

[Website](https://proprietarylegal.ai) · [PLI Labs](https://proprietarylegal.ai/labs.html) · [Research](https://proprietarylegal.ai/research/index.html) · [GitHub](https://github.com/ProprietaryLegal) · [Hugging Face](https://huggingface.co/ProprietaryLegal) · [Contact](https://proprietarylegal.ai/contact.html)

Proprietary Legal Intelligence LLC is a technology consultancy, not a law firm, and does not provide legal services or legal advice.

<!-- pli:footer:end -->
