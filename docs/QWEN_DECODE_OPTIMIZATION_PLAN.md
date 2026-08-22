# Qwen decode optimization plan

Status: Phase 3 synchronization candidate rejected; DeltaNet recurrent-kernel
candidate accepted

Planning review: GPT-5.6 Luna, 2026-08-21

This plan targets the Qwen3.6 35B-A3B decode path. It separates profiling from
optimization so that a plausible storage or GPU explanation does not become a
production change without end-to-end evidence.

The plan does not set Gemma throughput as the initial target. Qwen and Gemma
have different architectures, kernels, and runtime maturity. The first target
is a repeatable same-host Qwen improvement that preserves output, state, memory,
and cancellation behavior.

## Current evidence

The following measurements came from the same host and the same short raw
completion shape, with 64 generated tokens:

| Model or setting | Decode rate |
| --- | ---: |
| Gemma 4 26B-A4B | 20.568 tok/s |
| Qwen, original comparison | 3.648 tok/s |
| Qwen, 16-slot LFU | 3.683 tok/s |
| Qwen, 24-slot LFU | 3.700 tok/s |
| Qwen, 32-slot LFU | 3.647 tok/s |

The slot experiment does not show a meaningful decode improvement from 24 or
32 slots. Keep 16-slot LFU as the control unless a later, valid experiment
reverses that result.

A separate 4,002-token Qwen request completed in 893.452 seconds with 106 output
tokens. Live stack samples observed work in `PreadExpertStreamer` and `pread`,
but this does not prove that physical SSD traffic dominates decode. Filesystem
cache behavior, task scheduling, command-buffer waits, route readback, and GPU
work remain confounded until the runtime exports aggregate diagnostics.

## Current decode dependency graph

[`QwenForwardRunner`](../Sources/TurboFieldfare/Runtime/Inference/QwenForwardRunner.swift)
currently submits and synchronously completes:

1. One embedding command buffer.
2. Three command buffers for each of 40 layers:
   mixer, shared-expert/router, and routed-expert/combine.
3. One final-head command buffer.

That is:

```text
1 + (3 * 40) + 1 = 122
```

The count covers the Qwen forward runner, not the complete token loop.
[`RawCompletion`](../Sources/TurboFieldfare/Runtime/Generation/RawCompletion.swift)
also performs sampling work. For a 64-token completion, the first generated
token is normally seeded by prompt logits, so the loop performs 63 Qwen
`produce` calls and 64 sampling steps.

The 122-wait structure is a measured code fact. The following are hypotheses
until aggregate diagnostics quantify them:

- Command-buffer completion and CPU/GPU synchronization are a major cost.
- Expert fetch is a major cost.
- Route readback and cache-plan construction are a material cost.
- Exact-demand expert I/O can overlap enough shared-expert GPU work to improve
  the full token step.

## Goals and acceptance gates

Profiling must account for at least 90% of the defined token-step wall time.
An optimization is accepted only when all of these conditions hold:

- Same-host median decode improves by at least 15%.
- Baseline and candidate use the same model receipt, build mode, prompt,
  generation settings, cache settings, and run ordering policy.
- Greedy token IDs and output hashes match the baseline.
- DeltaNet recurrent state and full-attention KV state remain aligned.
- Continuation, prompt replay, and cancellation remain correct.
- Cache hit/miss accounting is internally consistent.
- No swap, sustained memory pressure, invalid resource sample, or artifact
  attribution failure occurs.
- Package tests pass through `Scripts/test.sh`.

The recent 3.683 tok/s row implies a provisional 15% threshold of about
4.24 tok/s. The formal threshold must be calculated from a fresh interleaved
baseline rather than treating one row as a permanent control.

## Phase 0: isolate the prefill repair

The working tree currently contains Qwen multi-chunk prefill changes in:

- [`QwenForwardRunner.swift`](../Sources/TurboFieldfare/Runtime/Inference/QwenForwardRunner.swift)
- [`PrefillRuntimeConfig.swift`](../Sources/TurboFieldfare/Runtime/Prefill/PrefillRuntimeConfig.swift)

Review and land that correctness repair independently. Decode profiling and
optimization must preserve it without folding unrelated decode changes into the
same review.

Rollback point: restore only the previous decode implementation. Do not remove
or rewrite the multi-chunk prefill repair as part of a decode rollback.

## Phase 1: aggregate and export diagnostics

### Runtime aggregate

Extend
[`QwenDecodeDiagnostics.swift`](../Sources/TurboFieldfare/Runtime/Inference/QwenDecodeDiagnostics.swift)
with a versioned aggregate value containing:

- Decode-step count.
- Summed forward wall time.
- Embedding, layer, final-head, and expert-fetch time.
- Command-buffer submission count.
- Router evaluation count.
- Routed expert count.
- Cache hit and miss counts.
- Logical estimated expert bytes.
- Per-layer elapsed and expert-fetch totals.
- Attributed and residual wall time.

Logical estimated bytes must remain explicitly labeled as an estimate. It is
`missCount * expertStride`, not measured physical SSD traffic.

Add enough phase timing to distinguish:

- Mixer and router completion.
- CPU route readback and cache-plan construction.
- Shared-expert completion.
- Expert-fetch wait.
- Routed-expert/combine completion.
- Sampling and outer-loop residual time.

Do not serialize JSON, write files, or perform extra Metal operations inside the
timed token path. Capture one completed token diagnostic after `produce`
returns, then aggregate in memory.

### Raw completion integration

Add an optional Qwen aggregate to `RawDecodeResult`. Preserve existing
initializers with a default `nil` value.

Aggregate only post-prefill Qwen `produce` calls. Exclude:

- Chunked prompt prefill.
- Scalar prompt replay.
- The first generated token when it is seeded directly from prompt logits.

Tests belong in:

- [`QwenDecodeDiagnosticsTests.swift`](../Tests/TurboFieldfare/Core/Runtime/Inference/QwenDecodeDiagnosticsTests.swift)
- [`RawCompletionLoopTests.swift`](../Tests/TurboFieldfare/Core/Runtime/Generation/RawCompletionLoopTests.swift)

Cover zero-step behavior, layer merging, overflow handling, attribution math,
63-versus-64 forward-step accounting, and exclusion of prompt work.

### CLI export

Add an opt-in `--diagnostics-json <path>` option in:

- [`Args.swift`](../Sources/TurboFieldfareCLI/Args.swift)
- [`Run.swift`](../Sources/TurboFieldfareCLI/Run.swift)
- [`CLIArgumentsTests.swift`](../Tests/TurboFieldfare/Core/CLI/CLIArgumentsTests.swift)

Write one JSON document after generation timing completes. Keep generated text
on stdout and preserve the existing timing footer on stderr.

### Server and harness export

After the CLI schema is stable, add opt-in server export. A normal OpenAI
response must not change unless diagnostics are explicitly enabled.

- Non-streaming: add one top-level `turbo_fieldfare_diagnostics` object.
- Streaming: add the object only to the final usage chunk.
- Serialize once after generation, never once per token.
- Preserve normal OpenAI `usage` fields.

Likely server files:

- [`ServerInference.swift`](../Sources/TurboFieldfareServer/Core/ServerInference.swift)
- [`HTTPServer.swift`](../Sources/TurboFieldfareServer/Core/HTTPServer.swift)
- [`ServerArguments.swift`](../Sources/TurboFieldfareServer/Core/ServerArguments.swift)
- [`HTTPServerTests.swift`](../Tests/TurboFieldfareServer/HTTPServerTests.swift)

The external benchmark harness should retain the versioned diagnostics object
in its machine-readable report and require it for profiling runs only.

## Phase 2: establish the formal baseline

Run baseline and later candidate rows with:

- Same host, artifact receipt, release build, and runtime settings.
- Temperature 0 and exact token-ID capture.
- 16-slot LFU and RDADVISE off.
- Three warmups and at least five valid measured runs per configuration.
- Interleaved baseline/candidate ordering.
- 64-, 256-, and 512-token decode workloads.
- Fresh-server comparisons when evaluating cache policy or capacity.

Record:

- Prompt and generated token counts.
- Time to first token, decode time, total time, and decode tok/s.
- Diagnostic attribution and residual time.
- Command-buffer count.
- Expert-fetch wait, hits, misses, and logical estimated bytes.
- CPU route-planning time.
- Peak RSS, available memory, pressure state, and run validity.
- Token IDs or an output hash suitable for exact parity checks.

Do not proceed to optimization until at least 90% of token-step wall time is
accounted for. If the residual is too large, improve diagnostics rather than
selecting a candidate by intuition.

The formal baseline passed on 2026-08-21 for Qwen3.6 35B-A3B using the settings
above. Three fresh-process parity probes and all five measured runs per target
produced identical greedy token IDs. Mean measured rates were 3.72 decode
tokens/s for 64-token completions, 3.75 decode tokens/s for 256-token
completions, and 3.73 decode tokens/s for 512-token completions. The release
build and package validation also passed.

## Phase 3: reduce command-buffer synchronization

The first candidate should reduce synchronization without changing model math.
Refactor the decode-only `encodeLayer` and `encodeMoE` boundary so one command
buffer contains:

1. Input RMS normalization.
2. Attention or DeltaNet mixer.
3. Residual update.
4. Post-attention RMS normalization.
5. Shared expert.
6. Router.

Wait once for router output, read the exact route IDs, construct the existing
cache plan, fetch required experts, then submit routed expert/combine work.

The expected forward-runner command count becomes:

```text
1 + (2 * 40) + 1 = 82
```

### Phase 3 measurement result

The candidate was measured against a clean `HEAD` baseline using the same
Qwen3.6 35B-A3B artifact, host, prompt, temperature `0`, 16-slot LFU cache,
prefill settings, `max_context=16384`, and RDADVISE off. The baseline release
binary SHA-256 was
`e2e68e456ffdc6e8d1a5f2185e242b36b72299e1d8038fdddb19b6ce19590e03`; the
candidate release binary SHA-256 was
`928bb31b737e35234bab8223085163a8c09f21f5745389379cf321f51a6b3abf`.

Five measured runs per target produced these medians:

| Completion target | Baseline | Candidate | Change | Forward-runner submissions per decode step |
| ---: | ---: | ---: | ---: | ---: |
| 64 tokens | 3.76 tok/s | 3.84 tok/s | +2.13% | 122 -> 82 |
| 256 tokens | 3.78 tok/s | 3.87 tok/s | +2.38% | 122 -> 82 |
| 512 tokens | 3.77 tok/s | 3.86 tok/s | +2.39% | 122 -> 82 |

All three exact output hashes matched. The candidate workload validation and
package tests passed, but the candidate is rejected because every measured
gain is below the required 15% median improvement. The baseline and candidate
were run as separate fresh-process series rather than interleaved rows, which
is a protocol deviation and prevents this result from being treated as a
final accepted performance comparison. The Phase 2 report format also did not
contain resource samples with backend-memory attribution, so the resource
gate remains unverified. The command-buffer reduction is therefore retained
as an exploratory result, not an accepted Phase 3 optimization.

Safety requirements:

- Keep all dependent encoders on the same Metal command queue.
- Read route IDs only after the combined command buffer completes.
- Keep expert buffers and argument buffers alive through routed command
  completion.
- Preserve cancellation cleanup before reset or runner reuse.
- Do not overlap separate Qwen layers; recurrent and KV state remain strictly
  ordered.
- Verify the diagnostic command count instead of assuming coalescing occurred.

Reject the candidate if the same-host median gain is below 15%, even if the
command-buffer count improves.

Rollback point: restore the current three-stage decode layer path while keeping
diagnostics and the prefill repair.

## DeltaNet recurrent-kernel follow-up

Profiling after the Phase 3 rejection attributed about 83% of Qwen forward time
to mixer work and only 6-7% to expert-fetch wait. The decode recurrent kernel
was launching one thread per value head and serializing every independent value
column inside that thread. The follow-up candidate changed
[`QwenGatedDeltaNet.swift`](../Sources/TurboFieldfare/Kernels/LinearAttention/QwenGatedDeltaNet.swift)
and
[`linear_attention.metal`](../Sources/TurboFieldfare/Metal/LinearAttention/linear_attention.metal)
to dispatch one thread per `(value head, value column)`. The production grid is
128 columns by 32 value heads. Each thread retains the original ordered
key-dimension accumulation for its column, so columns gain parallelism without
changing their arithmetic order or sharing writable state.

The rejected Phase 3 synchronization change was removed before this candidate
was built. The candidate therefore retains the original 122 forward-runner
submissions per decode step and isolates the recurrent-kernel change. A strict
alternating baseline/candidate comparison used three warmup cycles and five
measured cycles per arm and completion target. The clean baseline server
SHA-256 was
`e2e68e456ffdc6e8d1a5f2185e242b36b72299e1d8038fdddb19b6ce19590e03`, and
the candidate server SHA-256 was
`4f42c2594420b3f12229e65e4ce52cd201216bcf6e709739836b6bae65f21c00`.

| Completion target | Baseline median | Candidate median | Change |
| ---: | ---: | ---: | ---: |
| 64 tokens | 3.72 tok/s | 16.60 tok/s | +346.24% |
| 256 tokens | 3.73 tok/s | 16.48 tok/s | +341.82% |
| 512 tokens | 3.72 tok/s | 16.24 tok/s | +336.56% |

Exact greedy token IDs matched on every baseline/candidate row. The output
SHA-256 values were
`8e65e1b5adf49bd49523cffc8ee00c4d896927019d76abbc541b2ca861f61ddf`,
`62578360fa5015aaf9505788a93e824e1b6463a3566fce8e2e7ec5fad9c341ff`, and
`50286bf4360a7ed31e4011fd3e0ff360b2fad7c1c8c55bac700882e4b91b90d1` for
the 64-, 256-, and 512-token targets. The standalone candidate protocol also
passed with medians of 16.13, 16.48, and 16.34 tok/s, no validation errors, and
a minimum diagnostic attribution ratio of 0.995057.

All four focused DeltaNet tests passed, followed by 765 package tests across
141 suites and a release build. The 16 interleaved arm runs had no invalid
resource samples, a maximum peak RSS of 1.577 GB, and at least 5.245 GB
available memory.

The final G4 service gate passed on 2026-08-22. It covered a 4,002-token prompt,
a 15,362-token near-context-limit prompt, 50 successful sequential requests,
client-disconnect cleanup, ten successful server restarts, and orphan-listener
cleanup. Resource sampling was valid; peak process-group RSS was 1.483 GB
against the 3.0 GB limit, with at least 5.175 GB available memory. The temporary
Python harness required `jinja2==3.1.6` for chat-template rendering. Its
tokenizer-only Transformers installation did not include PyTorch, which is
expected because inference remained in the Swift/Metal server. No repository
dependency changed.

This candidate clears the 15% performance gate with exact parity and passes the
correctness, lifecycle, and resource gates, so the recurrent-kernel change is
accepted. Phase 4 remains deferred because measured expert-fetch wait does not
provide enough headroom to justify an I/O-overlap candidate.

## Phase 4: overlap exact-demand expert I/O

Attempt this only when Phase 1 shows that expert-fetch wait has enough
theoretical headroom to clear the 15% gate. Do not infer this from stack samples
or estimated bytes alone.

Evaluate this as a separate candidate before stacking it with Phase 3:

1. Submit mixer/router work.
2. Queue shared-expert work.
3. Wait for mixer/router completion.
4. Read exact route IDs and reserve the exact cache plan.
5. Start exact expert fetch while shared-expert GPU work executes.
6. Await shared-expert and expert-fetch completion.
7. Submit routed expert/combine work.

Useful patterns from
[`RealForwardRunner.swift`](../Sources/TurboFieldfare/Runtime/Inference/RealForwardRunner.swift)
include:

- Explicit pending-command records.
- Bounded pending depth.
- Expert and argument-buffer lifetime management.
- Cache-hit/miss phase splitting.
- Slot reservation until the consuming command completes.

Do not copy its prefill tile scheduler or assume Gemma's recurrent/KV
dependencies match Qwen. Do not pipeline multiple Qwen layers without
independent scratch buffers and explicit cache-slot lifetime protection.

Cancellation must not abandon a task that is still writing an expert-cache
slot. No later fetch may evict a reserved slot while a Metal command buffer
still references it.

Rollback point: disable overlap and return to exact sequential fetch after
router completion.

## Validation matrix

### Focused tests

- Diagnostic aggregation and schema stability.
- Decode-only accounting and first-token seeding.
- Command-buffer count before and after coalescing.
- Exact greedy token parity.
- Prompt continuation and prompt-state replay.
- Cancellation during GPU work and expert fetch.
- Cache-plan hit/miss consistency.
- Buffer lifetime under delayed command completion.

### Package validation

Run:

```bash
Scripts/test.sh
swift build -c release
```

### Live validation

Follow the repository model-process and memory-pressure checks before every
model run. Run only one model workload at a time. Report the commit, artifact
receipt, hardware, RAM, macOS, Swift version, exact command, exit code, timing
footer, resource validity, and every protocol deviation.

For each candidate, compare an interleaved baseline and candidate series. Do
not average invalid runs or compare different artifacts, hosts, prompts, output
lengths, or cache settings.

## Risks

| Risk | Required control |
| --- | --- |
| Metal uses a released expert or argument buffer | Explicit pending-command lifetime record |
| Cache slot is evicted while the GPU reads it | Reservation through command completion |
| Cancellation leaves a cache write active | Structured fetch task cleanup before reset |
| DeltaNet state advances out of order | One strictly ordered layer/token dependency chain |
| Full-attention KV position diverges | Position and continuation parity tests |
| Diagnostics change measured timing | Aggregate in memory; serialize after timing |
| Estimated bytes are reported as disk traffic | Label them logical estimated bytes |
| Lower command count does not improve wall time | Enforce the 15% end-to-end gate |

Any buffer-lifetime, cache-eviction, cancellation, recurrent-state, KV-state,
or token-parity failure is an immediate rollback, not a performance tradeoff.

## Deferred work

Do not implement these as part of the first decode optimization:

- Larger expert-cache defaults.
- Speculative expert prediction or prefetch.
- Expert-file layout reordering.
- RDADVISE default changes.
- Speculative decoding.
- Multiple Qwen layers in flight.
- Recurrent/KV state redesign.
- Gateway routing or default-model promotion.
- Gemma-level throughput promises.

Previous experiments found that several of these ideas improved local metrics
but failed broader runtime gates. Reopen one only with a new measured mechanism
and an interleaved same-host control.

## Execution checklist

- [x] Preserve and review the existing multi-chunk prefill diff.
- [x] Add versioned aggregate Qwen diagnostics and unit tests.
- [x] Attach decode-only aggregate diagnostics to `RawDecodeResult`.
- [x] Add opt-in CLI JSON output.
- [x] Add opt-in server and harness export after the CLI schema stabilizes.
- [x] Establish a valid interleaved baseline.
- [x] Verify the current 122 forward-runner command buffers per Qwen step.
- [x] Account for sampling and residual token-loop time.
- [x] Require at least 90% wall-time attribution.
- [x] Implement the decode command-buffer coalescing candidate.
- [x] Run parity, continuation, cancellation, package, and resource checks.
- [x] Enforce the 15% same-host median improvement gate; reject the candidate
  below threshold.
- [x] Profile the rejected candidate again.
- [x] Defer exact-demand I/O overlap because measured fetch wait is too small.
- [x] Accept the DeltaNet recurrent-kernel candidate after all correctness and
  resource gates pass.