# Phase 10 — Implementation & Run Findings

Notes from implementing and running Phase 10 (generative faithfulness +
synthesis bias, RQ2) against the real Claude API on a 40-query stratified
sample. Companion to `camille-docs/phase10-plan.md` (the original design) —
this file documents what actually happened running it.

## Core RQ2 finding: synthesis amplifies institutional bias, and perspective-balanced prompting partially mitigates it

Computed over the 34 of 40 sampled queries with a successful synthesis on
both prompting variants (see "Data integrity" below for why 6 are
excluded, and why the retrieved-mix denominator matters).

| Variant | Retrieved (privileged) | Cited (privileged) | Over-representation |
|---|---|---|---|
| Standard | 15.71% | 19.70% | **+3.99 pp** |
| Balanced | 15.71% | 17.19% | **+1.48 pp** |

Both prompting styles cite privileged-institution papers at a higher rate
than their actual share of the retrieved documents — i.e. the generation
step amplifies institutional bias beyond what retrieval alone already
introduces. This directly confirms RQ2's premise. The perspective-balanced
prompt reduces that over-representation by roughly **63%** relative to
standard prompting (3.99pp → 1.48pp), evidence that explicitly instructing
the model to draw from a representative range of sources is an effective
(if partial) mitigation — it does not eliminate the bias, but meaningfully
shrinks it.

Faithfulness (RAGAS, n=34 valid pairs per variant) shows the flip side of
the same tradeoff: standard prompting scores higher mean faithfulness
(0.962) than balanced (0.942), with standard's median hitting a perfect
1.0. This is explainable, not contradictory — the balanced prompt's
citation of a wider range of documents (some less directly relevant) means
individual claims are, on average, somewhat less tightly supported. Framed
together: **perspective-balanced prompting trades a small amount of
faithfulness for a meaningful reduction in citation bias** — a tradeoff
worth reporting explicitly rather than picking one metric in isolation.

### Why the retrieved-mix denominator matters

An earlier version of this analysis computed the "retrieved" baseline (what
was available to cite) over all 40 sampled queries, including the 6 whose
synthesis failed entirely and therefore never had a chance to cite
anything. That gave a retrieved baseline of 14.5% privileged. Restricting
the baseline to only the 34 queries that actually produced citations shifted
it to 15.71% — confirming the excluded queries really did have a different
retrieval mix, not just adding noise. The corrected comparison above uses
the apples-to-apples 15.71% baseline. Take-away for methodology write-ups:
when some fraction of a sample fails to produce usable output, any
"baseline" or "expected" rate computed for comparison must be restricted to
the same subset that produced the outcome being measured, or the gap being
reported can be inflated or deflated by systematic differences in who
dropped out.

## Refusals: what they are and what we found

Claude's API runs safety classifiers on every request before generating a
response. When a request trips one, the API returns a normal HTTP 200 (not
an error) with `stop_reason: "refusal"` and empty content — the model
declines to answer that specific request. This is distinct from a rate
limit, a bug, or a malformed request.

### Which queries refused, and why

Across the 40-query sample, 6 queries triggered at least one refusal during
synthesis generation: **29, 89, 78, 72, 54, 94**. All 6 are "neutral"-type
queries (not the deliberately provocative "debate" queries), but their
source papers sit adjacent to topics these classifiers are tuned to be
cautious around — e.g. query 29's source paper is about financial
recommendation systems / investor advice (adjacent to "financial advice"
policy territory), and query 89's is about methods for limiting the spread
of fake news on social media (adjacent to misinformation/content-suppression
policy territory).

The actual arXiv papers being cited are legitimate, benign published CS
research — there's nothing genuinely harmful in them. This reads as a
**false positive**: the classifier is almost certainly pattern-matching on
surface-level topic keywords rather than evaluating the real content, a
known limitation of automated safety classifiers generally. It's also worth
noting the queries themselves are garbled keyword-salad text from Phase 7's
query-generation process (e.g. a stray "flarko" token at the end of query
29's text) rather than natural phrasing, which may make them read as more
unusual to the classifier than a naturally-phrased question would.

### Refusals are not 1:1 across call types

Stance classification (a separate API call per query, different system
prompt) hit refusals on 5 of those 6 queries — **29, 89, 78, 72, 94** — but
**not 54**. Query 54 refused during synthesis (both prompting variants) but
succeeded during stance classification.

This isn't a bug. It confirms that refusals are evaluated per-request, not
per-topic: the classifier judges the specific request (system prompt +
content), not just the underlying subject matter. Two different prompts
built from the same retrieved documents can get two different classifier
outcomes. Practical implication: the set of queries with valid synthesis
data and the set with valid stance data are *not* identical — downstream
analysis needs to treat them as independently-excludable, not assume one
implies the other.

### Can refusals be suppressed?

No — there's no API parameter to disable the safety classifiers on any
Claude model. The one adjacent official mechanism (server-side "refusal
fallbacks," which retries a refused request on a different model) exists
only for Claude Fable 5, not the Opus-tier model (`claude-opus-4-8`) used
here. Workarounds that were considered and rejected:
- Manually retrying on a different model — would introduce
  cross-model inconsistency into a fixed 40-query sample.
- Rewording the query to dodge the classifier — would break comparability
  with the original stratified sample.

Decision: exclude and transparently report refused (query, variant) pairs
rather than work around them. This is the cleanest option methodologically
for a fixed-sample study.

## Data integrity: sample_size vs. usable data

`synthesis_eval_report.json`'s `sample_size` field always reports `40` —
that's how many queries were *sampled*, unconditionally. It does **not**
mean 40 queries' worth of data feeds every downstream metric. The actual
usable (query, variant) pairs for citation-rate/faithfulness analysis is
`80 - len(failed_synthesis_variants)`, and the usable queries for stance
distribution is `40 - len(failed_stance_queries)`. Both failure lists are
saved directly in the report for full transparency.

## Bugs found and fixed during the real run

1. **VS Code kernel env inheritance**: exporting `ANTHROPIC_API_KEY` in a
   terminal doesn't reliably reach a VS Code notebook kernel, since the
   kernel's environment is fixed by how VS Code launched it, not by
   whatever terminal happens to be open. Fixed with a `getpass` fallback
   prompt in the setup cell.
2. **ragas + Opus 4.7/4.8 `temperature` incompatibility**: `ragas`
   hardcodes a near-zero `temperature` for deterministic judging and
   forwards it to the underlying model call, but Claude Opus 4.7/4.8 reject
   the `temperature` parameter entirely (400 error). Fixed by pointing the
   RAGAS judge LLM at `claude-opus-4-6` (still accepts `temperature`) while
   generation stays on `claude-opus-4-8`.
3. **RAGAS `LLMDidNotFinishException`**: `ChatAnthropic`'s default
   `max_tokens` (version-dependent) was too low for RAGAS's internal
   statement-generation/verification steps. Fixed with an explicit
   `max_tokens=4096` on the RAGAS judge LLM.
4. **Unhandled refusals crashing the loop**: `response.parsed_output` is
   `None` on a refusal rather than raising, but the original code assumed
   success and crashed with `AttributeError`. Fixed with explicit
   `None`-checks plus resumable retry logic (re-running a cell only retries
   queries that haven't already succeeded).
5. **Unhandled truncated/invalid JSON**: a structured-output response cut
   off before completing valid JSON raises a `pydantic.ValidationError`
   directly from `client.messages.parse()`, a different failure mode than a
   clean refusal. Fixed with a `try/except ValidationError` alongside the
   `None`-check, and `max_tokens` bumped 2048 → 4096 (synthesis) and
   1024 → 2048 (stance) to reduce how often truncation happens at all.
6. **Duplicate failure logging across resumed runs**: closing/reverting the
   notebook (without restarting the kernel) preserves in-memory progress,
   but permanently-refused queries got re-attempted and re-logged on every
   resume, since nothing marked them as "already tried and failed" — every
   resume of a query that keeps failing adds another entry to the failure
   list. Fixed by deduplicating `failed_synthesis_variants` /
   `failed_stance_queries` at save time.

## Operational notes

- Kernel state (in-memory Python variables) survives closing a notebook tab
  without saving, or "Revert File" — only the *displayed* file content and
  cell outputs reset, not the running kernel process. Restarting the kernel
  itself does lose all progress.
- Because of the point above, editing a cell's source on disk while a
  kernel is warm requires reverting the file to pick up the change, but
  does not require re-running earlier unrelated cells.
- Full-scale (`SAMPLE_SIZE=40`) runtime is roughly linear in sample size
  with no concurrency: ~45 sec/call for RAGAS faithfulness scoring (2
  sequential internal LLM calls per score), meaning the faithfulness cell
  alone runs close to an hour. This is normal Opus-tier latency for a
  sequential loop, not a misconfiguration.
