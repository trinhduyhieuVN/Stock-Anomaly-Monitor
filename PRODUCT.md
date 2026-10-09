# Stock Anomaly Monitor

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The project owner demonstrating a research project and reviewers inspecting
Vietnamese stock-bar anomalies. Audience/job follows the approved reduced-scope
plan; no claim of institutional deployment or certified financial expertise.

## Product Purpose

Help inspect unusual five-minute bars and mild multi-bar shifts, replay what was
known at T, add mature forecast residual evidence, and record manual review.

## Operating Context

Windows local Streamlit/Plotly research application. Existing v1 app.py stays
unchanged. The separate v2 entrypoint runs a clearly labeled synthetic fixture
without real trained models or a provider connection. Vietnamese working copy.

## Capabilities and Constraints

Bar5m, FPT/HPG/VCB, causal context and empirical ranks, suffix3/6/12 heuristic,
zero-return h1/3 forecast, immutable SQLite event versions and annotations.
Scores/ranks are not probabilities, manipulation findings or trading advice.
Tick/order book, expert labels, unseen real holdout and production streaming are
unavailable. Real source rights remain pending; no public real-data export.
No framework change, notifications, paid service, provider call or deployment.

## Evidence on Hand

docs/UPGRADE_PLAN_DE_CUONG.md, docs/CODEX_PROMPTS_NANG_CAP.md,
docs/upgrade/progress.md and criteria_met=true reports U00–U07. Synthetic truth
does not prove market effectiveness; shared evaluation seeds are already exposed.

## Product Principles

- Preserve the information boundary and initial decision.
- Show reasons, coverage, missing evidence and provenance.
- Treat review as prioritization, not an accusation or trade instruction.
- Keep a distributable synthetic demo independent of local-only real artifacts.

## Open Decisions

No new brand identity requested. V2 extends the native Streamlit/Plotly
interaction language of v1; branding/preferences beyond that remain undecided.
