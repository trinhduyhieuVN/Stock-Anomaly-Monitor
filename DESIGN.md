---
name: Stock Anomaly Monitor — native review extension
description: Observed Streamlit and Plotly review surface; not a new brand system.
colors:
  price-volume: "#5b8ff9"
  candle-up: "#20a486"
  candle-down: "#e45756"
  point-review: "#d62728"
  collective-review: "#8054ad"
---

# Design System: Stock Anomaly Monitor

## Overview

This records the implemented U08 extension of the existing local Streamlit app,
not a newly commissioned identity. Product and information boundaries are in
PRODUCT.md and docs/upgrade/ui_v2.md. No creative metaphor, font scale or native
theme token is invented. V1 is unchanged.

## Colors

The frontmatter values are literal chart colors in app_upgrade.py.
OHLC candles use green/red for rising/falling closes, with explicit text and
OHLC hover values; volume remains blue. Points use red diamonds; collective estimates use
purple regions at opacity0.10 with text labels. Color is not the sole kind cue.
The page ground and native control colors inherit Streamlit's current theme;
the isolated browser QA used its light theme, not a global application setting.

## Typography

Use native st.title/subheader/caption and controls. No project-authored font,
size or weight scale exists. Do not turn dependency defaults into fixed tokens.

## Layout

Wide Streamlit page, native sidebar, timeline before event table/detail tabs.
The chart height is480px; its source margins are left15/right15/top10/bottom20.
Clock ticks use %H:%M; a single bar has a ±5minute extent. Price traces separate
real gap/lunch segments rather than bridging them. Mobile uses the native
collapsed sidebar. The corrected local UI defaults to the full synthetic session;
Replay remains prefix-only with first-bar guidance and a full-session button.
Historical U08/U10 captures describe the earlier line chart, not this correction.

## Components

Reuse native buttons, selects, slider, table, tabs, form and CSV download.
Replay step/reset changes the as-of query, never deletes history. Manual review
is available only in explicit current mode and is timestamped at actual now.
Estimated suffix bounds, reference sample count, pending/censored outcomes,
proxy context and synthetic/not-live source remain visible in their detail views.
Missing table values display an em dash without changing numerical data/CSV.
Point and suffix candidate schemas are shown separately. Pending/censored,
insufficient history, missing news and absent reviews have explicit explanations.
No decorative raster artwork, custom CSS material or authored animation ships.

## Do's and Don'ts

- Do keep kind labels and marker geometry alongside color.
- Do preserve truthful as-of, source and unavailable-state explanations.
- Do retain incumbent native controls and theme unless a redesign is requested.
- Don't imply empirical ranks are probabilities or replay is live.
- Don't infer a WCAG certification from these limited screenshots or tests.

Not canonized: native caption contrast and below-fold/all-state responsiveness
have not been visually audited in every theme. They are not certified or silently
repaired; this extension does not authorize a global visual redesign.
