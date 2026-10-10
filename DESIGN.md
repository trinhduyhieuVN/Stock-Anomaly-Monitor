---
name: Stock Anomaly Monitor — native review surfaces
description: Existing v2 surface plus the user-approved research-only dark extension.
colors:
  price-volume: "#5b8ff9"
  candle-up: "#20a486"
  candle-down: "#e45756"
  point-review: "#d62728"
  collective-review: "#8054ad"
  research-primary: "#2D6AC4"
  research-ground: "#0E141B"
  research-panel: "#18222E"
  research-text: "#E7EDF5"
  research-sidebar: "#141C26"
  research-sidebar-panel: "#1D2A39"
  research-link: "#8DB9F5"
  research-candle-up: "#42C6A0"
  research-candle-down: "#F08080"
  research-volume: "#79ACF7"
  research-point: "#F3BC68"
  research-collective: "#B89CEB"
  research-grid: "#334155"
  research-ticks: "#B7C5D8"
typography:
  research-ui:
    fontFamily: "sans-serif"
  research-chart:
    fontSize: "13px"
components:
  research-button-primary:
    backgroundColor: "{colors.research-primary}"
    textColor: "#FFFFFF"
  research-button-secondary:
    textColor: "{colors.research-text}"
  research-input:
    backgroundColor: "{colors.research-panel}"
    textColor: "{colors.research-text}"
  research-navigation:
    textColor: "{colors.research-text}"
  research-filter-chip:
    backgroundColor: "{colors.research-primary}"
    textColor: "#FFFFFF"
  research-review-form:
    textColor: "{colors.research-text}"
  research-expander:
    textColor: "{colors.research-text}"
---

# Design System: Stock Anomaly Monitor

## Overview

This records the implemented U08 extension of the existing local Streamlit app,
not a newly commissioned identity. Product and information boundaries are in
PRODUCT.md and docs/upgrade/ui_v2.md. No creative metaphor, font scale or native
theme token is invented. V1 is unchanged.

The research surface is a separate, approved extension: a quiet charcoal-blue
workspace with light text, native controls and an uninterrupted reading order.
Source, time and unavailable-state explanations remain visible before analysis.
The dark extension changes presentation, not data, models or market claims.

The original unprefixed v2 tokens and guidance below remain in force for
app_upgrade.py. Research-prefixed tokens belong only to app_research.py and its
local launcher. This is a merge, not a replacement identity for v1/v2.

**Key Characteristics:**

- Native controls and tonal separation, not decorative dashboard cards.
- Source/time truth and semantic candles with labels alongside color.
- Research-only process theme; no shared or machine-wide theme setting.

## Colors

The original five frontmatter values are literal chart colors in app_upgrade.py.
OHLC candles use green/red for rising/falling closes, with explicit text and
OHLC hover values; volume remains blue. Points use red diamonds; collective estimates use
purple regions at opacity0.10 with text labels. Color is not the sole kind cue.
The page ground and native control colors inherit Streamlit's current theme;
the isolated browser QA used its light theme, not a global application setting.

### Primary

Research primary blue marks actionable progress and save actions. Research link
blue is lighter for text links; it is not the button-fill token.

### Secondary

Research mint/coral candles distinguish rising/falling closes. Volume is pale
blue; an amber diamond means a window exceeds the pinned threshold, not fraud.
Purple bands retain their separate collective-event meaning. These chart colors
are semantic cues, not an alternative navigation or action palette.

### Neutral

Research ground, panel and sidebar tokens create restrained tonal separation.
Light research text carries body copy; subdued research ticks and grid colors
support the chart without competing with candle bodies. Native notices derive
their own colors from Streamlit; they are not new project-owned accent tokens.

**The Scope Rule.** Apply research tokens only to the research process and its
newly constructed charts; never silently recolor the archived v1/v2 applications.

## Typography

Use native st.title/subheader/caption and controls. No project-authored font,
size or weight scale exists. Do not turn dependency defaults into fixed tokens.

Research explicitly selects the native sans-serif family and a 13px chart font.
Installed Streamlit 1.64.0 rendered Source Sans: body/buttons 16px, captions 14px in
the sampled states. These are version-specific observations, not a new display
font or normative native heading scale. Keep Vietnamese diacritics legible and
retain native title/subheader/label/caption hierarchy.

## Layout

Wide Streamlit page, native sidebar, timeline before event table/detail tabs.
The chart height is480px; its source margins are left15/right15/top10/bottom20.
Clock ticks use %H:%M; a single bar has a ±5minute extent. Price traces separate
real gap/lunch segments rather than bridging them. Mobile uses the native
collapsed sidebar. The corrected local UI defaults to the full synthetic session;
Replay remains prefix-only with first-bar guidance and a full-session button.
Historical U08/U10 captures describe the earlier line chart, not this correction.

The research sidebar groups source and viewing level, instrument/time, Replay
controls and advanced filters. The latter two are native expanders; collapsing
them never removes candles or changes filters. Full-session remains the default,
and Replay stays prefix-only. Native mobile navigation wraps and the sidebar
collapses; content remains in the same order, with longer explanations below
the fold. Wide numerical tables retain local horizontal scrolling rather than
forcing document-wide overflow. The approved dark QA covers 1440/1516px desktop
and 390px mobile, not every possible viewport.

## Elevation & Depth

Research uses flat tonal layers. No raised-card shadow, glass, texture or authored
ambient glow ships. Native keyboard focus is visible: the sampled button had a
3.2px blue-alpha focus ring. This is a focus state, not decorative depth.

## Shapes

Keep native controls and borders. Sampled native buttons/forms have softly
rounded8px corners; these remain dependency-owned, not a project-wide radius
scale. Point events are diamonds, candle wicks preserve OHLC extent, and gaps
remain separate traces. Do not smooth or bridge missing market intervals.

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

Research reuses the same native families. Bước tiếp is primary while Replay is
active; Xem toàn phiên is primary on the first-bar recovery state; Lưu nhận xét
is the primary form action. Exports, guide and reset remain secondary. Labels
remain visible above inputs; expanders carry advanced controls without changing
their state keys or causal boundaries. Review submission stays explicit and
timestamped at actual now. Synthetic, local historical and unopened trial states
never share misleading LIVE copy or stale chart/export artifacts.

The sidecar holds self-contained documentation previews of observed native
primitives. It is not CSS injected into the app, a replacement widget library,
or a pixel-perfect implementation of Streamlit's version-dependent runtime.

## Do's and Don'ts

- Do keep kind labels and marker geometry alongside color.
- Do preserve truthful as-of, source and unavailable-state explanations.
- Do retain incumbent native controls and theme unless a redesign is requested.
- Don't imply empirical ranks are probabilities or replay is live.
- Don't infer a WCAG certification from these limited screenshots or tests.
- Do use the research launcher to apply its process-only native dark theme.
- Do pair candle/marker colors with shape, labels and factual explanations.
- Don't interpret a UI implementation acceptance as human UX or market effectiveness.

Not canonized: native caption contrast and below-fold/all-state responsiveness
have not been visually audited in every theme. They are not certified or silently
repaired; this extension does not authorize a global visual redesign.

Research QA sampled body, caption-opacity, notice and action text contrast and
keyboard focus, but does not certify all states or full accessibility. Wide
tables may need local scrolling on mobile; mobile explanations take more than
one viewport. The pre-existing v2-only PRODUCT.md context is not rewritten by
this research-only request. These limits are disclosed, not canonized as defects
to reproduce or silently repaired beyond scope.
