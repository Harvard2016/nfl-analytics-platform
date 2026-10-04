# Gridiron Lens design system

Source of truth: `apps/web/app/globals.css` (tokens, motion) and `apps/web/app/layout.tsx` (fonts, shell).
The redesign changed presentation and interaction only. No model, threshold, label, export or metric was changed.

## Tokens

| Token (Tailwind name) | Dark surface | Ivory surface (`.paper`) | Use |
|---|---|---|---|
| `bg` | `#101713` charcoal | `#f2efe5` ivory | page background |
| `surface` | `#18221c` | slightly darker ivory | panels, selected rows |
| `line` | `#2b3a30` | warm grey | rules, borders, ghost numerals |
| `ink` | `#f2efe5` | `#101713` | text |
| `muted` | `#9ba89d` sage | `#4f5b52` | secondary text |
| `teal` | `#d8f16e` chartreuse | `#55700a` | primary accent, model prediction, links. The token name is historical. |
| `defense` | `#8ebaf1` powder blue | darker blue | defenders, released/editorial labels |
| `amber` | `#e8ad66` | darker amber | highlights module accent, evidence highlight, cautions |
| `coral` | `#f0806f` | darker coral | disagreement / error states |
| `turf` | `#1c3527` | | field |

Wrap a section in `.paper` to flip every token to the ivory palette; components need no colour changes.
Inline SVG uses `var(--color-*)` so figures follow the surface. Fixed hex appears only where a figure is always dark
(field, stadium illustration, timeline, command block).

## Type

- Display: Barlow Condensed 600–800, uppercase, via `.display`. Page titles 6xl–9xl, section titles 4xl–6xl.
- `.narrow`: Barlow Condensed in sentence case for labels, tabs, buttons, sub-heads.
- Body: Inter. Body copy is capped near 62–70 characters.
- `.mono` / `.kicker`: IBM Plex Mono, small uppercase with letter-spacing, for timestamps, ids, units and section kickers only.
- `.num`: tabular figures for every table and statistic.
- `.outline-text`: stroked display text, used once per page at most.

## Components

- `SiteNav`: sticky, compacts on scroll, active link from the path, mobile menu button with `aria-expanded`, Esc closes. GitHub links point to the public project repository.
- Skip link, footer with system and evidence links (`layout.tsx`).
- `.btn-primary` (chartreuse block) and `.btn-quiet` (underlined) are the only two button styles for navigation actions.
- `Badge` (`ui.tsx`): Observed / Released label / Predicted / Derived. The word is always shown; colour is never the only signal.
- `Field`: turf bands, yard numbers, chartreuse line of scrimmage, ivory offense circles, powder-blue defender crosses, amber evidence highlight.
- Segmented controls use `aria-pressed`; panels use `role="tablist"` with arrow-key movement; the expanded field is a `role="dialog"` that returns focus.
- `Reveal`, `Parallax`, `StadiumScene` (original SVG illustration, not footage), `Architecture` (HTML figure).

## Page patterns

| Page | Surface | Title | Accent |
|---|---|---|---|
| Home | dark hero, ivory chapters 01–03, dark closing band | Read the field. | chartreuse |
| Coverage film room | dark, three columns: library / field + playback / evidence | Read the defense. | chartreuse + blue |
| Predictions | dark matchup header, ivory breakdown | Before kickoff. | chartreuse + blue |
| Highlights | dark timeline editor, ivory evidence and benchmark | Find the moment. | amber |
| Research | ivory case studies, dark experiment notebook | The playbook. | chartreuse |
| Engineering | dark architecture figure, ivory process and commands | Built to be inspected. | chartreuse |

Every result is shown with its limitation next to it. Standing wording: "released coverage label", "previously examined
benchmark", "No established gain over Elo", highlight scores are ranks and "not a probability".

## Motion

CSS and `IntersectionObserver` only; no animation library was added.

- Page transition: `app/template.tsx` → `.page-in`, 220 ms fade and 6 px rise.
- Homepage entrance: `.rise` with staggered `--d` delays (600 ms); route lines draw (`.route`), markers pop (`.marker`).
- Scroll reveals: `.reveal` → `.is-in`, once per element; shorter and smaller under 640 px.
- Parallax: hero illustration only, small translate, disabled under 768 px.
- Analytical changes (tab, play, game, model swaps): `.fade-swap` 200 ms; colour transitions 150 ms.
- `prefers-reduced-motion: reduce` turns off all of the above and shows content immediately.
- No scroll hijacking, no custom cursor, no autoplay beyond the hero entrance.

## Breakpoints and accessibility

Checked at 1440, 1024, 768 and 390 px. Coverage collapses to field → evidence with the play library in a drawer;
wide tables scroll inside their own container. One `h1` per page, labelled controls, visible focus, keyboard-operable
tabs, dialog and menu. Not done: a screen-reader (VoiceOver) pass.

## Cinematic artwork and playback

`StadiumBackdrop` uses original optimized WebP artwork with dark readability layers. The Home hero adds illustrative chalk routes; these are decorative, not observed player trajectories. `TeamPortrait` places original helmet artwork behind team names, with colour overlays that do not claim to reproduce official uniforms. See `ASSETS.md`.

Highlights places an interaction-loaded official YouTube player above `ClipEvidence`: model ranking and observed loudness remain separate lanes, and editorial labels remain evaluation-only. A compact full-game overview supplies navigation. Selected intervals have start/end limits, replay/next controls, a synchronized cursor, and an external source fallback. No graph is described as feature attribution; two-second source bins constrain evidence resolution. Source timestamps use existing trim offsets and are not independently frame-verified.

The new browser regression suite checks desktop/mobile overflow and saved matchup presentation. Its mocked player tests verify seek, reuse, boundary pause, replay and embedding-error states; they do not establish real YouTube embeddability.
