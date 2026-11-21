# Exercise 8 — Industrial Decision Making

| | |
|---|---|
| **Duration** | 20 min (discussion-based) |
| **Required hardware** | Any laptop |
| **Required dataset** | None (stored results); optional: any installed dataset in Compare Methods |
| **Required checkpoint** | None |

## Learning objectives
- Discuss how anomaly scores become industrial OK/NOK decisions.
- Balance latency, false rejects (scrap, re-inspection) and false accepts (escapes to the customer).
- Understand the limits of research models in production.

## Instructions
1. **Educational Experiments → C**: for two verified models, write down the false-acceptance and
   false-rejection rates at their *valid* thresholds (green column).
2. Choose an acceptable FAR/FRR for a hypothetical line *before* moving the slider; then check which models
   meet it at their valid threshold.
3. Read the indicative latency of the PatchCore variants in **Educational Experiments → A** and decide whether
   they fit a 100 ms cycle time including image capture and communication.
4. Discuss: what must be monitored after deployment (lighting drift, new defect types, calibration)?

## Important
A research anomaly-detection model is **not**, by itself, a certified machinery safety system. Safety
functions require dedicated, validated and certified safety equipment and procedures; IDD Lab is an
educational tool.

## Expected qualitative observations
- The "best" model depends on the costs of each error type and on the cycle time.
- Thresholds estimated from 19–37 validation images are themselves uncertain.

## Reflection questions
- Which costs more in your industry: a false reject or a false accept?
- Who should own the threshold decision in a factory, and how should it be documented?

## Scientific references
`hecklerkram2026mvtecad2`, `batzner2024efficientad` (latency-oriented design), `bergmann2021mvtecad_ijcv`.

## Alternative path
- Entirely discussion-based with stored results; no installation beyond the core profile is needed.
