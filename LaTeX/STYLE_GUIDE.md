# Curia Paper Style Guide (Shared Agent Reference)

## Scope
This guide governs edits to the paper in [main.tex](main.tex). It is written for human authors and AI agents collaborating on this document.

## Primary Objective
Reframe the paper so the main research contribution is model-size minimization while preserving two core capabilities:
- Keyword extraction quality
- Temporal extraction quality

## Secondary Objective
Maintain a concise system overview of Curia only as context for the compression/minimization narrative.

## Collaboration Rule (Non-Negotiable)
Before making any design or direction decision, ask the user first and wait for confirmation.

## Audience and Tone
- Audience: Class project graders (IEEE-style writeup)
- Tone: Formal, cautious, evidence-first
- Stance: Make claims proportional to evidence

## Language Rules
- Prefer precise, testable language over promotional phrasing.
- Use verbs such as "measured", "observed", "compared", "evaluated".
- Avoid unsupported superlatives such as "best", "state-of-the-art", "optimal".
- Distinguish clearly between measured outcomes and hypotheses.
- Keep sentence structure direct and technical.

## Role Definitions
- User role:
  - Final decision-maker on framing, claims, and emphasis
  - Approves all direction changes before implementation
- Lead writing agent role:
  - Ask targeted questions for each section
  - Convert approved answers into IEEE-ready prose
  - Preserve internal consistency across sections
- Support/review agent role:
  - Check claim-evidence alignment
  - Check citation coverage for external claims
  - Flag ambiguity, over-claiming, and missing metrics

## Section Functions
- Title:
  - Signal efficiency/minimization focus and preserved functionality
- Abstract:
  - Problem, efficiency goal, method summary, key quantitative outcomes, significance
- Keywords:
  - Include model compression/efficiency and extraction tasks
- Introduction:
  - Frame the size-performance tradeoff and research question
- Related Work:
  - Position against compression, lightweight NLP, temporal extraction, query understanding
- Methodology:
  - Describe minimization strategy, candidate model sizes, retained feature definitions, fallback behavior
- Experiments:
  - Report setup, model variants, metrics, and size-performance frontier
- Conclusion:
  - Summarize tradeoffs, practical implications, and limitations

## Evidence and Reporting Rules
Every comparison table should include, where available:
- Parameter count
- Memory footprint (MB)
- Inference latency (ms)
- Keyword extraction quality metric(s)
- Temporal extraction quality metric(s)

When presenting results:
- Include hardware/runtime context
- State whether numbers are exact, estimated, or placeholders
- Prefer paired comparisons to isolate the effect of model size

## Claim Discipline
- Only claim "preserved" capability when metrics are comparable to baseline under stated thresholds.
- If thresholds are not yet defined, label as "preliminary" and request threshold confirmation.
- If data is incomplete, explicitly state limitation and avoid causal claims.

## Citation Discipline
- Cite external factual claims, methods, and baselines.
- Do not cite for internal project-specific observations unless externally sourced.
- Keep citation keys in text synchronized with [references.bib](references.bib).

## Section-by-Section Workflow
For each section, follow this sequence:
1. Ask focused questions (content, claims, evidence, scope).
2. Summarize intended direction in 3-6 bullets.
3. Ask for explicit user approval.
4. Draft or revise the section in IEEE format.
5. Run a quick consistency pass against this guide.

## First Section Startup Checklist (Title, Abstract, Keywords)
Before drafting this section, confirm:
- Target title direction (efficiency-first wording)
- Core quantitative results to include in abstract
- Exact definition of "preserving" keyword and temporal extraction
- Whether to include fallback mechanisms in abstract
- Preferred keyword list (5-8 terms)

## Approved Decisions (Current Round)
- Title direction approved:
  - Curia-Lite: Minimizing Model Size for Keyword and Temporal Query Extraction
- Baseline anchor approved:
  - Largest Curia model variant
- Abstract quantitative anchor approved:
  - Model-size range from 27B parameters to 256M parameters
- Preservation criterion approved:
  - No fixed threshold; describe size-quality tradeoff qualitatively
- Abstract robustness note approved:
  - Include fallback mechanisms briefly
- Keywords approved:
  - Lightweight Models
  - Temporal Grounding
  - Information Retrieval
  - Query Parsing
  - Edge Inference

## Approved Decisions (Introduction Round)
- Intro framing approved:
  - Size-performance tradeoff in event query understanding
- Research questions approved:
  - Can model size be reduced while retaining keyword extraction?
  - Can model size be reduced while retaining temporal extraction?
- Contribution format approved:
  - Three focused contributions
- Numeric range placement approved:
  - Keep specific size range details in Experiments (not early Introduction emphasis)
- Claim strength approved:
  - State that temporal structure does not preserve well under downsizing, while keyword extraction remains viable at very small models (including 256M)
- Fallback placement approved:
  - Brief mention in Introduction, details in Methodology

## Approved Decisions (Methodology Round)
- Core focus approved:
  - Emphasize divergence between keyword extraction and temporal extraction under model downsizing
- Divergence definition approved:
  - Include both performance-gap divergence and error-type divergence
- Minimization mechanism approved:
  - Model family scaling only (no quantization, pruning, or distillation)
- Model variants approved:
  - 31B, 27B, 5B, 1B, 0.5B, 256M
- Metrics approved:
  - Keyword F1, Temporal exact match, latency, memory
- Fallback policy approved:
  - Controlled robustness layer applied consistently across all model sizes
- Citation policy approved:
  - New references require user approval before insertion
- Methodology citation additions approved this round:
  - Kaplan et al. (2020)
  - Hoffmann et al. (2022)
