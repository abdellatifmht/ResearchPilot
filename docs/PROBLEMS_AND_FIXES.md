# ResearchPilot Problems and Fixes

This document records the problems found while testing the research workflow,
their root causes, the changes made, and the validation performed.

## 1. Valid sources were discarded before analysis

### Symptom

The command printed a useful research summary, but the `Sources` section was
empty:

```text
Sources:

Analysis Report:
Source Summaries: []
Atomic Facts: []
Contradictions: []
```

### Root cause

The `web_search` tool normalizes Tavily results into records containing
`relevance_score`. Later, `research_agent` passed those already-normalized
records through `_filter_sources` again.

`_filter_sources` only read the raw Tavily field `score`:

```python
score = result.get("score", 0)
```

Because normalized records did not have `score`, their score became `0` and
they were all removed by the minimum relevance check.

### Fix

Updated `_filter_sources` in
[`src/researchpilot/agents/researcher.py`](src/researchpilot/agents/researcher.py)
to accept both raw and normalized records:

```python
score = result.get("score", result.get("relevance_score", 0))
```

The function now also preserves:

- Existing `excerpt` values.
- Existing `reliability_score` values.
- Raw `content` values when present.

This makes source filtering idempotent: applying it more than once no longer
removes valid sources.

## 2. The analysis agent exceeded the Gemini input quota

### Symptom

Sources were displayed, but the analysis report was empty:

```text
Sources:
- https://people.umass.edu/klement/tlp/tlp-hyperlinked.html
- https://plato.stanford.edu/entries/wittgenstein-atomism
...

Analysis Report:
Source Summaries: []
Atomic Facts: []
Contradictions: []
```

### Root cause

The analyst fetched complete page contents and inserted every page into one
Gemini prompt. For the Wittgenstein query, seven pages contained approximately
1.5 million characters in total.

The Gemini request failed with:

```text
GoogleRateLimitError: 429 RESOURCE_EXHAUSTED
GenerateContentInputTokensPerModelPerMinute-FreeTier
```

The request exceeded the free-tier input-token limit.

### Fix

Updated `_content_for_analysis` in
[`src/researchpilot/agents/analyst.py`](src/researchpilot/agents/analyst.py)
to limit the content sent for each source to the existing
`PAGE_CONTENT_LIMIT` value:

```python
content = page["content"]
sections.append(
    f"SOURCE URL: {page['url']}\n{content[:PAGE_CONTENT_LIMIT]}"
)
```

This keeps every source URL in the analysis while preventing one very large
page or collection of pages from exhausting the model input quota.

## 3. Analysis failures were silently presented as successful empty reports

### Symptom

When analysis failed, the application printed empty arrays rather than
reporting that the analysis stage had failed. This made a quota or extraction
problem look like a valid conclusion containing no facts.

### Root cause

`run_analysis` caught every exception and returned a fabricated empty report:

```python
{
    "source_summaries": [],
    "atomic_facts": [],
    "contradictions": [],
}
```

`run_researcher` similarly caught every exception and returned empty research
data.

### Fix

Updated [`src/researchpilot/graph.py`](src/researchpilot/graph.py) so both
workflow stages raise an explicit error with the original exception preserved
as its cause:

```python
except Exception as exc:
    raise RuntimeError("Analysis stage failed.") from exc
```

and:

```python
except Exception as exc:
    raise RuntimeError("Research stage failed.") from exc
```

The application can no longer silently report an empty result when a stage
actually failed. The underlying exception remains available in the traceback
for diagnosis.

## 4. Source extraction and analysis use different content limits

### Observation

Tavily search results are compacted using `CHARACTER_LIMIT`, while full-page
extraction can return much larger content. Sending full extracted pages
directly to the analyst created an avoidable mismatch between retrieval size
and model input size.

### Fix

The analysis stage now applies `PAGE_CONTENT_LIMIT` before constructing the
model prompt. This keeps the retrieval layer capable of collecting content
while ensuring the model-facing layer has a predictable maximum input size.

## Validation performed

### Source filtering

Verified that a normalized source containing `relevance_score` remains in the
filtered output and retains its excerpt.

### Source extraction

Verified that the seven Wittgenstein URLs returned seven extracted pages,
including the PDF and web pages.

### Analysis

Before the fix, the same source set failed with Gemini `429
RESOURCE_EXHAUSTED`. After limiting the prompt, the analysis completed with:

```text
summaries: 7
facts: 8
contradictions: 0
```

### Syntax

Pylance syntax checks passed for:

- [`src/researchpilot/agents/analyst.py`](src/researchpilot/agents/analyst.py)
- [`src/researchpilot/graph.py`](src/researchpilot/graph.py)
- [`src/researchpilot/agents/researcher.py`](src/researchpilot/agents/researcher.py)

## Current behavior

- Valid sources are displayed.
- Each successfully extracted source can receive an individual summary.
- Atomic facts include source URLs and evidence.
- Contradictions are listed when the model finds incompatible claims.
- Research and analysis failures are raised explicitly instead of being
  converted into misleading empty arrays.

## Refactoring

The researcher agent previously contained both LLM orchestration and
research-data processing. The following research concerns now live in
[`src/researchpilot/tools/research/__init__.py`](src/researchpilot/tools/research/__init__.py):

- Complementary search-query construction.
- Source reliability and freshness scoring.
- Search-result filtering, ranking, and normalization.
- Parsing serialized source records returned in tool messages.
- Coordinating the three Tavily searches used by a research query.

[`src/researchpilot/agents/researcher.py`](src/researchpilot/agents/researcher.py)
now focuses on:

- Defining the LangChain tools exposed to the model.
- Running the researcher graph.
- Collecting the final summary and source records.

This separation keeps Tavily-specific behavior reusable and testable without
loading the researcher graph, while preserving the existing public
`research_agent` behavior.
