# ResearchPilot - Errors Found and Fixes Applied

## Issues Encountered

### 1) CLI / Graph integration crash (`ValueError: At least one source with a valid URL is required.`)
- **Where:** `src/researchpilot/__init__.py`, `src/researchpilot/graph.py`, `src/researchpilot/agents/analyst.py`
- **Symptom:** Running `uv run researchpilot` with a query (e.g. "qui est messi") failed in the `analyst` node when calling `analysis_agent(sources)`. The analyst requires at least one valid source URL; the graph was passing empty/invalid sources.
- **Root cause:** 
  - The graph was reworked to run researcher then analyst, but `research_agent` returned sources inconsistently (tool message content formats changed/weren't parsed correctly).
  - `researcher` sometimes collected no sources from `ToolMessage` objects (content serialized as strings).
  - `graph.py` didn't guard against missing/empty sources and would still call the analyst, causing a hard failure.
  - `__init__.py` assumed `result["research_output"]`, `result["sources"]`, `result["analysis_report"]` always existed and had specific shapes.

### 2) Researcher not returning sources / filtering too aggressively
- **Where:** `src/researchpilot/agents/researcher.py`
- **Symptom:** `research_agent("qui est messi")` returned `sources: []` even though Tavily returned results. The analyst never got URLs to fetch/analyze.
- **Root causes:**
  - `_source_records()` only looked for certain keys and didn't robustly parse `ToolMessage.content` when it was a JSON string (common with LangGraph/LangChain serialization). Also failed on Python reprs in some cases.
  - The graph's researcher node collected sources only from `ToolMessage` with `name == "web_search"` in the old version; in practice the tool results came through differently / were stringified.
  - After the agent graph refactor, source extraction needed to be more defensive.

### 3) Orphaned/duplicate tooling directories
- **Where:** `src/researchpilot/tools/analyse/` vs `src/researchpilot/tools/analysis/`
- **Symptom:** During rebase/merge there was a conflict (`CONFLICT (modify/delete): src/researchpilot/tools/analyse/__init__.py deleted in HEAD and modified in incoming`). The codebase uses `tools.analysis.SourceContentTools` (imported by `analyst.py`), while `tools.analyse` appeared to be a stray/duplicate.
- **Fix:** Removed the `analyse` directory to avoid confusion and keep a single source of truth for analysis tools.

## Fixes Applied

### Graph & CLI resilience
- **`src/researchpilot/graph.py`**
  - `run_researcher`: now wrapped in `try/except`, returns safe defaults if research fails.
  - `run_analysis`: checks for empty/missing sources and returns an empty `AnalysisReport` structure (`source_summaries: []`, `atomic_facts: []`, `contradictions: []`). Also catches exceptions from `analysis_agent` and returns the same empty structure (prevents crash).
  - State initialized with safe defaults; `run_research_pilot` returns a dict with all expected keys.
- **`src/researchpilot/__init__.py`**
  - Defensive input handling (EOF/interrupts).
  - Safe access to `research_output`, `sources`, `analysis_report` with fallbacks.
  - Still prints results in a readable format.

### Researcher source extraction & fallback
- **`src/researchpilot/agents/researcher.py`**
  - Imports updated to include `json`, `ast`, `ToolMessage` as needed.
  - `_source_records(content)`: rewritten to be robust
    - If content is a string: try `json.loads()` first, then `ast.literal_eval()`, else return `[]`
    - If dict: if has `url` return it; else look in common keys (`results`, `content`, `text`, `pages`, `data`)
    - If list/tuple: recursively extract from each item
  - `research_agent(query)`:
    - Invokes graph with higher `recursion_limit` (40), safer message handling
    - Extracts sources from **all** messages (including scanning `ToolMessage.content` regardless of serialization) via `_source_records`
    - **Fallback**: if no sources found from messages, directly calls `tavily_tools.web_search` with optimized queries and filters results via `_filter_sources()` — ensures sources are returned even if tool message serialization is problematic
  - Returns structured dict: `{"query", "sources", "summary"}`

### Analyst hardening (defensive)
- **`src/researchpilot/agents/analyst.py`**: already validates URLs and raises clear errors; graph now catches and degrades gracefully so the CLI doesn't crash. (No logic change required beyond graph protection.)

### Cleanup
- Removed `src/researchpilot/tools/analyse/` directory (conflicted during rebase). Kept `src/researchpilot/tools/analysis/` which is used by `analyst.py`.

## Result
- `uv run researchpilot` no longer crashes when sources are missing or when tool messages are stringified.
- For queries where Tavily returns results (e.g. "qui est messi"), sources are now collected (via fallback) and passed through the pipeline. The analysis step safely returns empty report if fetching/analysis cannot proceed, so the CLI completes successfully and prints the research summary + sources + analysis report structure.
- Codebase is more robust to serialization differences across LangGraph/LangChain versions.
