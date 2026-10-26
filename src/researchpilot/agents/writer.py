import json
import re
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from researchpilot.tools.write import ReportWritingTools
from researchpilot.utils import get_gemini_api_key


class ReportParagraph(BaseModel):
    """A report paragraph and the source URLs supporting it."""

    text: str = Field(min_length=1)
    source_urls: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    """A thematic section in the report."""

    heading: str = Field(min_length=1)
    paragraphs: list[ReportParagraph] = Field(min_length=1)


class ReportDraft(BaseModel):
    """Structured draft returned by the language model."""

    title: str = Field(min_length=1)
    introduction: str = Field(min_length=1)
    sections: list[ReportSection] = Field(min_length=1)
    conclusion: str = Field(min_length=1)


WRITER_PROMPT = """
You are the Writer Agent of ResearchPilot.

Produce a complete, detailed research report, not an executive summary. Use only
the supplied analysis and source metadata. The input contains atomic facts with
evidence, findings grouped by thematic category, source summaries, and any
detected contradictions.

STRICT SOURCE RULES:
- Use only the source URLs explicitly provided in the input.
- Never invent, guess, reconstruct, or introduce a new source URL.
- Every citation must match one of the supplied URLs exactly.
- Do not cite URLs found in your pretrained knowledge or in unrelated context.
- If the supplied sources do not support a claim, qualify the claim or omit it.
- Do not add a bibliography entry unless its URL belongs to the supplied sources.

Requirements:
- Write in the requested output language.
- Do not invent facts, quotations, sources, dates, or URLs.
- Cover every atomic fact at least once. Do not omit facts to make the report
  shorter.
- Create one thematic section per category in organized_findings. Merge only
  closely related categories, and preserve every distinct theme.
- Develop each fact into readable prose, preferably in its own paragraph or in
  a small cluster of closely related facts. State the finding, use the evidence
  or context to explain it, and explain its significance. Do not compress
  unrelated facts into one sentence, copy the fact statements, or turn the
  report into a bullet list.
- Use the source summaries to add context and comparisons, without introducing
  facts that are absent from the analysis.
- If contradictions exist, discuss each one in a dedicated section and explain
  the competing positions without declaring a false winner.
- Keep the introduction focused on the research question, scope, and themes.
  It should be one or two substantial paragraphs.
- Write a conclusion that synthesizes the main findings, limitations, and
  practical implications without adding new facts.
- For each factual paragraph, include the exact source URL(s) supporting it.
- Aim for the target length in coverage_requirements when the evidence supports
  it. Never pad, repeat, or add generic filler.
- Do not present translated evidence as a verbatim quotation; paraphrase it.
- Return only the structured report.
"""

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=get_gemini_api_key(),
)
structured_llm = llm.with_structured_output(ReportDraft)


def _language_name(language: str) -> str:
    """Convert common language codes to readable language names."""
    names = {
        "fr": "French",
        "en": "English",
        "ar": "Arabic",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "nl": "Dutch",
        "ru": "Russian",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "tr": "Turkish",
    }
    code = (language or "en").strip().lower().split("-")[0]
    return names.get(code, code)


def _analysis_payload(analysis: dict[str, Any] | BaseModel) -> dict[str, Any]:
    """Normalize the analysis result to a dictionary."""
    if isinstance(analysis, BaseModel):
        return analysis.model_dump()
    if isinstance(analysis, dict):
        return analysis
    raise TypeError("analysis must be a dictionary or a Pydantic model")


def _source_urls(sources: list[dict[str, Any]]) -> set[str]:
    """Collect exact, non-empty URLs from source records."""
    return {
        url.strip()
        for source in sources
        if isinstance(source, dict)
        for url in [source.get("url")]
        if isinstance(url, str) and url.strip()
    }


def _draft_word_count(draft: ReportDraft) -> int:
    """Count the prose words in a draft, including title and conclusion."""
    parts = [draft.title, draft.introduction, draft.conclusion]
    parts.extend(
        paragraph.text for section in draft.sections for paragraph in section.paragraphs
    )
    return len(re.findall(r"\b[\w'-]+\b", " ".join(parts)))

def _draft_to_sections(draft: ReportDraft) -> list[dict[str, Any]]:
    """Convert the structured LLM draft to ReportWritingTools' input format."""
    return [
        {
            "heading": section.heading,
            "paragraphs": [
                {
                    "text": paragraph.text,
                    **(
                        {"source_ids": paragraph.source_urls}
                        if paragraph.source_urls
                        else {}
                    ),
                }
                for paragraph in section.paragraphs
            ],
        }
        for section in draft.sections
    ]


def _draft_report(
    analysis: dict[str, Any],
    sources: list[dict[str, Any]],
    language: str,
    user_query: str | None,
    organized_findings: dict[str, list[dict[str, Any]]],
) -> ReportDraft:
    """Ask the LLM for a structured report draft."""
    request = {
        "requested_language": _language_name(language),
        "language_code": language,
        "research_question": user_query or analysis.get("research_question", ""),
        "analysis": analysis,
        "organized_findings": organized_findings,
        "available_sources": [
            {
                key: source.get(key)
                for key in ("url", "title", "author", "publisher")
                if source.get(key) is not None
            }
            | (
                {"published_date": source.get("published_date") or source.get("date")}
                if source.get("published_date") or source.get("date")
                else {}
            )
            for source in sources
            if isinstance(source, dict) and isinstance(source.get("url"), str)
        ],
    }

    response = structured_llm.invoke(
        [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(
                content=json.dumps(
                    request,
                    ensure_ascii=False,
                    default=str,
                )
            ),
        ]
    )
    return cast(ReportDraft, response)


def generate_report(
    analysis: dict[str, Any] | BaseModel,
    sources: list[dict[str, Any]],
    language: str = "en",
    user_query: str | None = None,
    output_format: Literal["markdown", "md", "pdf"] | None = "markdown",
    filename: str | None = None,
    output_dir: str = "reports",
) -> dict[str, Any]:
    """Draft, format, validate, and optionally export the research report."""
    language = language.strip().lower() if isinstance(language, str) else "en"
    language = language or "en"

    if not sources:
        raise ValueError("At least one source is required to write a cited report.")

    analysis_data = _analysis_payload(analysis)
    available_urls = _source_urls(sources)
    if not available_urls:
        raise ValueError("At least one source with a valid URL is required.")

    # Create one utility instance and reuse it for every report operation.
    active_tools = ReportWritingTools(
        output_dir=output_dir,
        language=language,
    )

    raw_findings = (
        analysis_data.get("atomic_facts") or analysis_data.get("findings") or []
    )
    findings = (
        [item for item in raw_findings if isinstance(item, dict)]
        if isinstance(raw_findings, list)
        else []
    )
    grouped_findings = active_tools.organize_findings(findings) if findings else {}

    draft = _draft_report(
        analysis=analysis_data,
        sources=sources,
        language=language,
        user_query=user_query,
        organized_findings=grouped_findings,
    )

    sections = _draft_to_sections(draft)
    for section in sections:
        for paragraph in section["paragraphs"]:
            paragraph_sources = paragraph.get(
                "source_ids",
                paragraph.get("source_urls", []),
            )
            invalid_urls = [
                url for url in paragraph_sources if url not in available_urls
            ]
            if invalid_urls:
                raise ValueError(
                    "The draft contains source URLs that were not supplied: "
                    + ", ".join(invalid_urls)
                )

    registry = active_tools.build_citation_registry(sources)
    report = active_tools.generate_report(
        title=draft.title,
        introduction=draft.introduction,
        sections=sections,
        conclusion=draft.conclusion,
        registry=registry,
        language=language,
    )

    result: dict[str, Any] = {
        "query": user_query or analysis_data.get("research_question", ""),
        "language": language,
        "report": report,
        "registry": registry,
        "validation": {
            "structure": active_tools.validate_report(report, language=language),
            "citations": active_tools.validate_citations(report, registry),
        },
        "export_path": None,
    }

    if output_format is not None:
        selected_format = output_format.strip().lower().lstrip(".")
        if selected_format not in {"markdown", "md", "pdf"}:
            raise ValueError("output_format must be 'markdown', 'md', or 'pdf'.")

        extension = ".pdf" if selected_format == "pdf" else ".md"
        export_filename = filename or f"research_report{extension}"
        if not export_filename.lower().endswith(extension):
            raise ValueError(
                f"A {selected_format} filename must end with '{extension}'."
            )

        result["export_path"] = active_tools.export_report(
            report=report,
            output_format="pdf" if selected_format == "pdf" else "markdown",
            filename=export_filename,
            output_dir=output_dir,
        )

    return result


def writer_agent(
    analysis: dict[str, Any] | BaseModel,
    sources: list[dict[str, Any]],
    language: str = "en",
    user_query: str | None = None,
    output_format: Literal["markdown", "md", "pdf"] | None = "markdown",
    filename: str | None = None,
    output_dir: str = "reports",
) -> dict[str, Any]:
    """Public agent entry point; delegates to the report-generation pipeline."""
    return generate_report(
        analysis=analysis,
        sources=sources,
        language=language,
        user_query=user_query,
        output_format=output_format,
        filename=filename,
        output_dir=output_dir,
    )
