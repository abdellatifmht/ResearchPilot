"""Report-writing tools for ResearchPilot.

``ReportWritingTools`` follows the same pattern as ``TavilySearchTools`` and
``SourceContentTools``: it owns the writing-stage behaviour, while the writer
agent wraps the relevant methods with LangChain's ``@tool`` decorator.

The class covers three responsibilities:

1. Build a stable citation registry from the analysed sources.
2. Assemble a structured Markdown report (introduction, thematic sections,
   conclusion, references) with ``[1]``, ``[2]`` citation markers.
3. Save the report as Markdown.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

__all__ = ["ReportWritingTools"]


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
HEADING_PATTERN = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
REFERENCE_LINE_PATTERN = re.compile(r"^\s*\[\d+\]\s+.*$", re.MULTILINE)

_LABELS = {
    "fr": {
        "title": "Rapport de recherche",
        "introduction": "Introduction",
        "conclusion": "Conclusion",
        "references": "Références",
        "section": "Section",
        "no_content": "_Aucun contenu fourni._",
        "no_source": "_Aucune source fournie._",
        "untitled": "Source sans titre",
    },
    "en": {
        "title": "Research Report",
        "introduction": "Introduction",
        "conclusion": "Conclusion",
        "references": "References",
        "section": "Section",
        "no_content": "_No content provided._",
        "no_source": "_No source provided._",
        "untitled": "Untitled source",
    },
}


class ReportWritingTools:
    """Generate, validate, and export cited Markdown reports."""

    def __init__(
        self,
        output_dir: str = "reports",
        language: str = "fr",
    ) -> None:
        self.output_dir = output_dir
        self.language = language

    def organize_findings(
        self,
        findings: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group findings into thematic sections for a report.

        Each finding may contain: statement/claim, summary, category/theme,
        status, and source_ids. Findings without a category are grouped under
        "General". The returned mapping preserves the input order inside each
        theme; use its keys as section headings when calling generate_report.
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for finding in findings:
            if not isinstance(finding, dict):
                continue
            category = str(
                finding.get("category") or finding.get("theme") or "General"
            ).strip()
            groups[category or "General"].append(finding)

        return dict(groups)

    def generate_report(
        self,
        title: str,
        introduction: str,
        sections: list[dict[str, Any]],
        conclusion: str,
        sources: list[dict[str, Any]] | None = None,
        registry: list[dict[str, Any]] | None = None,
        language: str | None = None,
    ) -> str:
        """Generate a structured Markdown report with numbered citations.

        The report always contains a title, an introduction, thematic sections,
        a conclusion, and a references list.

        Each section accepts:
        - heading: thematic heading.
        - content: a string, or paragraphs: a list of strings/mappings.
        - source_ids: optional sources cited at the end of the section.

        A paragraph may be a string or a mapping such as
        {"text": "Claim...", "source_ids": ["S1", "S2"]}. Paragraph-level
        citations take precedence over section-level citations. Citations can
        be numbers, source ids, or URLs and are rendered as [1], [2].

        Prefer passing the output of build_citation_registry as registry. When
        registry is omitted, it is built automatically from sources. When both
        are omitted, the report has no references.
        """
        active_registry = registry
        if not active_registry and sources:
            active_registry = self.build_citation_registry(sources)

        return self._render_report(
            title=title,
            introduction=introduction,
            sections=sections,
            conclusion=conclusion,
            registry=active_registry or [],
            language=language or self.language,
        )

    def _render_report(
        self,
        title: str,
        introduction: str,
        sections: list[dict[str, Any]],
        conclusion: str,
        registry: list[dict[str, Any]],
        language: str,
    ) -> str:
        labels = self._labels(language)
        lines = [
            f"# {title.strip() or labels['title']}",
            "",
            f"## {labels['introduction']}",
            "",
            introduction.strip() or labels["no_content"],
            "",
        ]

        for index, section in enumerate(sections, start=1):
            if not isinstance(section, dict):
                continue
            heading = str(
                section.get("heading")
                or section.get("title")
                or f"{labels['section']} {index}"
            ).strip()
            lines.extend([f"## {heading}", ""])
            paragraphs = self._section_paragraphs(
                section,
                registry,
                labels["no_content"],
            )
            lines.extend(["\n\n".join(paragraphs), ""])

        lines.extend(
            [
                f"## {labels['conclusion']}",
                "",
                conclusion.strip() or labels["no_content"],
                "",
                f"## {labels['references']}",
                "",
            ]
        )

        ordered = sorted(
            (
                source
                for source in registry
                if isinstance(source, dict)
                and str(source.get("reference", "")).isdigit()
            ),
            key=lambda source: int(source["reference"]),
        )
        if ordered:
            lines.extend(self._reference_line(source, labels) for source in ordered)
        else:
            lines.append(labels["no_source"])

        return "\n".join(lines).rstrip() + "\n"

    def _section_paragraphs(
        self,
        section: dict[str, Any],
        registry: list[dict[str, Any]],
        no_content: str,
    ) -> list[str]:
        """Render one section's paragraphs and apply citation markers."""
        raw_paragraphs = self._paragraph_values(section.get("paragraphs"))
        if raw_paragraphs:
            items = [
                self._paragraph_text_and_references(item) for item in raw_paragraphs
            ]
        else:
            content = str(section.get("content") or "").strip()
            chunks = [
                chunk.strip()
                for chunk in re.split(r"\n\s*\n", content)
                if chunk.strip()
            ]
            items = [(chunk, None) for chunk in chunks]

        section_references = None
        for key in ("source_ids", "sources", "references", "citations"):
            if key in section:
                section_references = self._coerce_references(section[key])
                break

        if not items:
            return [no_content]

        paragraphs: list[str] = []
        last_index = len(items) - 1
        for index, (text, references) in enumerate(items):
            effective_references = (
                references
                if references is not None
                else section_references
                if index == last_index
                else None
            )
            if effective_references:
                text = self.insert_citations(text, effective_references, registry)
            paragraphs.append(text)

        return paragraphs

    @staticmethod
    def _paragraph_values(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return []

    @staticmethod
    def _paragraph_text_and_references(item: Any) -> tuple[str, list[Any] | None]:
        """Read a paragraph expressed either as text or as a structured mapping."""
        if isinstance(item, dict):
            text = (
                item.get("text") or item.get("content") or item.get("paragraph") or ""
            )
            references: list[Any] | None = None
            for key in ("source_ids", "sources", "references", "citations"):
                if key in item:
                    references = ReportWritingTools._coerce_references(item[key])
                    break
            return str(text), references
        if isinstance(item, str):
            return item, None
        return str(item), None


    def build_citation_registry(
        self,
        sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create a stable numbered citation registry from sources.

        Sources are numbered in the order supplied, starting at [1], and are
        de-duplicated by normalized URL. Pass sources in a deterministic order
        so the same report keeps the same references when regenerated.
        """
        registry: list[dict[str, Any]] = []
        seen_urls: set[str] = set()
        used_ids: set[str] = set()

        for source in sources:
            if not isinstance(source, dict):
                continue

            url = source.get("url")
            if not isinstance(url, str) or not url.strip():
                continue

            normalized_url = self._normalize_url(url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)

            reference = len(registry) + 1
            base_id = (
                str(source.get("id") or f"S{reference}").strip() or f"S{reference}"
            )
            source_id = base_id
            suffix = 2
            while source_id in used_ids:
                source_id = f"{base_id}-{suffix}"
                suffix += 1
            used_ids.add(source_id)

            registry.append(
                {
                    "id": source_id,
                    "reference": reference,
                    "citation": f"[{reference}]",
                    "title": str(source.get("title") or "Untitled source").strip(),
                    "url": url.strip(),
                    "normalized_url": normalized_url,
                    "author": self._author_text(source),
                    "publisher": source.get("publisher"),
                    "published_date": self._published_date(source),
                }
            )

        return registry

    def insert_citations(
        self,
        text: str,
        references: list[int | str],
        registry: list[dict[str, Any]],
    ) -> str:
        """Insert [1], [2] citation markers at the end of a passage.

        References can be registry numbers, source ids (for example "S3"),
        URLs, or markers such as "[2]". Markers already present in the passage
        are not duplicated. Markers are placed before the final punctuation.
        """
        numbers: list[int] = []
        for reference in references:
            number = self._resolve_reference(reference, registry)
            if number not in numbers:
                numbers.append(number)

        existing = {int(number) for number in CITATION_PATTERN.findall(text)}
        missing = [number for number in numbers if number not in existing]
        if not missing:
            return text

        markers = "".join(f"[{number}]" for number in missing)
        return self._append_markers(text, markers)

    def validate_citations(
        self,
        report: str,
        registry: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Check that in-text citation markers match the citation registry.

        Reference-list lines such as "[1] Title. URL" are ignored so that
        unused_references reflects sources never cited in the report body. This
        does not verify that a source actually supports the adjacent claim.
        """
        body = REFERENCE_LINE_PATTERN.sub("", report)
        matches = CITATION_PATTERN.findall(body)
        used_numbers = sorted({int(number) for number in matches})
        known_numbers = {
            int(source["reference"])
            for source in registry
            if isinstance(source, dict) and str(source.get("reference", "")).isdigit()
        }

        unknown = sorted(set(used_numbers) - known_numbers)
        unused = sorted(known_numbers - set(used_numbers))

        return {
            "valid": not unknown,
            "used_references": used_numbers,
            "unknown_references": unknown,
            "unused_references": unused,
            "citation_count": len(matches),
            "message": (
                "All citations exist in the registry."
                if not unknown
                else f"Unknown citation numbers: {unknown}"
            ),
        }

    def _reference_line(
        self,
        source: dict[str, Any],
        labels: dict[str, str],
    ) -> str:
        """Format one registry entry as a readable Markdown reference."""
        reference = int(source["reference"])
        title = str(source.get("title") or labels["untitled"]).strip()
        url = str(source.get("url") or "").strip()
        author = self._author_text(source)
        publisher = str(source.get("publisher") or "").strip()
        published_date = self._published_date(source)

        segments: list[str] = []
        if author:
            segments.append(author.rstrip("."))
        segments.append(f"*{title.rstrip('.')}*")
        if publisher:
            segments.append(f"{publisher.rstrip('.')}")

        line = ". ".join(segments)
        if published_date:
            line = f"{line}, {published_date}"
        if url:
            return f"[{reference}] {line}. {url}"
        return f"[{reference}] {line}."

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize trivial URL differences for citation de-duplication."""
        parts = urlsplit(url.strip())
        scheme = (parts.scheme or "https").lower()
        netloc = parts.netloc.lower()
        path = parts.path.rstrip("/") or "/"
        # Fragments never identify different content, while query strings can.
        return urlunsplit((scheme, netloc, path, parts.query, ""))

    @staticmethod
    def _author_text(source: dict[str, Any]) -> str:
        author = source.get("author") or source.get("authors")
        if isinstance(author, str):
            return author.strip()
        if isinstance(author, (list, tuple)):
            names = [str(name).strip() for name in author if str(name).strip()]
            return ", ".join(names)
        return ""

    @staticmethod
    def _published_date(source: dict[str, Any]) -> Any:
        return source.get("published_date") or source.get("date")

    @staticmethod
    def _coerce_references(value: Any) -> list[Any] | None:
        """Normalize a single reference or a collection of references."""
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError("Citation references must be numbers, source ids, or URLs.")
        if isinstance(value, (str, int)):
            return [value]
        if isinstance(value, (list, tuple, set)):
            references: list[Any] = []
            for item in value:
                if isinstance(item, dict):
                    candidate = (
                        item.get("id")
                        or item.get("url")
                        or item.get("reference")
                        or item.get("citation")
                    )
                    if candidate is not None:
                        references.append(candidate)
                elif isinstance(item, (str, int)) and not isinstance(item, bool):
                    references.append(item)
            return references
        raise ValueError("Citation references must be a number, source id, or URL.")

    def _resolve_reference(
        self,
        reference: Any,
        registry: list[dict[str, Any]],
    ) -> int:
        """Resolve a source id, URL, marker, or number to its registry number."""
        if isinstance(reference, bool):
            raise TypeError("Citation references must be numbers, source ids, or URLs.")

        if isinstance(reference, int):
            number = reference
        else:
            text = str(reference).strip()
            if re.fullmatch(r"\[?\d+\]?", text):
                number = int(text.strip("[]"))
            else:
                index = self._registry_index(registry)
                number = index.get(text) or index.get(text.casefold())
                if number is None:
                    raise ValueError(f"Unknown source reference: {reference!r}")

        known_numbers = {
            int(source["reference"])
            for source in registry
            if isinstance(source, dict) and str(source.get("reference", "")).isdigit()
        }
        if number not in known_numbers:
            raise ValueError(f"Citation [{number}] does not exist in the registry.")
        return number

    @staticmethod
    def _registry_index(registry: list[dict[str, Any]]) -> dict[str, int]:
        """Index a registry by id, citation marker, URL, and reference number."""
        index: dict[str, int] = {}
        for source in registry:
            if not isinstance(source, dict):
                continue
            try:
                reference = int(source["reference"])
            except KeyError, TypeError, ValueError:
                continue

            url = source.get("url")
            keys = [source.get("id"), source.get("citation"), str(reference)]
            if isinstance(url, str) and url.strip():
                keys.extend(
                    (
                        url.strip(),
                        source.get("normalized_url")
                        or ReportWritingTools._normalize_url(url),
                    )
                )

            for key in keys:
                if isinstance(key, (str, int)) and str(key).strip():
                    text = str(key).strip()
                    index[text] = reference
                    index[text.casefold()] = reference

        return index

    @staticmethod
    def _append_markers(text: str, markers: str) -> str:
        """Append markers before the final punctuation, preserving whitespace."""
        if not text:
            return text

        stripped = text.rstrip()
        trailing = text[len(stripped) :]
        match = re.search(r"[.!?…]+[\"'”’)]*$", stripped)
        if match:
            body = stripped[: match.start()].rstrip()
            punctuation = stripped[match.start() :]
            return f"{body} {markers}{punctuation}{trailing}"
        return f"{stripped} {markers}{trailing}"

    def validate_report(
        self,
        report: str,
        required_sections: list[str] | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Validate the basic structure of a generated Markdown report.

        By default, checks for a title, Introduction, Conclusion, and References
        (or their French equivalents when language starts with "fr").
        """
        labels = self._labels(language or self.language)
        if required_sections is None:
            required_sections = [
                labels["introduction"],
                labels["conclusion"],
                labels["references"],
            ]

        headings = HEADING_PATTERN.findall(report)
        missing_sections = [
            section
            for section in required_sections
            if not any(section.casefold() in heading.casefold() for heading in headings)
        ]

        has_title = bool(re.search(r"^#\s+\S", report, flags=re.MULTILINE))
        valid = bool(report.strip()) and has_title and not missing_sections

        return {
            "valid": valid,
            "has_title": has_title,
            "headings": headings,
            "missing_sections": missing_sections,
            "word_count": len(re.findall(r"\b[\w'-]+\b", report)),
            "character_count": len(report),
        }

    def export_markdown(
        self,
        report: str,
        filename: str = "research_report.md",
        output_dir: str | None = None,
    ) -> str:
        """Save a Markdown report as a UTF-8 file and return its path."""
        return str(
            self._write_markdown(
                report,
                filename,
                output_dir or self.output_dir,
            )
        )

    def export_report(
        self,
        report: str,
        output_format: str = "markdown",
        filename: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Export a report as Markdown.

        Markdown is always the source format.
        """
        selected = output_format.strip().lower().lstrip(".")
        destination_dir = output_dir or self.output_dir

        if selected == "markdown":
            return self.export_markdown(
                report,
                filename or "research_report.md",
                destination_dir,
            )

        else:
            raise ValueError("output_format must be 'markdown'")

    def _write_markdown(
        self,
        report: str,
        filename: str,
        output_dir: str,
    ) -> Path:
        if not report.strip():
            raise ValueError("report cannot be empty")

        safe_name = self._safe_filename(filename, ".md")
        base = Path(output_dir).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        destination = (base / safe_name).resolve()

        if destination.parent != base:
            raise ValueError("Invalid output path")

        destination.write_text(report, encoding="utf-8")
        return destination

    @staticmethod
    def _safe_filename(filename: str, suffix: str) -> str:
        """Reject paths and wrong extensions for generated files."""
        name = Path(filename).name
        if name != filename or Path(name).suffix.lower() != suffix:
            raise ValueError(f"filename must be a simple filename ending in {suffix}")
        return name

    @staticmethod
    def _labels(language: str) -> dict[str, str]:
        """Return localized report labels, falling back to English."""
        return _LABELS["fr" if language.strip().lower().startswith("fr") else "en"]
