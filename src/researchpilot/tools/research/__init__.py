import ast
import json
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from researchpilot.utils import CHARACTER_LIMIT, PAGE_CONTENT_LIMIT, get_tavily_client

__all__ = [
    "TavilySearchTools",
    "filter_sources",
    "source_records",
]


TRUSTED_DOMAIN_SUFFIXES = (".gov", ".edu", ".int")
TRUSTED_DOMAINS = {
    "bbc.com",
    "reuters.com",
    "who.int",
    "un.org",
    "worldbank.org",
}
MIN_RELEVANCE_SCORE = 0.35


def _search_queries(query: str) -> list[str]:
    """Create complementary queries instead of relying on one search wording."""
    return [
        query,
        f"{query} official primary source",
        f"{query} latest evidence report",
    ]


def _reliability_score(url: str) -> float:
    """Estimate source reliability from domain signals available in a result."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().removeprefix("www.")

    if hostname in TRUSTED_DOMAINS or hostname.endswith(TRUSTED_DOMAIN_SUFFIXES):
        return 1.0
    if parsed.scheme == "https":
        return 0.6
    return 0.4


def _freshness_score(published_date: Any) -> float:
    """Return a comparable freshness score without discarding undated sources."""
    if not isinstance(published_date, str) or not published_date:
        return 0.5

    try:
        published = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
    except ValueError:
        return 0.5

    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_days = max((datetime.now(timezone.utc) - published).days, 0)
    return max(0.0, 1.0 - (age_days / 3650))


def filter_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter, rank, and normalize raw or already-normalized search results."""
    sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for result in results:
        url = result.get("url")
        title = result.get("title")
        score = result.get("score", result.get("relevance_score", 0))
        if (
            not isinstance(url, str)
            or not url
            or url in seen_urls
            or not isinstance(title, str)
            or not title
            or not isinstance(score, (int, float))
            or score < MIN_RELEVANCE_SCORE
        ):
            continue

        seen_urls.add(url)
        published_date = result.get("published_date")
        sources.append(
            {
                "url": url,
                "title": title,
                "excerpt": result.get("content", result.get("excerpt", "")),
                "relevance_score": score,
                "freshness_score": _freshness_score(published_date),
                "reliability_score": result.get(
                    "reliability_score",
                    _reliability_score(url),
                ),
                "published_date": published_date,
            }
        )

    sources.sort(
        key=lambda source: (
            source["relevance_score"]
            * source["freshness_score"]
            * source["reliability_score"]
        ),
        reverse=True,
    )
    return sources


def source_records(content: Any) -> list[dict[str, Any]]:
    """Extract source records from serialized tool-message content."""
    if isinstance(content, str):
        serialized = content.strip()
        if not serialized:
            return []
        try:
            return source_records(json.loads(serialized))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        try:
            return source_records(ast.literal_eval(serialized))
        except (SyntaxError, ValueError, TypeError):
            return []

    if isinstance(content, dict):
        if isinstance(content.get("url"), str):
            return [content]
        for key in ("results", "content", "text", "pages", "data"):
            if key in content:
                return source_records(content[key])
        return []

    if isinstance(content, (list, tuple)):
        records: list[dict[str, Any]] = []
        for item in content:
            records.extend(source_records(item))
        return records

    return []


class ResearchError(RuntimeError):
    """Raised when a topic cannot be researched."""

class TavilySearchTools:
    """A class to perform searches using the Tavily API."""

    def __init__(self):
        self.client = get_tavily_client()

    @staticmethod
    def _compact_results(
        results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Compact the search results to only include relevant information.

        Args:
            results (list[dict]): The search results.

        Returns:
            list[dict]: A list of compacted search results.
        """
        compacted = []

        for result in results:
            item = {
                key: result.get(key) 
                for key in (
                    "title",
                    "url",
                    "score",
                    "content",
                    "published_date",
                )
                if key in result
            }

            # Avoid sending excessively long content to the LLM.
            content = item.get("content")
            if isinstance(content, str):
                item["content"] = content[:CHARACTER_LIMIT]

            compacted.append(item)

        return compacted

    def web_search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Perform a web search using the Tavily API."""
        try:
            search_results = self.client.search(query, max_results=max_results)
            return self._compact_results(search_results.get("results", []))

        except (ConnectionError, TimeoutError) as e:
            raise ResearchError(f"Error during web search: {e}")

    def search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Run complementary searches and return ranked source records."""
        raw_results: list[dict[str, Any]] = []
        for optimized_query in _search_queries(query.strip()):
            raw_results.extend(self.web_search(optimized_query, max_results))
        return filter_sources(raw_results)[: max_results * 3]

    def extract_page_content(self, urls: list[str]) -> dict[str, Any]:
        """Extract content from a web page using the Tavily API."""
        try:
            page_contents = self.client.extract(urls[:20])  # Limit to first 20 URLs (recommended by Tavily API)
            
            successful_results = []

            for page in page_contents.get("results", []):
                url = page.get("url")
                content = page.get("raw_content")

                if not url or not content:
                    continue  # Skip if URL or content is missing

                successful_results.append({
                    "url": url,
                    "raw_content": content[:PAGE_CONTENT_LIMIT],
                })

            return {
                "results": successful_results,
                "failed_results": page_contents.get("failed_results", []),
            }

        except (ConnectionError, TimeoutError) as e:
            raise ResearchError(f"Error during page content extraction: {e}")


    def map_website(self, url: str) -> list[dict]:
        """Map websites using the Tavily API."""
        try:
            mapped_websites = self.client.map(url)
            return mapped_websites.get("results", [])

        except (ConnectionError, TimeoutError) as e:
            raise ResearchError(f"Error during website mapping: {e}")

        except Exception as e:  # noqa: BLE001
            raise ResearchError(f"Unexpected error during website mapping: {e}")


    def crawl_website(self, url: str, max_depth: int = 2) -> list[dict]:
        """Crawl a website using the Tavily API."""
        try:
            crawled_websites = self.client.crawl(url, max_depth)
            results = []
            for page in crawled_websites.get("results", []):
                url = page.get("url")
                content = page.get("raw_content")

                if not url or not content:
                    continue  # Skip if URL or content is missing

                if isinstance(content, str):
                    content = content[:PAGE_CONTENT_LIMIT]

                results.append({
                    "url": url,
                    "raw_content": content,
                })

            return results

        except (ConnectionError, TimeoutError) as e:
            raise ResearchError(f"Error during website crawling: {e}")
