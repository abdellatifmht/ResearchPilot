from typing import Any

from researchpilot.utils import get_tavily_client

__all__ = ["SourceContentTools"]


class SourceContentTools:
    """Retrieve source content for downstream analysis."""

    def __init__(self) -> None:
        self.client = get_tavily_client()

    def fetch_pages(self, urls: list[str]) -> list[dict[str, Any]]:
        """Fetch available content for up to 20 URLs without local truncation."""
        valid_urls = list(dict.fromkeys(url.strip() for url in urls if url.strip()))
        if not valid_urls:
            raise ValueError("At least one non-empty URL is required.")

        response = self.client.extract(valid_urls[:20])
        pages: list[dict[str, Any]] = []
        for page in response.get("results", []):
            url = page.get("url")
            content = page.get("raw_content")
            if isinstance(url, str) and url and isinstance(content, str) and content:
                pages.append({"url": url, "content": content})

        return pages
