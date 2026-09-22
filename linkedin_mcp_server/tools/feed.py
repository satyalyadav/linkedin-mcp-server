"""
LinkedIn feed scraping tool.

Fetches posts from the authenticated user's LinkedIn home feed using
innerText extraction. Scrolls until the requested number of post
permalinks have been observed in SDUI pagination responses — a
locale-independent progress signal, since the feed DOM exposes no
stable per-post container selector.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.tools import ToolResult
from pydantic import Field

from linkedin_mcp_server.config.schema import DEFAULT_TOOL_TIMEOUT_SECONDS
from linkedin_mcp_server.core.exceptions import AuthenticationError
from linkedin_mcp_server.dependencies import get_ready_extractor, handle_auth_error
from linkedin_mcp_server.error_handler import raise_tool_error
from linkedin_mcp_server.scraping.extractor import _RATE_LIMITED_MSG
from linkedin_mcp_server.scraping.link_metadata import Reference

logger = logging.getLogger(__name__)


def _post_search_summary(result: dict[str, Any], keywords: str = "search") -> str:
    """Build the small human-readable tool message shown by MCP clients."""
    coverage = result.get("coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    cards = int(coverage.get("result_card_count") or 0)
    canonical = int(coverage.get("canonical_url_count") or 0)
    paired = int(coverage.get("paired_post_count") or 0)
    scrolls = int(coverage.get("scrolls_attempted") or 0)
    status = str(coverage.get("status") or "unknown")
    oldest = coverage.get("oldest_relative_time")
    if not oldest and isinstance(coverage.get("oldest_age_hours"), int | float):
        oldest = f"{coverage['oldest_age_hours']:g}h"

    lines = [
        f"Post search complete: {cards} cards, {canonical} canonical URLs, "
        f"{paired} card/URL pairs.",
        f"Coverage: {status}; {scrolls} scroll attempts"
        + (f"; oldest observed result {oldest}." if oldest else "."),
        "Structured post records and coverage are available to the assistant "
        "without repeating the raw LinkedIn page text.",
    ]
    if result.get("section_errors"):
        lines.append(
            "The search also returned a section error; inspect structured data."
        )
    payload_path = _write_search_posts_payload(keywords, result)
    if payload_path:
        lines.append(f"Full structured payload saved to: {payload_path}")
    return "\n".join(lines)


def _write_search_posts_payload(keywords: str, result: dict[str, Any]) -> str | None:
    """Persist the full search_posts result for bridges that drop structured data.

    Some MCP clients only forward the text content and discard
    ``structured_content``. Writing the payload to /tmp keeps the hiring
    workflow working there: the text reply carries the file path and the
    agent ingests that file. ``structured_content`` itself is unchanged.
    """
    try:
        slug = re.sub(r"[^A-Za-z0-9_-]+", "-", keywords.strip().lower()).strip("-")
        slug = slug[:40] or "search"
        path = Path(f"/tmp/linkedin-search-posts-{int(time.time() * 1000)}-{slug}.json")
        path.write_text(json.dumps(result, default=str))
        return str(path)
    except Exception:
        logger.exception("Failed to persist search_posts payload for '%s'", keywords)
        return None


def register_feed_tools(
    mcp: FastMCP, *, tool_timeout: float = DEFAULT_TOOL_TIMEOUT_SECONDS
) -> None:
    """Register feed-related tools with the MCP server."""

    @mcp.tool(
        timeout=tool_timeout,
        title="Get Feed",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"feed", "scraping"},
        exclude_args=["extractor"],
    )
    async def get_feed(
        ctx: Context,
        num_posts: Annotated[int, Field(ge=1, le=50)] = 10,
        extractor: Any | None = None,
    ) -> dict[str, Any]:
        """
        Get posts from the authenticated user's LinkedIn feed.

        Args:
            ctx: FastMCP context for progress reporting
            num_posts: Number of posts to fetch (1-50, default 10).
                       Posts are loaded in batches of ~5 as the page scrolls,
                       so the actual count may slightly exceed the target.

        Returns:
            Dict with url, sections (name -> raw text), and optional keys:
            - references["feed"]: list of {kind: "feed_post", url, ...}
              entries. URLs are relative paths and may carry either
              ``/feed/update/<urn>/`` (DOM-anchor-derived) or
              ``/posts/<slug>`` (SDUI-derived) shape — both are valid
              LinkedIn permalinks.
            - section_errors: present when the feed is rate-limited or
              extraction fails.

            Truncated posts are not auto-expanded; full text for any post
            is reachable via its permalink in references["feed"]. The LLM
            should parse sections["feed"] for post bodies.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="get_feed"
            )
            logger.info("Scraping feed (num_posts=%d)", num_posts)

            await ctx.report_progress(
                progress=0, total=100, message="Starting feed scrape"
            )

            extracted = await extractor.extract_feed(num_posts=num_posts)

            url = "https://www.linkedin.com/feed/"
            sections: dict[str, str] = {}
            references: dict[str, list[Reference]] = {}
            section_errors: dict[str, dict[str, Any]] = {}
            if extracted.text and extracted.text != _RATE_LIMITED_MSG:
                sections["feed"] = extracted.text
                if extracted.references:
                    references["feed"] = extracted.references
            elif extracted.text == _RATE_LIMITED_MSG:
                section_errors["feed"] = {
                    "error_type": "rate_limit",
                    "error_message": extracted.text,
                }
            elif extracted.error:
                section_errors["feed"] = extracted.error

            await ctx.report_progress(progress=100, total=100, message="Complete")

            result: dict[str, Any] = {"url": url, "sections": sections}
            if references:
                result["references"] = references
            if section_errors:
                result["section_errors"] = section_errors
            return result

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "get_feed")
        except Exception as e:
            raise_tool_error(e, "get_feed")

    @mcp.tool(
        timeout=tool_timeout,
        title="Search Posts",
        annotations={"readOnlyHint": True, "openWorldHint": True},
        tags={"feed", "search", "scraping"},
        exclude_args=["extractor"],
    )
    async def search_posts(
        keywords: str,
        ctx: Context,
        date_posted: str | None = None,
        sort_by: str | None = None,
        max_pages: Annotated[int, Field(ge=1, le=100)] = 3,
        include_raw: bool = False,
        extractor: Any | None = None,
    ) -> ToolResult:
        """
        Search LinkedIn posts/content by keyword or Boolean query.

        Args:
            keywords: Search keywords or LinkedIn-supported Boolean query.
                Example: '"software engineer" AND (intern OR internship) AND hiring'
            ctx: FastMCP context for progress reporting
            date_posted: Optional recency filter. Known values:
                "past_24_hours", "past_week", "past_month".
                LinkedIn content search does not expose an exact last-3-days
                filter; use "past_week" and post-filter the relative dates.
            sort_by: Optional sort order. Known values:
                "date", "date_posted", "relevance". Default is "date_posted".
            max_pages: Maximum result batches/scroll attempts to load (1-100,
                default 3). Broad past-24-hour searches can have many noisy
                early matches. Inspect ``coverage`` and increase or shard the
                query when the oldest result does not reach the time boundary.
            include_raw: Include raw page text and reference arrays for
                diagnostics. Defaults to False because structured posts,
                canonical URLs, and coverage contain the useful search data.

        Returns:
            A concise readable summary plus compact structured data. The
            structured payload contains ``url``, ``post_urls``, ``posts``, and
            ``coverage``. Set ``include_raw=True`` only for connector
            diagnostics that require ``sections`` and ``references``.
            ``post_urls`` contains full canonical LinkedIn permalinks.
            ``posts`` contains structured full-text cards; when LinkedIn
            exposes a permalink or activity URN inside a rendered card, the
            URL is captured from that same card subtree.
            Otherwise, unique actor names beside payload permalinks anchor
            order-preserving card/URL segments; ambiguous gaps remain unpaired.
            ``coverage`` distinguishes a delayed-retry stable browser bottom
            from an explicit end-of-results marker, reports oldest-result age
            and URL pairing gaps, and records whether the requested time-window
            boundary was observed. A stable bottom alone is incomplete
            evidence, and the tool never claims LinkedIn indexing is exhaustive.
        """
        try:
            extractor = extractor or await get_ready_extractor(
                ctx, tool_name="search_posts"
            )
            logger.info(
                "Searching posts: keywords='%s', date_posted='%s', sort_by='%s', max_pages=%d",
                keywords,
                date_posted,
                sort_by,
                max_pages,
            )

            await ctx.report_progress(
                progress=0, total=100, message="Starting post search"
            )

            result = await extractor.search_posts(
                keywords,
                date_posted=date_posted,
                sort_by=sort_by,
                max_pages=max_pages,
                include_raw=include_raw,
            )

            await ctx.report_progress(progress=100, total=100, message="Complete")

            return ToolResult(
                content=_post_search_summary(result, keywords),
                structured_content=result,
            )

        except AuthenticationError as e:
            try:
                await handle_auth_error(e, ctx)
            except Exception as relogin_exc:
                raise_tool_error(relogin_exc, "search_posts")
        except Exception as e:
            raise_tool_error(e, "search_posts")
