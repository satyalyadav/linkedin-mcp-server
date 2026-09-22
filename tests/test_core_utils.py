"""Tests for core utility functions (rate-limit detection, scrolling, modals)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin_mcp_server.core.exceptions import RateLimitError
from linkedin_mcp_server.core.utils import detect_rate_limit, scroll_to_bottom


@pytest.fixture
def mock_page():
    """Create a mock Patchright page for rate-limit tests."""
    page = MagicMock()
    page.url = "https://www.linkedin.com/in/testuser/details/experience/"

    mock_locator = MagicMock()
    mock_locator.count = AsyncMock(return_value=0)
    mock_locator.inner_text = AsyncMock(return_value="")
    page.locator = MagicMock(return_value=mock_locator)
    return page


class TestDetectRateLimit:
    async def test_checkpoint_url_raises(self, mock_page):
        mock_page.url = "https://www.linkedin.com/checkpoint/challenge/123"
        with pytest.raises(RateLimitError, match="security checkpoint"):
            await detect_rate_limit(mock_page)

    async def test_authwall_url_raises(self, mock_page):
        mock_page.url = "https://www.linkedin.com/authwall?trk=login"
        with pytest.raises(RateLimitError, match="security checkpoint"):
            await detect_rate_limit(mock_page)

    async def test_normal_page_with_main_skips_body_heuristic(self, mock_page):
        """A normal page with <main> should NOT trigger body text checks."""
        main_locator = MagicMock()
        main_locator.count = AsyncMock(return_value=1)

        body_locator = MagicMock()
        # Body contains a phrase that would false-positive
        body_locator.inner_text = AsyncMock(
            return_value="Helping SaaS teams slow down churn with data-driven retention"
        )

        def locator_side_effect(selector):
            if selector == "main":
                return main_locator
            if selector == "body":
                return body_locator
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        # Should NOT raise — the page has <main>, so body heuristic is skipped
        await detect_rate_limit(mock_page)

    async def test_error_page_without_main_triggers_heuristic(self, mock_page):
        """A short error page without <main> with rate-limit text should raise."""
        main_locator = MagicMock()
        main_locator.count = AsyncMock(return_value=0)

        body_locator = MagicMock()
        body_locator.inner_text = AsyncMock(
            return_value="Too many requests. Slow down."
        )

        def locator_side_effect(selector):
            if selector == "main":
                return main_locator
            if selector == "body":
                return body_locator
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        with pytest.raises(RateLimitError, match="Rate limit message"):
            await detect_rate_limit(mock_page)

    async def test_long_body_without_main_does_not_trigger(self, mock_page):
        """A page without <main> but with long body text (>2000 chars) is not an error page."""
        main_locator = MagicMock()
        main_locator.count = AsyncMock(return_value=0)

        body_locator = MagicMock()
        # Long body with a matching phrase buried in content
        body_locator.inner_text = AsyncMock(
            return_value="x" * 2000 + " try again later"
        )

        def locator_side_effect(selector):
            if selector == "main":
                return main_locator
            if selector == "body":
                return body_locator
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        # Should NOT raise — body is too long to be an error page
        await detect_rate_limit(mock_page)

    async def test_normal_url_no_error_passes(self, mock_page):
        """A clean normal page passes all checks without raising."""
        main_locator = MagicMock()
        main_locator.count = AsyncMock(return_value=1)

        def locator_side_effect(selector):
            if selector == "main":
                return main_locator
            return MagicMock(count=AsyncMock(return_value=0))

        mock_page.locator = MagicMock(side_effect=locator_side_effect)
        await detect_rate_limit(mock_page)


class TestScrollToBottom:
    async def test_observer_can_stop_before_first_scroll(self, mock_page):
        observer = AsyncMock(return_value="time_window_boundary")

        result = await scroll_to_bottom(
            mock_page,
            pause_time=0,
            max_scrolls=100,
            observation_callback=observer,
        )

        assert result["attempts"] == 0
        assert result["stop_reason"] == "time_window_boundary"
        mock_page.evaluate.assert_not_called()

    async def test_observer_can_stop_after_new_content(self, mock_page):
        first = {
            "scrollTop": 0,
            "scrollHeight": 2500,
            "clientHeight": 800,
            "contentLength": 1000,
            "contentTail": "new",
        }
        second = {**first, "scrollTop": 600, "contentLength": 2000}
        mock_page.evaluate = AsyncMock(side_effect=[first, None, second])
        observer = AsyncMock(side_effect=[None, "time_window_boundary"])

        result = await scroll_to_bottom(
            mock_page,
            pause_time=0,
            max_scrolls=100,
            observation_callback=observer,
        )

        assert result["attempts"] == 1
        assert result["stop_reason"] == "time_window_boundary"

    async def test_scrolls_internal_linkedin_workspace_container(self, mock_page):
        mock_page.evaluate = AsyncMock(
            side_effect=[
                {"scrollTop": 0, "scrollHeight": 2500, "clientHeight": 800},
                None,
                {"scrollTop": 600, "scrollHeight": 3200, "clientHeight": 800},
                {"scrollTop": 600, "scrollHeight": 3200, "clientHeight": 800},
                None,
                {"scrollTop": 1200, "scrollHeight": 4200, "clientHeight": 800},
            ]
        )

        result = await scroll_to_bottom(mock_page, pause_time=0, max_scrolls=2)

        assert mock_page.evaluate.await_count == 6
        scripts = "\n".join(call.args[0] for call in mock_page.evaluate.await_args_list)
        assert "main#workspace" in scripts
        assert "* 1.5" in scripts
        assert "target.scrollBy(0, step)" in scripts
        assert result["attempts"] == 2
        assert result["stop_reason"] == "max_scrolls"

    async def test_reports_stable_bottom(self, mock_page):
        stable = {
            "scrollTop": 1700,
            "scrollHeight": 2500,
            "clientHeight": 800,
            "contentLength": 5000,
            "contentTail": "last card",
        }
        mock_page.evaluate = AsyncMock(
            side_effect=[stable, None, stable, stable, None, stable]
        )

        result = await scroll_to_bottom(
            mock_page,
            pause_time=0,
            max_scrolls=10,
            stale_limit=2,
        )

        assert result["attempts"] == 2
        assert result["stop_reason"] == "stable_bottom"
        assert result["stable_bottom_reached"] is True
        assert result["explicit_end_marker_seen"] is False
        assert result["end_reached"] is False

    async def test_delayed_bottom_retry_observes_late_content(self, mock_page):
        stable = {
            "scrollTop": 1700,
            "scrollHeight": 2500,
            "clientHeight": 800,
            "contentLength": 5000,
            "contentTail": "last card",
        }
        grown = {
            "scrollTop": 1700,
            "scrollHeight": 3500,
            "clientHeight": 800,
            "contentLength": 6500,
            "contentTail": "new card",
        }
        later = {
            "scrollTop": 2600,
            "scrollHeight": 4200,
            "clientHeight": 800,
            "contentLength": 7200,
            "contentTail": "later card",
        }
        mock_page.evaluate = AsyncMock(
            side_effect=[
                stable,
                None,
                stable,
                stable,
                None,
                stable,
                grown,
                grown,
                None,
                later,
            ]
        )

        result = await scroll_to_bottom(
            mock_page,
            pause_time=0,
            max_scrolls=3,
            stale_limit=2,
            bottom_retry_limit=1,
            bottom_retry_pause=0,
        )

        assert result["attempts"] == 3
        assert result["bottom_retries"] == 1
        assert result["stop_reason"] == "max_scrolls"
        assert result["stable_bottom_reached"] is False
