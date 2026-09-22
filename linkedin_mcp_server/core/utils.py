"""Utility functions for scraping operations."""

import asyncio
from collections.abc import Awaitable, Callable
import logging

from patchright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from .exceptions import RateLimitError

logger = logging.getLogger(__name__)


async def detect_rate_limit(page: Page) -> None:
    """Detect if LinkedIn has rate-limited or security-challenged the session.

    Checks (in order):
    1. URL contains /checkpoint or /authwall (security challenge)
    2. Body text contains rate-limit phrases on error-shaped pages (throttling)

    The body-text heuristic only runs on pages without a ``<main>`` element
    and with short body text (<2000 chars), since real rate-limit pages are
    minimal error pages.  This avoids false positives from profile content
    that happens to contain phrases like "slow down" or "try again later".

    Raises:
        RateLimitError: If any rate-limiting or security challenge is detected
    """
    # Check URL for security challenges
    current_url = page.url
    if "linkedin.com/checkpoint" in current_url or "authwall" in current_url:
        raise RateLimitError(
            "LinkedIn security checkpoint detected. "
            "You may need to verify your identity or wait before continuing.",
            suggested_wait_time=30,
        )

    # Check for rate limit messages — only on error-shaped pages.
    # Real rate-limit pages have no <main> element and short body text.
    # Normal LinkedIn pages (profiles, jobs) have <main> and long content
    # that may incidentally contain phrases like "slow down".
    try:
        has_main = await page.locator("main").count() > 0
        if has_main:
            return  # Normal page with content, skip body text heuristic

        body_text = await page.locator("body").inner_text(timeout=1000)
        if body_text and len(body_text) < 2000:
            body_lower = body_text.lower()
            if any(
                phrase in body_lower
                for phrase in [
                    "too many requests",
                    "rate limit",
                    "slow down",
                    "try again later",
                ]
            ):
                raise RateLimitError(
                    "Rate limit message detected on page.",
                    suggested_wait_time=30,
                )
    except RateLimitError:
        raise
    except PlaywrightTimeoutError:
        pass


async def scroll_to_bottom(
    page: Page,
    pause_time: float = 1.0,
    max_scrolls: int = 10,
    stale_limit: int = 2,
    bottom_retry_limit: int = 0,
    bottom_retry_pause: float | None = None,
    observation_callback: Callable[[], Awaitable[str | None]] | None = None,
) -> dict[str, object]:
    """Scroll the primary content container to trigger lazy loading.

    LinkedIn sometimes makes the document body fixed-height and puts results
    in an internal scroll container such as ``main#workspace``. Scroll the
    largest visible scrollable container instead of assuming the body scrolls.

    Args:
        page: Patchright page object
        pause_time: Time to pause between scrolls (seconds)
        max_scrolls: Maximum number of scroll attempts
        stale_limit: Consecutive unchanged attempts before stopping
        bottom_retry_limit: Extra delayed probes before accepting a stable bottom
        bottom_retry_pause: Base delay for stable-bottom probes. Later probes
            use a linear backoff.
        observation_callback: Optional hook used to snapshot lazy-loaded
            content before it can be virtualized out of the DOM. Returning a
            non-empty string stops scrolling and records that value as the
            stop reason.

    Returns:
        Compact telemetry describing how scrolling stopped. Callers that need
        deep-search coverage can distinguish an exhausted scroll budget from a
        page that reached a stable bottom.
    """
    scroll_state_js = """
        () => {
            const candidates = [];
            const add = element => {
                if (!element || candidates.includes(element)) return;
                candidates.push(element);
            };

            add(document.scrollingElement);
            add(document.documentElement);
            add(document.body);
            add(document.querySelector('main#workspace'));
            for (const element of document.querySelectorAll('main, [role="main"], section, div')) {
                add(element);
            }

            const scrollable = candidates
                .filter(element => {
                    const clientHeight = element.clientHeight || 0;
                    const scrollHeight = element.scrollHeight || 0;
                    if (clientHeight < 100 || scrollHeight <= clientHeight + 20) {
                        return false;
                    }
                    const style = window.getComputedStyle(element);
                    if (style.display === 'none' || style.visibility === 'hidden') {
                        return false;
                    }
                    return true;
                })
                .map(element => ({
                    element,
                    score: (element.scrollHeight - element.clientHeight) * element.clientHeight,
                }))
                .sort((left, right) => right.score - left.score);

            const target = scrollable[0]?.element || document.scrollingElement || document.documentElement;
            return {
                scrollTop: target.scrollTop || window.scrollY || 0,
                scrollHeight: target.scrollHeight || document.documentElement.scrollHeight || 0,
                clientHeight: target.clientHeight || window.innerHeight || 0,
                contentLength: (document.querySelector('main')?.innerText || '').length,
                contentTail: (document.querySelector('main')?.innerText || '').slice(-500),
            };
        }
    """
    scroll_step_js = """
        () => {
            const candidates = [];
            const add = element => {
                if (!element || candidates.includes(element)) return;
                candidates.push(element);
            };

            add(document.scrollingElement);
            add(document.documentElement);
            add(document.body);
            add(document.querySelector('main#workspace'));
            for (const element of document.querySelectorAll('main, [role="main"], section, div')) {
                add(element);
            }

            const scrollable = candidates
                .filter(element => {
                    const clientHeight = element.clientHeight || 0;
                    const scrollHeight = element.scrollHeight || 0;
                    if (clientHeight < 100 || scrollHeight <= clientHeight + 20) {
                        return false;
                    }
                    const style = window.getComputedStyle(element);
                    if (style.display === 'none' || style.visibility === 'hidden') {
                        return false;
                    }
                    return true;
                })
                .map(element => ({
                    element,
                    score: (element.scrollHeight - element.clientHeight) * element.clientHeight,
                }))
                .sort((left, right) => right.score - left.score);

            const target = scrollable[0]?.element || document.scrollingElement || document.documentElement;
            const step = Math.max(Math.floor((target.clientHeight || window.innerHeight) * 1.5), 1000);
            target.scrollBy(0, step);
        }
    """

    unchanged_count = 0
    attempts = 0
    bottom_retries = 0
    stop_reason = "max_scrolls"
    final_state: dict[str, object] = {}
    if observation_callback is not None:
        observation_stop = await observation_callback()
        if observation_stop:
            stop_reason = observation_stop
            return {
                "attempts": attempts,
                "max_scrolls": max_scrolls,
                "stale_limit": stale_limit,
                "bottom_retries": bottom_retries,
                "stop_reason": stop_reason,
                "stable_bottom_reached": False,
                "explicit_end_marker_seen": False,
                "end_reached": False,
                "final_scroll_top": 0,
                "final_scroll_height": 0,
                "final_client_height": 0,
            }

    for i in range(max_scrolls):
        previous_state = await page.evaluate(scroll_state_js)
        await page.evaluate(scroll_step_js)
        await asyncio.sleep(pause_time)

        new_state = await page.evaluate(scroll_state_js)
        if observation_callback is not None:
            observation_stop = await observation_callback()
        attempts = i + 1
        final_state = new_state
        if observation_callback is not None and observation_stop:
            stop_reason = observation_stop
            logger.debug(
                "Scrolling stopped after %d attempts: %s",
                attempts,
                stop_reason,
            )
            break
        previous_top = previous_state.get("scrollTop", 0)
        previous_height = previous_state.get("scrollHeight", 0)
        previous_content_length = previous_state.get("contentLength")
        previous_content_tail = previous_state.get("contentTail")
        new_top = new_state.get("scrollTop", 0)
        new_height = new_state.get("scrollHeight", 0)
        new_content_length = new_state.get("contentLength")
        new_content_tail = new_state.get("contentTail")
        if (
            new_top == previous_top
            and new_height == previous_height
            and new_content_length == previous_content_length
            and new_content_tail == previous_content_tail
        ):
            unchanged_count += 1
        else:
            unchanged_count = 0

        if unchanged_count >= stale_limit:
            new_client_height = new_state.get("clientHeight", 0)
            at_bottom = (
                isinstance(new_top, int | float)
                and isinstance(new_height, int | float)
                and isinstance(new_client_height, int | float)
                and new_top + new_client_height >= new_height - 20
            )
            if at_bottom and bottom_retries < bottom_retry_limit:
                bottom_retries += 1
                retry_pause = bottom_retry_pause
                if retry_pause is None:
                    retry_pause = max(pause_time * 2, 1.0)
                await asyncio.sleep(retry_pause * bottom_retries)
                retry_state = await page.evaluate(scroll_state_js)
                if observation_callback is not None:
                    observation_stop = await observation_callback()
                final_state = retry_state
                if observation_callback is not None and observation_stop:
                    stop_reason = observation_stop
                    break
                changed_during_retry = any(
                    retry_state.get(key) != new_state.get(key)
                    for key in (
                        "scrollTop",
                        "scrollHeight",
                        "contentLength",
                        "contentTail",
                    )
                )
                unchanged_count = 0 if changed_during_retry else stale_limit - 1
                continue

            stop_reason = "stable_bottom" if at_bottom else "stalled"
            logger.debug(
                "Scrolling stopped after %d attempts: %s",
                i + 1,
                stop_reason,
            )
            break

    return {
        "attempts": attempts,
        "max_scrolls": max_scrolls,
        "stale_limit": stale_limit,
        "bottom_retries": bottom_retries,
        "stop_reason": stop_reason,
        "stable_bottom_reached": stop_reason == "stable_bottom",
        "explicit_end_marker_seen": False,
        "end_reached": False,
        "final_scroll_top": final_state.get("scrollTop", 0),
        "final_scroll_height": final_state.get("scrollHeight", 0),
        "final_client_height": final_state.get("clientHeight", 0),
    }


async def scroll_job_sidebar(
    page: Page, pause_time: float = 1.0, max_scrolls: int = 10
) -> None:
    """Scroll the job search sidebar to load all job cards.

    LinkedIn renders job search results in a scrollable sidebar container,
    not the main page body. This function finds that container by locating
    a job card link and walking up to its scrollable ancestor, then scrolls
    it iteratively until no new content loads.

    Args:
        page: Patchright page object
        pause_time: Time to pause between scrolls (seconds)
        max_scrolls: Maximum number of scroll attempts
    """
    # Wait for at least one job card link to render before scrolling
    try:
        await page.wait_for_selector('a[href*="/jobs/view/"]', timeout=5000)
    except PlaywrightTimeoutError:
        logger.debug("No job card links found, skipping sidebar scroll")
        return

    scrolled = await page.evaluate(
        """async ({pauseTime, maxScrolls}) => {
            const link = document.querySelector('a[href*="/jobs/view/"]');
            if (!link) return -2;

            let container = link.parentElement;
            while (container && container !== document.body) {
                const style = window.getComputedStyle(container);
                const overflowY = style.overflowY;
                if ((overflowY === 'auto' || overflowY === 'scroll')
                    && container.scrollHeight > container.clientHeight) {
                    break;
                }
                container = container.parentElement;
            }

            if (!container || container === document.body) {
                return -1;
            }

            let scrollCount = 0;
            for (let i = 0; i < maxScrolls; i++) {
                const prevHeight = container.scrollHeight;
                container.scrollTop = container.scrollHeight;
                await new Promise(r => setTimeout(r, pauseTime * 1000));
                if (container.scrollHeight === prevHeight) break;
                scrollCount++;
            }
            return scrollCount;
        }""",
        {"pauseTime": pause_time, "maxScrolls": max_scrolls},
    )
    if scrolled == -2:
        logger.debug("Job card link disappeared before evaluate, skipping scroll")
    elif scrolled == -1:
        logger.debug("No scrollable container found for job sidebar")
    elif scrolled:
        logger.debug("Scrolled job sidebar %d times", scrolled)
    else:
        logger.debug("Job sidebar container found but no new content loaded")


async def handle_modal_close(page: Page) -> bool:
    """Close any popup modals that might be blocking content.

    Returns:
        True if a modal was closed, False otherwise
    """
    try:
        close_button = page.locator(
            'button[aria-label="Dismiss"], '
            'button[aria-label="Close"], '
            "button.artdeco-modal__dismiss"
        ).first

        if await close_button.is_visible(timeout=1000):
            await close_button.click()
            await asyncio.sleep(0.5)
            logger.debug("Closed modal")
            return True
    except PlaywrightTimeoutError:
        pass
    except Exception as e:
        logger.debug("Error closing modal: %s", e)

    return False
