"""Declarative catalogue of the public travel resources used for the knowledge base.

Content is fetched through the MediaWiki API (Wikivoyage / Wikipedia). Both are
published under CC BY-SA 4.0, which allows redistribution with attribution --
so the extracted documents can legitimately be committed to this repository
together with their source title and URL.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KbSource:
    """One document to ingest."""

    slug: str
    title: str
    wiki: str  # MediaWiki host, e.g. "en.wikivoyage.org"
    page: str  # MediaWiki page title
    topics: tuple[str, ...]
    license: str = "CC BY-SA 4.0"

    @property
    def url(self) -> str:
        return f"https://{self.wiki}/wiki/{self.page.replace(' ', '_')}"

    @property
    def api_url(self) -> str:
        return f"https://{self.wiki}/w/api.php"

    @property
    def publisher(self) -> str:
        return "Wikivoyage" if "wikivoyage" in self.wiki else "Wikipedia"


SINGAPORE_SOURCES: tuple[KbSource, ...] = (
    KbSource(
        slug="wikivoyage-singapore",
        title="Wikivoyage: Singapore travel guide",
        wiki="en.wikivoyage.org",
        page="Singapore",
        topics=(
            "overview",
            "districts",
            "attractions",
            "transport",
            "food",
            "practicalities",
            "itineraries",
        ),
    ),
    KbSource(
        slug="wikivoyage-singapore-riverside",
        title="Wikivoyage: Singapore/Riverside",
        wiki="en.wikivoyage.org",
        page="Singapore/Riverside",
        topics=("neighbourhood", "attractions", "museums", "nightlife"),
    ),
    KbSource(
        slug="wikivoyage-singapore-chinatown",
        title="Wikivoyage: Singapore/Chinatown",
        wiki="en.wikivoyage.org",
        page="Singapore/Chinatown",
        topics=("neighbourhood", "culture", "temples", "food"),
    ),
    KbSource(
        slug="wikivoyage-singapore-little-india",
        title="Wikivoyage: Singapore/Little India",
        wiki="en.wikivoyage.org",
        page="Singapore/Little India",
        topics=("neighbourhood", "culture", "temples", "food", "shopping"),
    ),
    KbSource(
        slug="wikivoyage-singapore-bugis",
        title="Wikivoyage: Singapore/Bugis and Kampong Glam",
        wiki="en.wikivoyage.org",
        page="Singapore/Bugis",
        topics=("neighbourhood", "culture", "heritage", "food"),
    ),
    KbSource(
        slug="wikivoyage-singapore-marina-bay",
        title="Wikivoyage: Singapore/Marina Bay",
        wiki="en.wikivoyage.org",
        page="Singapore/Marina Bay",
        topics=("neighbourhood", "attractions", "indoor", "gardens", "views"),
    ),
    KbSource(
        slug="wikivoyage-singapore-orchard",
        title="Wikivoyage: Singapore/Orchard",
        wiki="en.wikivoyage.org",
        page="Singapore/Orchard",
        topics=("neighbourhood", "shopping", "indoor", "dining"),
    ),
    KbSource(
        slug="wikivoyage-singapore-sentosa",
        title="Wikivoyage: Singapore/Sentosa and Harbourfront",
        wiki="en.wikivoyage.org",
        page="Singapore/Sentosa and Harbourfront",
        topics=("neighbourhood", "family", "beaches", "theme-parks", "attractions"),
    ),
    KbSource(
        slug="wikivoyage-singapore-east-coast",
        title="Wikivoyage: Singapore/East Coast",
        wiki="en.wikivoyage.org",
        page="Singapore/East Coast",
        topics=("neighbourhood", "outdoor", "food", "beaches"),
    ),
    KbSource(
        slug="wikipedia-tourism-in-singapore",
        title="Wikipedia: Tourism in Singapore",
        wiki="en.wikipedia.org",
        page="Tourism in Singapore",
        topics=("attractions", "visitor-statistics", "practicalities"),
    ),
    KbSource(
        slug="wikipedia-mrt-singapore",
        title="Wikipedia: Mass Rapid Transit (Singapore)",
        wiki="en.wikipedia.org",
        page="Mass Rapid Transit (Singapore)",
        topics=("transport", "mrt", "practicalities"),
    ),
    KbSource(
        slug="wikipedia-singaporean-cuisine",
        title="Wikipedia: Singaporean cuisine",
        wiki="en.wikipedia.org",
        page="Singaporean cuisine",
        topics=("food", "hawker", "culture"),
    ),
)
