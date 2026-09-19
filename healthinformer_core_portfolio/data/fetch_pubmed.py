"""Fetch PubMed abstracts via E-utilities (esearch + efetch).

Usage:
    from data.fetch_pubmed import search_pubmed, fetch_abstracts

    pmids = search_pubmed("type 2 diabetes management", max_results=200)
    articles = fetch_abstracts(pmids)
"""

import time
import xml.etree.ElementTree as ET
from typing import Any

import requests

from config.settings import (
    PUBMED_BASE_URL,
    PUBMED_BATCH_SIZE,
    PUBMED_EMAIL,
    PUBMED_MAX_RESULTS_PER_QUERY,
    PUBMED_RATE_LIMIT_DELAY,
)


def search_pubmed(query: str, max_results: int = PUBMED_MAX_RESULTS_PER_QUERY) -> list[str]:
    """Search PubMed and return a list of PMIDs for the given query.

    Uses the esearch endpoint. Results are sorted by relevance (default).
    """
    url = f"{PUBMED_BASE_URL}/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": PUBMED_EMAIL,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    pmids: list[str] = data.get("esearchresult", {}).get("idlist", [])
    print(f"  Found {len(pmids)} PMIDs for: {query!r}")
    return pmids


def _parse_article(article_elem: ET.Element) -> dict[str, Any] | None:
    """Parse a single PubmedArticle XML element into a dict.

    Returns None if the article has no abstract text.
    """
    medline = article_elem.find("MedlineCitation")
    if medline is None:
        return None

    pmid_elem = medline.find("PMID")
    pmid = pmid_elem.text if pmid_elem is not None else ""

    article = medline.find("Article")
    if article is None:
        return None

    # Title
    title_elem = article.find("ArticleTitle")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

    # Abstract — may have multiple AbstractText sections (structured abstract)
    abstract_elem = article.find("Abstract")
    if abstract_elem is None:
        return None
    abstract_parts: list[str] = []
    for text_elem in abstract_elem.findall("AbstractText"):
        label = text_elem.get("Label", "")
        # Capture all text including child elements (e.g. <i>, <b>, <sub>)
        full_text = "".join(text_elem.itertext()).strip()
        if not full_text:
            continue
        if label:
            abstract_parts.append(f"{label}: {full_text}")
        else:
            abstract_parts.append(full_text)
    abstract = "\n".join(abstract_parts)
    if not abstract.strip():
        return None

    # Journal
    journal_elem = article.find("Journal/Title")
    journal = journal_elem.text if journal_elem is not None else ""

    # Year
    year = ""
    pub_date = article.find("Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year_elem = pub_date.find("Year")
        if year_elem is not None:
            year = year_elem.text or ""
        else:
            medline_date = pub_date.find("MedlineDate")
            if medline_date is not None and medline_date.text:
                year = medline_date.text[:4]

    # Authors
    authors: list[str] = []
    author_list = article.find("AuthorList")
    if author_list is not None:
        for author in author_list.findall("Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None and last.text:
                name = last.text
                if fore is not None and fore.text:
                    name += f" {fore.text}"
                authors.append(name)

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "authors": authors,
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def fetch_abstracts(pmids: list[str]) -> list[dict[str, Any]]:
    """Fetch full article metadata for a list of PMIDs using efetch.

    Processes PMIDs in batches to respect API limits. Articles without
    abstracts are silently skipped.
    """
    articles: list[dict[str, Any]] = []
    total = len(pmids)

    for i in range(0, total, PUBMED_BATCH_SIZE):
        batch = pmids[i : i + PUBMED_BATCH_SIZE]
        print(f"  Fetching batch {i // PUBMED_BATCH_SIZE + 1} "
              f"({len(batch)} IDs, {i + len(batch)}/{total} total)...")

        url = f"{PUBMED_BASE_URL}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
            "email": PUBMED_EMAIL,
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"  WARNING: Batch fetch failed: {e}. Skipping batch.")
            time.sleep(PUBMED_RATE_LIMIT_DELAY)
            continue

        root = ET.fromstring(response.content)
        for article_elem in root.findall("PubmedArticle"):
            parsed = _parse_article(article_elem)
            if parsed is not None:
                articles.append(parsed)

        time.sleep(PUBMED_RATE_LIMIT_DELAY)

    return articles


if __name__ == "__main__":
    test_query = "type 2 diabetes management"
    print(f"Testing PubMed fetch for: {test_query!r}")
    ids = search_pubmed(test_query, max_results=5)
    if ids:
        results = fetch_abstracts(ids)
        print(f"Fetched {len(results)} articles with abstracts")
        for art in results[:2]:
            print(f"  [{art['pmid']}] {art['title'][:80]}...")
    else:
        print("No PMIDs returned. Check PUBMED_EMAIL in .env")
