"""Archive PRC 2026 public documentation, team directory, and leaderboard.

This tool makes public GET requests only; it neither reads .env nor submits work.
Use --offline to rebuild derived files, or --refresh to replace cached responses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/competition"
SOURCES = DOCS / "sources"
BASE = "https://prc-data-challenge-2026.netlify.app/"
COMPETITION_ID = "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c"
API = f"https://datacomp.opensky-network.org/api/competitions/{COMPETITION_ID}"
SITE_PAGES = [
    "index.html", "rationale.html", "data.html", "eligibility.html", "ranking.html",
    "teams/index.html", "teams/vigorous-whistle.html",
]
REFERENCES = {
    "minio_instructions_2024.html": "https://ansperformance.eu/study/data-challenge/dc2024/data.html",
    "leaderboard_app.html": "https://prc-challenge-2026.vercel.app/",
    "github_search_2026.json": "https://api.github.com/search/repositories?q=prc+data+challenge+2026&per_page=50",
    "github_2025_repos.json": "https://api.github.com/orgs/prc-data-challenge-2025/repos?per_page=100",
    "github_2024_repos.json": "https://api.github.com/orgs/prc-data-challenge-2024/repos?per_page=100",
    "official_site_tree.json": "https://api.github.com/repos/euctrl-pru/prc_data_challenge_website_2026/git/trees/main?recursive=1",
    "taxiout_methodology.html": "https://ansperformance.eu/efficiency/taxiout/",
    "eurocontrol_2026_mirror.html": "https://ansperformance.eu/study/data-challenge/dc2026/",
    "eurocontrol_2026_ranking.html": "https://ansperformance.eu/study/data-challenge/dc2026/ranking.html",
    "eurocontrol_2026_data.html": "https://ansperformance.eu/study/data-challenge/dc2026/data.html",
    "eurocontrol_2026_eligibility.html": "https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html",
    "external_iem_metar.html": "https://mesonet.agron.iastate.edu/request/download.phtml",
    "external_ourairports.html": "https://ourairports.com/data/",
    "external_osm_license.html": "https://www.openstreetmap.org/copyright",
    "paper_8764.html": "https://journals.open.tudelft.nl/joas/article/view/8764",
    "paper_7963.html": "https://journals.open.tudelft.nl/joas/article/view/7963",
    "paper_8750.html": "https://journals.open.tudelft.nl/joas/article/view/8750",
    "paper_8252.html": "https://journals.open.tudelft.nl/joas/article/view/8252",
}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class Archive:
    def __init__(self, offline=False, refresh=False):
        self.offline = offline
        self.refresh = refresh
        self.path = DOCS / "source_manifest.json"
        old = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else []
        self.records = {row["local_path"]: row for row in old}

    def fetch(self, url, relative_path):
        path = SOURCES / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path_key = path.relative_to(ROOT).as_posix()
        record = dict(self.records.get(path_key, {}))
        if not path.exists() or (self.refresh and not self.offline):
            if self.offline:
                raise RuntimeError(f"Missing cached response: {relative_path}")
            request = urllib.request.Request(url, headers={"User-Agent": "PRC-2026-vigorous-whistle-research/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                payload = response.read()
                record.update({
                    "url": url,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "final_url": response.url,
                    "http_status": response.status,
                    "content_type": response.headers.get("Content-Type"),
                    "last_modified": response.headers.get("Last-Modified"),
                })
            path.write_bytes(payload)
        else:
            payload = path.read_bytes()
            if not record:
                record["retrieved_at_utc"] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
                record["timestamp_basis"] = "local snapshot file modification time"
        record.update({
            "local_path": path_key, "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        record.setdefault("url", url)
        self.records[path_key] = record
        write_json(self.path, sorted(self.records.values(), key=lambda row: row["local_path"]))
        return payload

    def json(self, url, path):
        return json.loads(self.fetch(url, path))


def markdown(node, base_url):
    if isinstance(node, NavigableString):
        return re.sub(r"\s+", " ", str(node))
    if not isinstance(node, Tag) or node.name in {"script", "style", "nav"}:
        return ""
    inner = lambda: "".join(markdown(child, base_url) for child in node.children)
    if node.name == "a":
        label = inner().strip()
        url = urllib.parse.urljoin(base_url, node.get("href", ""))
        return f"[{label}]({url})" if label and node.get("href") else label
    if node.name in {"strong", "b"}:
        return f"**{inner().strip()}**"
    if node.name in {"em", "i"}:
        return f"*{inner().strip()}*" if node.get_text(strip=True) else ""
    if node.name == "pre":
        return "\n\n```\n" + node.get_text().strip() + "\n```\n\n"
    if node.name == "code":
        return "`" + node.get_text().strip() + "`"
    if node.name == "br":
        return "\n"
    if node.name == "img":
        return f"[{node.get('alt') or 'Image'}]({urllib.parse.urljoin(base_url, node.get('src', ''))})"
    if re.fullmatch(r"h[1-6]", node.name):
        return f"\n\n{'#' * int(node.name[1])} {inner().strip()}\n\n"
    if node.name == "table":
        rows = []
        for row in node.find_all("tr"):
            cells = [re.sub(r"\s+", " ", markdown(cell, base_url)).strip().replace("|", "\\|")
                     for cell in row.find_all(["th", "td"], recursive=False)]
            if cells:
                rows.append("| " + " | ".join(cells) + " |")
        if rows:
            count = len(node.find("tr").find_all(["th", "td"], recursive=False))
            rows.insert(1, "| " + " | ".join(["---"] * count) + " |")
        return "\n\n" + "\n".join(rows) + "\n\n"
    if node.name == "li":
        return "\n- " + inner().strip() + "\n"
    if node.name in {"p", "section", "header", "div", "ul", "ol", "blockquote"}:
        return "\n\n" + inner().strip() + "\n\n"
    return inner()


def extract_page(payload, url, target):
    soup = BeautifulSoup(payload, "html.parser")
    main = soup.find("main") or soup.find("article") or soup.body or soup
    content = re.sub(r"\n[ \t]+", "\n", markdown(main, url))
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"Source: {url}\n\n{content}\n", encoding="utf-8")
    return {
        "url": url,
        "title": soup.title.get_text(" ", strip=True) if soup.title else "",
        "main_links": [{"text": a.get_text(" ", strip=True), "url": urllib.parse.urljoin(url, a["href"])}
                       for a in main.find_all("a", href=True)],
        "internal_pages": sorted({urllib.parse.urljoin(url, a["href"]).split("#")[0]
                                  for a in soup.find_all("a", href=True)
                                  if urllib.parse.urljoin(url, a["href"]).startswith(BASE)}),
    }


def pull_site(archive):
    pages = []
    for name in SITE_PAGES:
        url = BASE + name
        payload = archive.fetch(url, "site/" + name)
        pages.append(extract_page(payload, url, DOCS / "pages" / Path(name).with_suffix(".md")))
    archive.fetch(BASE + "search.json", "site/search.json")
    archive.fetch(BASE + "media/minio_access_keys.png", "site/minio_access_keys.png")
    archive.fetch("https://datacomp.opensky-network.org/api/swagger-ui/", "api/swagger.html")
    archive.fetch("https://datacomp.opensky-network.org/api/openapi?format=json", "api/openapi.json")
    write_json(DOCS / "page_links.json", pages)
    print(f"Archived {len(pages)} official pages and the API documentation.", flush=True)


def pull_leaderboard(archive):
    rows = {}
    cursor = None
    cursors = set()
    page_number = 1
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        url = API + "/leaderboard?" + urllib.parse.urlencode(params)
        page = archive.json(url, f"api/leaderboard_200_page_{page_number:03d}.json")
        for row in page.get("items", []):
            rows[row["submissionId"]] = row
        cursor = page.get("nextCursor") or page.get("next")
        if not cursor:
            break
        if cursor in cursors:
            raise RuntimeError("Leaderboard pagination returned a repeated cursor.")
        cursors.add(cursor)
        page_number += 1
        if page_number > 500:
            raise RuntimeError("Leaderboard pagination exceeded 500 pages.")
    ordered = sorted(rows.values(), key=lambda row: (row["score"], row["processedAt"], row["submissionId"]))
    best = {}
    for row in ordered:
        best.setdefault(row["teamId"], row)
    standings = [{"rank": rank, **row} for rank, row in enumerate(best.values(), 1)]
    retrieval_times = [archive.records[f"docs/competition/sources/api/leaderboard_200_page_{number:03d}.json"]["retrieved_at_utc"]
                       for number in range(1, page_number + 1)]
    write_json(DOCS / "leaderboard_snapshot.json", {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_retrieval_started_at_utc": min(retrieval_times),
        "source_retrieval_finished_at_utc": max(retrieval_times),
        "source": API + "/leaderboard", "pages": page_number,
        "successful_submissions": len(ordered), "teams_with_scores": len(standings),
        "used_pairs_values": sorted({row["usedPairs"] for row in ordered}),
        "team_standings": standings, "submissions": ordered,
        "note": "Public API pages are fetched sequentially; this is not an atomic database snapshot.",
    })
    with (DOCS / "leaderboard_teams.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(standings[0]) if standings else ["rank", "teamName", "score"])
        writer.writeheader()
        writer.writerows(standings)
    print(f"Leaderboard: {len(ordered)} successful submissions; {len(standings)} scored teams; {page_number} pages.", flush=True)
    for row in standings[:10]:
        print(f"  {row['rank']:2d} {row['teamName']:<26} {row['score']:.4f} seconds", flush=True)


def pull_teams(archive):
    teams = []
    page_number = 1
    while True:
        url = API + f"/teams?page={page_number}&size=100&participants=false"
        page = archive.json(url, f"api/teams_page_{page_number:03d}.json")
        teams.extend(page["content"])
        if page_number >= page["totalPages"]:
            break
        page_number += 1
    write_json(DOCS / "teams_snapshot.json", {
        "built_at_utc": datetime.now(timezone.utc).isoformat(), "source": API + "/teams",
        "total_teams_reported": page["totalTeams"], "teams": teams,
    })
    print(f"Archived {len(teams)} approved team records (participant details not requested).", flush=True)


def pull_references(archive):
    for name, url in REFERENCES.items():
        payload = archive.fetch(url, "references/" + name)
        if name.endswith(".html"):
            extract_page(payload, url, DOCS / "pages/references" / Path(name).with_suffix(".md"))
    # Public source for the five substantive official website pages.
    tree = archive.json(REFERENCES["official_site_tree.json"], "references/official_site_tree.json")
    revision = tree["sha"]
    for name in ["index.qmd", "data.qmd", "eligibility.qmd", "ranking.qmd", "rationale.qmd", "README.md", "media/references.bib"]:
        url = f"https://raw.githubusercontent.com/euctrl-pru/prc_data_challenge_website_2026/{revision}/{name}"
        archive.fetch(url, "official_source/" + name)
    repositories = archive.json(REFERENCES["github_search_2026.json"], "references/github_search_2026.json")
    summary = []
    for repo in repositories.get("items", []):
        name, branch = repo["full_name"], repo["default_branch"]
        record = {key: repo.get(key) for key in ["full_name", "html_url", "description", "default_branch", "license", "pushed_at"]}
        url = f"https://raw.githubusercontent.com/{name}/{branch}/README.md"
        try:
            path = "repositories/" + name.replace("/", "__") + "__README.md"
            archive.fetch(url, path)
            record["readme_path"] = "sources/" + path
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            # GitHub's README endpoint resolves Readme.md / readme.md casing.
            endpoint = f"https://api.github.com/repos/{name}/readme"
            try:
                response = archive.json(endpoint, "repositories/" + name.replace("/", "__") + "__readme_response.json")
                archive.fetch(response["download_url"], path)
                record["readme_path"] = "sources/" + path
                record["upstream_readme_name"] = response["path"]
            except urllib.error.HTTPError as fallback:
                record["readme_http_error"] = fallback.code
        summary.append(record)
    write_json(DOCS / "public_repositories.json", summary)
    print(f"Archived references and README files for {len(summary)} matching public repositories.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--section", choices=["all", "site", "leaderboard", "teams", "references"], default="all")
    args = parser.parse_args()
    archive = Archive(args.offline, args.refresh)
    operations = {"site": pull_site, "leaderboard": pull_leaderboard, "teams": pull_teams, "references": pull_references}
    for name, operation in operations.items():
        if args.section in {"all", name}:
            operation(archive)


if __name__ == "__main__":
    main()
