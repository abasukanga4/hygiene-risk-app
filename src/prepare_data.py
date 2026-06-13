"""
Build the model training table from the Food Standards Agency open API.

Self-contained (doesn't depend on the sibling analysis project): pulls
establishments for a curated set of local authorities, keeps the ones with a
numeric 0-5 FHRS rating, and engineers the features the model uses.

Run:
    python src/prepare_data.py
Output:
    data/processed/model_table.parquet
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

API = "https://api.ratings.food.gov.uk"
HEADERS = {"x-api-version": "2", "Accept": "application/json"}
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "model_table.parquet"
SNAPSHOT = pd.Timestamp("2026-06-13")

# local_authority -> (region, settlement_type). Curated for regional + urban/rural
# spread. FHRS only (England/Wales/NI); Scotland's FHIS scheme is excluded.
AUTHORITIES = {
    "Westminster": ("London", "Urban"),
    "Tower Hamlets": ("London", "Urban"),
    "Newham": ("London", "Urban"),
    "Manchester": ("North West", "Urban"),
    "Liverpool": ("North West", "Urban"),
    "Leeds": ("Yorkshire & Humber", "Urban"),
    "Sheffield": ("Yorkshire & Humber", "Urban"),
    "Birmingham": ("West Midlands", "Urban"),
    "Bristol": ("South West", "Urban"),
    "Nottingham City": ("East Midlands", "Urban"),
    "Cambridge City": ("East of England", "Urban"),
    "Cornwall": ("South West", "Rural"),
    "Herefordshire": ("West Midlands", "Rural"),
    "Shropshire": ("West Midlands", "Rural"),
    "North Norfolk": ("East of England", "Rural"),
    "Cardiff": ("Wales", "Urban"),
    "Swansea": ("Wales", "Urban"),
    "Belfast City": ("Northern Ireland", "Urban"),
}
NUMERIC = {"0", "1", "2", "3", "4", "5"}


def authority_ids() -> dict[str, int]:
    rows = requests.get(f"{API}/Authorities/basic", headers=HEADERS, timeout=60).json()["authorities"]
    by_name = {r["Name"]: int(r["LocalAuthorityId"]) for r in rows}
    ids = {}
    for target in AUTHORITIES:
        match = next((n for n in by_name if target.lower() in n.lower()), None)
        if match:
            ids[target] = by_name[match]
    return ids


def fetch(authority_id: int) -> list[dict]:
    out, page = [], 1
    while True:
        url = f"{API}/Establishments?localAuthorityId={authority_id}&pageNumber={page}&pageSize=5000"
        payload = requests.get(url, headers=HEADERS, timeout=60).json()
        out.extend(payload["establishments"])
        if page >= payload["meta"]["totalPages"]:
            return out
        page += 1
        time.sleep(0.3)


def main() -> None:
    ids = authority_ids()
    frames = []
    for name, aid in ids.items():
        df = pd.json_normalize(fetch(aid))
        df = df[df["RatingValue"].isin(NUMERIC)].copy()
        df["region"], df["settlement_type"] = AUTHORITIES[name]
        frames.append(df)
        print(f"  {name:<22} {len(df):>6} rated")

    raw = pd.concat(frames, ignore_index=True)
    raw["rating_date"] = pd.to_datetime(raw["RatingDate"], errors="coerce")

    table = pd.DataFrame({
        "business_type": raw["BusinessType"],
        "region": raw["region"],
        "settlement_type": raw["settlement_type"],
        "years_since_rating": (SNAPSHOT - raw["rating_date"]).dt.days / 365.25,
        "needs_improvement": (raw["RatingValue"].astype(int) <= 2).astype(int),
    }).dropna(subset=["years_since_rating"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUT, index=False)
    print(f"\nSaved {len(table):,} rows -> {OUT}")
    print(f"Positive (needs improvement): {table['needs_improvement'].mean():.2%}")


if __name__ == "__main__":
    main()
