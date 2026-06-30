from __future__ import annotations

import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
DOCS_DATA = ROOT / "docs" / "data"
for folder in [RAW, PROCESSED, DOCS_DATA]:
    folder.mkdir(parents=True, exist_ok=True)

STATE_FIPS_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL","13":"GA",
    "15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA",
    "26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY",
    "37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX",
    "49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY","60":"AS","66":"GU","69":"MP","72":"PR","78":"VI"
}

US_50_DC_STATE_ABBRS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY"
}


def read_config() -> dict[str, Any]:
    with open(ROOT / "scripts" / "config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_url(url: str, timeout: int = 180) -> requests.Response:
    headers = {"User-Agent": "environmental-health-adaptation-gap-dashboard/1.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        re.sub(r"_+", "_", str(c).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_").replace(".", "_")).strip("_")
        for c in df.columns
    ]
    return df


def zfill_fips(value: Any, width: int = 5) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"\D", "", s)
    if not s:
        return None
    return s.zfill(width)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def pct_rank(series: pd.Series, higher_is_worse: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 3:
        return pd.Series(np.nan, index=series.index)
    p = x.rank(pct=True, method="average") * 100
    return p if higher_is_worse else 100 - p


def mean_available(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    return df[available].mean(axis=1, skipna=True)


def fetch_county_centroids(cfg: dict[str, Any]) -> pd.DataFrame:
    try:
        import geopandas as gpd
        year = cfg["years"].get("county_shapes", 2024)
        url = cfg["sources"]["county_shapes_zip"].format(year=year)
        print(f"Downloading county shapes: {url}")
        zip_path = RAW / f"county_shapes_{year}.zip"
        zip_path.write_bytes(get_url(url).content)
        gdf = gpd.read_file(zip_path).to_crs("EPSG:4326")
        points = gdf.geometry.representative_point()
        out = pd.DataFrame({
            "county_fips": gdf["GEOID"].astype(str).str.zfill(5),
            "county_name_shape": gdf["NAME"].astype(str),
            "state_fips": gdf["STATEFP"].astype(str).str.zfill(2),
            "lon": points.x,
            "lat": points.y,
        })
        out["state"] = out["state_fips"].map(STATE_FIPS_TO_ABBR).fillna(out["state_fips"])
        print(f"Loaded county_centroids: {len(out):,} rows")
        return out
    except Exception as e:
        print(f"WARNING county_centroids failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["county_fips", "lat", "lon"])


def fetch_acs(cfg: dict[str, Any]) -> pd.DataFrame:
    """ACS variables for adaptive capacity.

    This function is deliberately defensive because the Census API can return an
    HTML/error response or reject one variable while accepting the rest. It tries
    the configured ACS year first, then falls back to earlier recent ACS 5-year
    releases.
    """
    configured_year = int(cfg["years"].get("acs", 2024))
    candidate_years = []
    for y in [configured_year, configured_year - 1, configured_year - 2, 2023, 2022]:
        if y not in candidate_years and y >= 2020:
            candidate_years.append(y)

    profile_var_groups = [
        ["NAME", "DP05_0001E", "DP03_0128PE", "DP03_0099PE", "DP04_0046PE", "DP05_0024PE"],
        ["NAME", "DP05_0001E", "DP03_0128PE", "DP04_0046PE", "DP05_0024PE"],
        ["NAME", "DP05_0001E"],
    ]

    for year in candidate_years:
        profile_base = cfg["sources"]["acs_profile_api"].format(year=year)
        detail_base = cfg["sources"].get("acs_detail_api", "https://api.census.gov/data/{year}/acs/acs5").format(year=year)

        for profile_vars in profile_var_groups:
            try:
                print(f"Downloading ACS profile {year}: {profile_base}")
                params = {"get": ",".join(profile_vars), "for": "county:*"}
                census_key = os.getenv("CENSUS_API_KEY")
                if census_key:
                    params["key"] = census_key
                res = requests.get(profile_base, params=params, timeout=180)
                res.raise_for_status()
                data = res.json()
                if not isinstance(data, list) or len(data) < 2:
                    raise RuntimeError("ACS profile returned no rows")

                prof = clean_columns(pd.DataFrame(data[1:], columns=data[0]))
                prof["county_fips"] = prof["state"].astype(str).str.zfill(2) + prof["county"].astype(str).str.zfill(3)
                rename = {
                    "name": "acs_name",
                    "dp05_0001e": "population",
                    "dp03_0128pe": "poverty_rate",
                    "dp03_0099pe": "uninsured_rate",
                    "dp04_0046pe": "renter_share",
                    "dp05_0024pe": "age65plus_share",
                }
                keep = ["county_fips"] + [c for c in rename if c in prof.columns]
                out = prof[keep].rename(columns=rename)

                # Optional detail table variables: no vehicle and no internet.
                try:
                    detail_vars = ["NAME", "B08201_002E", "B08201_001E", "B28002_013E", "B28002_001E"]
                    print(f"Downloading ACS detail {year}: {detail_base}")
                    params2 = {"get": ",".join(detail_vars), "for": "county:*"}
                    census_key = os.getenv("CENSUS_API_KEY")
                    if census_key:
                        params2["key"] = census_key
                    res2 = requests.get(detail_base, params=params2, timeout=180)
                    res2.raise_for_status()
                    data2 = res2.json()
                    det = clean_columns(pd.DataFrame(data2[1:], columns=data2[0]))
                    det["county_fips"] = det["state"].astype(str).str.zfill(2) + det["county"].astype(str).str.zfill(3)
                    for c in ["b08201_002e", "b08201_001e", "b28002_013e", "b28002_001e"]:
                        if c in det.columns:
                            det[c] = pd.to_numeric(det[c], errors="coerce")
                    det["no_vehicle_share"] = np.where(det.get("b08201_001e", np.nan) > 0, det.get("b08201_002e", np.nan) / det.get("b08201_001e", np.nan) * 100, np.nan)
                    det["no_internet_share"] = np.where(det.get("b28002_001e", np.nan) > 0, det.get("b28002_013e", np.nan) / det.get("b28002_001e", np.nan) * 100, np.nan)
                    out = out.merge(det[["county_fips", "no_vehicle_share", "no_internet_share"]], on="county_fips", how="left")
                except Exception as e:
                    print(f"WARNING ACS detail failed for {year}; continuing with profile variables only: {type(e).__name__}: {e}")

                for c in out.columns:
                    if c not in ["county_fips", "acs_name"]:
                        out[c] = pd.to_numeric(out[c], errors="coerce")
                out["acs_year_used"] = year
                print(f"Loaded acs: {len(out):,} rows from ACS {year}")
                return out

            except Exception as e:
                print(f"WARNING ACS profile failed for {year} with variables {profile_vars}: {type(e).__name__}: {e}")
                continue

    print("WARNING acs failed: all ACS fallback attempts failed")
    return pd.DataFrame(columns=["county_fips"])


def fetch_places(cfg: dict[str, Any]) -> pd.DataFrame:
    """Download CDC PLACES county health measures.

    The CDC PLACES GIS-friendly file is usually WIDE, with columns such as
    CASTHMA_CrudePrev, COPD_CrudePrev, CHD_CrudePrev, PHLTH_CrudePrev,
    MHLTH_CrudePrev, and DEPRESSION_CrudePrev. Some older/open-data tables are
    LONG, with measureid + data_value. This function supports both formats.
    """
    try:
        url = cfg["sources"]["places_county_csv"]
        print(f"Downloading CDC PLACES: {url}")
        df = clean_columns(pd.read_csv(io.BytesIO(get_url(url).content), low_memory=False))

        fips_col = find_col(df, ["locationid", "location_id", "countyfips", "county_fips", "geolocationid"])
        if fips_col is None:
            print("WARNING places: no county FIPS column found")
            return pd.DataFrame(columns=["county_fips"])

        df["county_fips"] = df[fips_col].apply(zfill_fips)
        name_col = find_col(df, ["locationname", "countyname", "county_name", "name"])
        state_col = find_col(df, ["stateabbr", "state_abbr", "state"])

        # Start with one row per county.
        out = df[["county_fips"]].dropna().drop_duplicates().copy()

        # -------- WIDE GIS-friendly format, e.g. CASTHMA_CrudePrev --------
        wide_candidates = {
            "asthma_prev": ["casthma_crudeprev", "asthma_crudeprev", "current_asthma_crudeprev"],
            "copd_prev": ["copd_crudeprev"],
            "chd_prev": ["chd_crudeprev"],
            "poor_physical_health_prev": ["phlth_crudeprev"],
            "poor_mental_health_prev": ["mhlth_crudeprev"],
            "depression_prev": ["depression_crudeprev"],
            "uninsured_places_prev": ["access2_crudeprev"],
        }

        wide_found = False
        for new_name, candidates in wide_candidates.items():
            col = find_col(df, candidates)
            if col:
                wide_found = True
                small = df[["county_fips", col]].dropna(subset=["county_fips"]).drop_duplicates("county_fips")
                small = small.rename(columns={col: new_name})
                small[new_name] = pd.to_numeric(small[new_name], errors="coerce")
                out = out.merge(small, on="county_fips", how="left")

        # -------- LONG format, e.g. measureid + data_value --------
        # If the wide columns were not present, try long-format parsing.
        if not wide_found:
            measure_col = find_col(df, ["measureid", "measure_id"])
            value_col = find_col(df, ["data_value", "datavalue", "value"])
            type_col = find_col(df, ["data_value_type", "data_value_typeid"])
            wanted = {
                "CASTHMA": "asthma_prev",
                "COPD": "copd_prev",
                "CHD": "chd_prev",
                "PHLTH": "poor_physical_health_prev",
                "MHLTH": "poor_mental_health_prev",
                "DEPRESSION": "depression_prev",
                "ACCESS2": "uninsured_places_prev",
            }
            if measure_col and value_col:
                temp = df.copy()
                if type_col:
                    crude = temp[type_col].astype(str).str.contains("crude", case=False, na=False)
                    if crude.any():
                        temp = temp[crude]
                temp[measure_col] = temp[measure_col].astype(str).str.upper()
                temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")
                temp = temp[temp[measure_col].isin(wanted.keys())].copy()
                temp["measure_name_clean"] = temp[measure_col].map(wanted)
                long_out = temp.pivot_table(
                    index="county_fips",
                    columns="measure_name_clean",
                    values=value_col,
                    aggfunc="mean",
                ).reset_index()
                long_out.columns.name = None
                out = out[["county_fips"]].merge(long_out, on="county_fips", how="left")
            else:
                print("WARNING places: neither wide health columns nor long measure columns were found")

        if name_col:
            names = df[["county_fips", name_col]].dropna().drop_duplicates("county_fips").rename(columns={name_col: "county_name"})
            out = out.merge(names, on="county_fips", how="left")

        if state_col:
            states = df[["county_fips", state_col]].dropna().drop_duplicates("county_fips").rename(columns={state_col: "state"})
            out = out.merge(states, on="county_fips", how="left")

        health_cols = ["asthma_prev", "copd_prev", "chd_prev", "poor_physical_health_prev", "poor_mental_health_prev", "depression_prev"]
        found_health = [c for c in health_cols if c in out.columns and pd.to_numeric(out[c], errors="coerce").notna().any()]
        print(f"Loaded places: {len(out):,} rows; health columns loaded: {found_health}")
        return out

    except Exception as e:
        print(f"WARNING places failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["county_fips"])


def read_zip_csv(url: str, filename: str) -> pd.DataFrame:
    print(f"Downloading {url}")
    content = get_url(url).content
    (RAW / filename).write_bytes(content)
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        csv_files = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csv_files:
            raise RuntimeError(f"No CSV found in {url}")
        with z.open(csv_files[0]) as f:
            return clean_columns(pd.read_csv(f, low_memory=False))


def fetch_airdata(cfg: dict[str, Any]) -> pd.DataFrame:
    try:
        year = cfg["years"].get("airdata", 2024)
        aqi_url = cfg["sources"]["airdata_annual_aqi_by_county"].format(year=year)
        conc_url = cfg["sources"]["airdata_annual_conc_by_monitor"].format(year=year)
        aqi = read_zip_csv(aqi_url, f"annual_aqi_by_county_{year}.zip")
        conc = read_zip_csv(conc_url, f"annual_conc_by_monitor_{year}.zip")

        def make_fips(df: pd.DataFrame) -> pd.Series:
            state_col = find_col(df, ["state_code", "state_fips"])
            county_col = find_col(df, ["county_code", "county_fips"])
            if state_col and county_col:
                return df[state_col].apply(lambda x: zfill_fips(x, 2)).fillna("") + df[county_col].apply(lambda x: zfill_fips(x, 3)).fillna("")
            fips_col = find_col(df, ["county_fips", "fips"])
            if fips_col:
                return df[fips_col].apply(zfill_fips)
            return pd.Series([None] * len(df), index=df.index)

        aqi["county_fips"] = make_fips(aqi)
        aqi_keep = ["county_fips"]
        rename = {}
        possible = {
            "max_aqi": "max_aqi",
            "90th_percentile_aqi": "p90_aqi",
            "median_aqi": "median_aqi",
            "days_with_aqi": "days_with_aqi",
            "unhealthy_days": "unhealthy_days",
            "unhealthy_for_sensitive_groups_days": "usg_days",
            "very_unhealthy_days": "very_unhealthy_days",
            "hazardous_days": "hazardous_days",
        }
        for old, new in possible.items():
            if old in aqi.columns:
                aqi_keep.append(old)
                rename[old] = new
        aqi_out = aqi[aqi_keep].rename(columns=rename)
        for c in aqi_out.columns:
            if c != "county_fips":
                aqi_out[c] = pd.to_numeric(aqi_out[c], errors="coerce")
        aqi_out = aqi_out.groupby("county_fips", as_index=False).mean(numeric_only=True)

        conc["county_fips"] = make_fips(conc)
        param_col = find_col(conc, ["parameter_name", "parameter"])
        mean_col = find_col(conc, ["arithmetic_mean", "mean"])
        if param_col and mean_col:
            conc[mean_col] = pd.to_numeric(conc[mean_col], errors="coerce")
            pm25 = conc[conc[param_col].astype(str).str.contains("PM2.5", case=False, na=False)].groupby("county_fips")[mean_col].mean().reset_index(name="pm25_mean")
            ozone = conc[conc[param_col].astype(str).str.contains("Ozone", case=False, na=False)].groupby("county_fips")[mean_col].mean().reset_index(name="ozone_mean")
            out = aqi_out.merge(pm25, on="county_fips", how="outer").merge(ozone, on="county_fips", how="outer")
        else:
            out = aqi_out
        print(f"Loaded airdata: {len(out):,} rows")
        return out
    except Exception as e:
        print(f"WARNING airdata failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["county_fips"])


def fetch_nri(cfg: dict[str, Any]) -> pd.DataFrame:
    """FEMA National Risk Index Counties from ArcGIS feature service."""
    try:
        item_api = cfg["sources"]["nri_counties_item_api"]
        print(f"Discovering FEMA NRI ArcGIS service: {item_api}")
        item = get_url(item_api).json()
        service_url = item.get("url")
        if not service_url:
            raise RuntimeError("No service URL found for FEMA NRI item")
        # Query layer 0. For FEMA NRI Counties item this is the county layer.
        layer_url = service_url.rstrip("/") + "/0/query"
        print(f"Downloading FEMA NRI counties from {layer_url}")
        rows = []
        offset = 0
        page = 2000
        while True:
            params = {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "false",
                "f": "json",
                "resultRecordCount": page,
                "resultOffset": offset,
            }
            response = requests.get(layer_url, params=params, timeout=180)
            response.raise_for_status()
            data = response.json()
            features = data.get("features", [])
            if not features:
                break
            rows.extend([f.get("attributes", {}) for f in features])
            if len(features) < page:
                break
            offset += page
        if not rows:
            raise RuntimeError("No FEMA NRI features returned")
        df = clean_columns(pd.DataFrame(rows))
        fips_col = find_col(df, ["stcofips", "countyfips", "county_fips", "geoid", "fips"])
        if not fips_col:
            raise RuntimeError(f"No county FIPS field found in FEMA NRI fields: {list(df.columns)[:20]}")
        df["county_fips"] = df[fips_col].apply(zfill_fips)

        # Flexible field matching for FEMA's evolving schema.
        def maybe_num(col: str):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for c in df.columns:
            low = c.lower()
            if any(key in low for key in ["score", "value", "eal", "sovi", "resl", "risk", "exposure"]):
                maybe_num(c)

        out = pd.DataFrame({"county_fips": df["county_fips"]})
        mapping_candidates = {
            "risk_score": ["risk_score", "riskscore", "risk_score", "risk_scr"],
            "expected_annual_loss_score": ["eal_score", "ealscore", "eal_scr", "eal_score"],
            "social_vulnerability_score": ["sovi_score", "soviscore", "sovi_scr", "sovi_score"],
            "community_resilience_score": ["resl_score", "reslscore", "resl_scr", "resilience_score"],
        }
        for out_col, cands in mapping_candidates.items():
            col = find_col(df, cands)
            if col:
                out[out_col] = pd.to_numeric(df[col], errors="coerce")

        # Add hazard-specific risk/score columns when available.
        hazard_prefixes = ["htex", "wfir", "drgt", "cfld", "rfld", "coas", "hrcn", "trnd", "ltng", "swnd", "hail", "hwav", "cwav", "erqk", "tsun", "vlcn", "land"]
        hazard_cols = []
        for c in df.columns:
            low = c.lower()
            if any(low.startswith(p) for p in hazard_prefixes) and any(k in low for k in ["score", "risk", "eal"]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
                if df[c].notna().sum() > 50:
                    hazard_cols.append(c)
        if hazard_cols:
            out["hazard_specific_mean_score"] = df[hazard_cols].mean(axis=1, skipna=True)
        out = out.groupby("county_fips", as_index=False).mean(numeric_only=True)
        print(f"Loaded nri: {len(out):,} rows")
        return out
    except Exception as e:
        print(f"WARNING nri failed: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=["county_fips"])


def build_index(df: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    numeric_candidates = [
        "risk_score", "expected_annual_loss_score", "social_vulnerability_score", "community_resilience_score", "hazard_specific_mean_score",
        "max_aqi", "p90_aqi", "median_aqi", "unhealthy_days", "usg_days", "very_unhealthy_days", "hazardous_days", "pm25_mean", "ozone_mean",
        "asthma_prev", "copd_prev", "chd_prev", "poor_physical_health_prev", "poor_mental_health_prev", "depression_prev",
        "population", "poverty_rate", "uninsured_rate", "uninsured_places_prev", "renter_share", "age65plus_share", "no_vehicle_share", "no_internet_share"
    ]
    for c in numeric_candidates:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c + "_pct"] = pct_rank(out[c])

    # Community resilience: high resilience means less deficit, so invert percentile.
    if "community_resilience_score" in out.columns:
        out["community_resilience_deficit_pct"] = pct_rank(out["community_resilience_score"], higher_is_worse=False)

    out["climate_hazard_burden"] = mean_available(out, [
        "risk_score_pct", "expected_annual_loss_score_pct", "hazard_specific_mean_score_pct"
    ])
    out["air_pollution_exposure"] = mean_available(out, [
        "max_aqi_pct", "p90_aqi_pct", "median_aqi_pct", "pm25_mean_pct", "ozone_mean_pct", "unhealthy_days_pct", "usg_days_pct", "very_unhealthy_days_pct", "hazardous_days_pct"
    ])
    out["health_vulnerability"] = mean_available(out, [
        "asthma_prev_pct", "copd_prev_pct", "chd_prev_pct", "poor_physical_health_prev_pct", "poor_mental_health_prev_pct", "depression_prev_pct"
    ])
    out["adaptive_capacity_deficit"] = mean_available(out, [
        "social_vulnerability_score_pct", "community_resilience_deficit_pct", "poverty_rate_pct", "uninsured_rate_pct", "uninsured_places_prev_pct", "renter_share_pct", "age65plus_share_pct", "no_vehicle_share_pct", "no_internet_share_pct"
    ])

    if not weights:
        weights = {
            "climate_hazard_burden": 0.30,
            "air_pollution_exposure": 0.25,
            "health_vulnerability": 0.25,
            "adaptive_capacity_deficit": 0.20,
        }
    weighted_sum = pd.Series(0.0, index=out.index)
    weight_total = pd.Series(0.0, index=out.index)
    for domain, weight in weights.items():
        if domain in out.columns:
            valid = out[domain].notna()
            weighted_sum.loc[valid] += out.loc[valid, domain] * float(weight)
            weight_total.loc[valid] += float(weight)
    out["ehagi"] = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
    out["ehagi_rank"] = out["ehagi"].rank(ascending=False, method="min")

    def classify(row: pd.Series) -> str:
        hazard = row.get("climate_hazard_burden", np.nan)
        pollution = row.get("air_pollution_exposure", np.nan)
        health = row.get("health_vulnerability", np.nan)
        deficit = row.get("adaptive_capacity_deficit", np.nan)
        if pd.notna(hazard) and pd.notna(health) and pd.notna(deficit) and hazard >= 75 and health >= 75 and deficit >= 75:
            return "Highest adaptation gap"
        if pd.notna(pollution) and pd.notna(health) and pd.notna(deficit) and pollution >= 75 and health >= 75 and deficit >= 70:
            return "Pollution-health adaptation priority"
        if pd.notna(hazard) and pd.notna(deficit) and hazard >= 75 and deficit >= 75:
            return "Climate adaptation priority"
        if pd.notna(health) and pd.notna(deficit) and health >= 75 and deficit >= 75:
            return "Public-health preparedness priority"
        if pd.notna(hazard) and hazard >= 75:
            return "High hazard burden"
        return "Lower combined gap"

    out["priority_group"] = out.apply(classify, axis=1)
    return out


def write_dictionary(path: Path) -> None:
    rows = [
        ("county_fips", "5-digit county FIPS code"),
        ("county_name", "County name"),
        ("state", "State abbreviation"),
        ("lat", "County representative latitude"),
        ("lon", "County representative longitude"),
        ("risk_score", "FEMA National Risk Index composite risk score where available"),
        ("expected_annual_loss_score", "FEMA National Risk Index expected annual loss score where available"),
        ("social_vulnerability_score", "FEMA National Risk Index social vulnerability score where available"),
        ("community_resilience_score", "FEMA National Risk Index community resilience score where available"),
        ("max_aqi", "EPA AirData maximum annual AQI"),
        ("pm25_mean", "Mean annual PM2.5 concentration across county monitors"),
        ("ozone_mean", "Mean annual ozone concentration across county monitors"),
        ("asthma_prev", "CDC PLACES asthma prevalence"),
        ("copd_prev", "CDC PLACES COPD prevalence"),
        ("chd_prev", "CDC PLACES coronary heart disease prevalence"),
        ("poor_physical_health_prev", "CDC PLACES poor physical health prevalence"),
        ("poor_mental_health_prev", "CDC PLACES poor mental health prevalence"),
        ("poverty_rate", "ACS poverty rate"),
        ("uninsured_rate", "ACS uninsured rate"),
        ("no_vehicle_share", "ACS share of households without a vehicle"),
        ("no_internet_share", "ACS share of households without internet subscription"),
        ("climate_hazard_burden", "Percentile-based climate/natural hazard domain score"),
        ("air_pollution_exposure", "Percentile-based air pollution domain score"),
        ("health_vulnerability", "Percentile-based health vulnerability domain score"),
        ("adaptive_capacity_deficit", "Percentile-based adaptive capacity deficit domain score"),
        ("ehagi", "Environmental Health Adaptation Gap Index"),
        ("priority_group", "Policy screening category"),
    ]
    pd.DataFrame(rows, columns=["variable", "description"]).to_csv(path, index=False)


def main() -> None:
    cfg = read_config()
    print("Building Environmental Health Adaptation Gap dashboard data...")
    loaders = [
        ("nri", fetch_nri),
        ("airdata", fetch_airdata),
        ("places", fetch_places),
        ("acs", fetch_acs),
        ("county_centroids", fetch_county_centroids),
    ]
    pieces = []
    warnings = []
    for name, func in loaders:
        try:
            df = func(cfg)
            if df is None or df.empty or "county_fips" not in df.columns:
                warnings.append(f"{name}: no usable rows returned")
                continue
            df["county_fips"] = df["county_fips"].apply(zfill_fips)
            df = df.dropna(subset=["county_fips"]).drop_duplicates(subset=["county_fips"])
            pieces.append((name, df))
            df.to_csv(PROCESSED / f"{name}.csv", index=False)
        except Exception as e:
            message = f"{name}: {type(e).__name__}: {e}"
            print(f"WARNING {message}")
            warnings.append(message)

    if not pieces:
        raise RuntimeError("No data sources loaded. Check source URLs.")
    base = None
    for name, df in pieces:
        if base is None:
            base = df.copy()
        else:
            base = base.merge(df, on="county_fips", how="outer", suffixes=("", f"_{name}"))
    assert base is not None

    if "county_name" not in base.columns:
        base["county_name"] = np.nan
    for possible in ["county_name_shape", "acs_name"]:
        if possible in base.columns:
            base["county_name"] = base["county_name"].fillna(base[possible])

    if "state" not in base.columns:
        base["state"] = np.nan
    if "state_fips" in base.columns:
        base["state"] = base["state"].fillna(base["state_fips"].map(STATE_FIPS_TO_ABBR)).fillna(base["state_fips"])
    base["state"] = base["state"].fillna(base["county_fips"].astype(str).str[:2].map(STATE_FIPS_TO_ABBR))

    # CDC PLACES county health measures are for the 50 states and DC. Keeping the
    # dashboard to 50 states + DC prevents territories with missing health fields
    # from dominating the top adaptation-gap table.
    for required in ["lat", "lon"]:
        if required not in base.columns:
            base[required] = np.nan

    before_filter = len(base)
    base = base[base["state"].isin(US_50_DC_STATE_ABBRS)].copy()

    # Drop legacy/unmatched county records that do not exist in the current Census
    # county boundary file. This removes old Connecticut county records from NRI
    # while keeping the current CT planning regions used by 2024 Census boundaries.
    before_geometry_filter = len(base)
    base = base[base["lat"].notna() & base["lon"].notna()].copy()
    print(
        f"Filtered to 50 states + DC with current county geometry: {len(base):,} rows kept "
        f"out of {before_filter:,}; dropped {before_geometry_filter - len(base):,} unmatched legacy rows"
    )

    out = build_index(base, cfg.get("index_weights", {}))
    priority_cols = [
        "county_fips", "county_name", "state", "lat", "lon", "population",
        "risk_score", "expected_annual_loss_score", "social_vulnerability_score", "community_resilience_score", "hazard_specific_mean_score",
        "max_aqi", "p90_aqi", "median_aqi", "pm25_mean", "ozone_mean", "unhealthy_days", "usg_days", "very_unhealthy_days", "hazardous_days",
        "asthma_prev", "copd_prev", "chd_prev", "poor_physical_health_prev", "poor_mental_health_prev", "depression_prev",
        "poverty_rate", "uninsured_rate", "uninsured_places_prev", "renter_share", "age65plus_share", "no_vehicle_share", "no_internet_share",
        "climate_hazard_burden", "air_pollution_exposure", "health_vulnerability", "adaptive_capacity_deficit", "ehagi", "ehagi_rank", "priority_group"
    ]
    existing = [c for c in priority_cols if c in out.columns]
    remaining = [c for c in out.columns if c not in existing]
    out = out[existing + remaining]
    out = out.sort_values(["ehagi", "climate_hazard_burden"], ascending=[False, False])
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(4)

    county_csv = ROOT / cfg["output"]["county_csv"]
    metadata_json = ROOT / cfg["output"]["metadata_json"]
    dictionary_csv = ROOT / cfg["output"]["dictionary_csv"]
    out.to_csv(county_csv, index=False)
    write_dictionary(dictionary_csv)
    metadata = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "airdata_year": cfg["years"].get("airdata"),
        "acs_year": cfg["years"].get("acs"),
        "places_release": cfg["years"].get("places_release"),
        "nri_release": cfg["years"].get("nri_release"),
        "county_shapes_year": cfg["years"].get("county_shapes"),
        "rows": int(len(out)),
        "sources_loaded": [name for name, _ in pieces],
        "warnings": warnings,
        "interpretation": "EHAGI is a screening index, not a causal health-impact estimate.",
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {county_csv}")
    print(f"Wrote {metadata_json}")
    print(f"Wrote {dictionary_csv}")
    print("Build complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
