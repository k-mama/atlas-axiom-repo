# scripts/main_generator.py
# AtlasAxiom Pipeline Orchestrator (repo-structure aligned)
#
# - Reads collectors in:   scripts/collectors/*.py
# - Reads processors in:   scripts/processors/*.py
# - Writes outputs into:   data/*.json
# - Updates:              data/meta.json  (last_success_utc is used by index.html)
#
# 실행(레포 루트에서):
#   python scripts/main_generator.py --run-type hourly-hot
#   python scripts/main_generator.py --run-type daily-briefing
#
# 주의:
# - 원문(게시글/기사/본문/자막) "저장"은 하지 않음.
# - 저장 파일에는 링크 + 신호필드 + 우리 문장(요약/인사이트)만 들어가게 설계.

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import importlib.util


# -------------------------
# Paths (repo-aligned)
# -------------------------
SCRIPTS_DIR = Path(__file__).resolve().parent               # .../scripts
REPO_ROOT = SCRIPTS_DIR.parent                              # repo root
DATA_DIR = REPO_ROOT / "data"

META_PATH = DATA_DIR / "meta.json"
HOT_PATH = DATA_DIR / "hot_cards.json"
BRIEF_AM_PATH = DATA_DIR / "briefing_am.json"
BRIEF_PM_PATH = DATA_DIR / "briefing_pm.json"
WATCHLIST_META_PATH = DATA_DIR / "watchlist_meta.json"

# Optional config place (you already have it): src/sources.json
SOURCES_JSON_PATH = REPO_ROOT / "src" / "sources.json"


# -------------------------
# Meta defaults
# -------------------------
DEFAULT_META: Dict[str, Any] = {
    "meta_version": "1.0",
    "service": "AtlasAxiom",
    "channel": "us-stocks-ai",
    "primary_tickers": ["NVDA", "MSFT", "GOOGL", "AMD", "TSLA"],

    "success": False,
    "last_attempt_utc": None,
    "last_success_utc": None,

    "run": {
        "run_id": None,
        "run_type": None,
        "environment": None,
        "started_utc": None,
        "finished_utc": None,
        "duration_ms": None
    },

    "outputs": {
        "hot_cards_path": "data/hot_cards.json",
        "briefing_am_path": "data/briefing_am.json",
        "briefing_pm_path": "data/briefing_pm.json",
        "watchlist_meta_path": "data/watchlist_meta.json",
        "meta_path": "data/meta.json"
    },

    "stats": {
        "collected_total": 0,
        "signals_total": 0,
        "cards_total": 0,
        "entities_total": 0,
        "tickers_total": 0
    },

    "sources": [
        {"id": "official_rss", "tier": "A", "enabled": True, "status": "unknown", "items_collected": 0, "last_run_utc": None},
        {"id": "youtube_rss",  "tier": "A", "enabled": True, "status": "unknown", "items_collected": 0, "last_run_utc": None},
        {"id": "reddit_api",   "tier": "B", "enabled": True, "status": "unknown", "items_collected": 0, "last_run_utc": None},
    ],

    "risk_policy": {
        "raw_content_storage": "forbidden",
        "allowed_fields": [
            "source_type", "source_name", "url", "timestamp",
            "tickers", "entities", "keywords_top", "sentiment",
            "intensity", "velocity", "topic_cluster", "confidence"
        ],
        "notes": "Store signals only. Never archive full post/article text, images, or transcripts. For Tier B/C store link + signal fields only."
    },

    "errors": []
}


# -------------------------
# Utilities
# -------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, data: Any) -> bool:
    try:
        safe_mkdir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return True
    except Exception:
        return False


def load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def merge_meta(existing: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(existing, dict):
        return json.loads(json.dumps(DEFAULT_META))  # deep copy
    # shallow merge with defaults; keep unknown keys too
    meta = json.loads(json.dumps(DEFAULT_META))
    meta.update(existing)
    # ensure nested keys exist
    meta.setdefault("run", DEFAULT_META["run"])
    meta.setdefault("stats", DEFAULT_META["stats"])
    meta.setdefault("outputs", DEFAULT_META["outputs"])
    meta.setdefault("sources", DEFAULT_META["sources"])
    meta.setdefault("errors", [])
    meta.setdefault("risk_policy", DEFAULT_META["risk_policy"])
    return meta


def append_error(meta: Dict[str, Any], stage: str, err: Exception, keep_last: int = 50) -> None:
    e = {
        "utc": utc_now_iso(),
        "stage": stage,
        "type": err.__class__.__name__,
        "message": str(err),
    }
    meta.setdefault("errors", [])
    meta["errors"].append(e)
    meta["errors"] = meta["errors"][-keep_last:]


def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec: {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def resolve_callable(mod: Any, candidates: List[str]) -> Optional[Callable]:
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return fn
    return None


def strip_raw_content_fields(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    저장용 구조에서 원문성 필드(본문/설명/전체텍스트 등)는 제거.
    (메모리 내 처리에는 있을 수 있으나, data/*.json에는 남기지 않는 방향)
    """
    banned_keys = {
        "text", "content", "body", "full_text", "selftext",
        "description", "summary", "transcript", "html",
        "title"  # 타이틀도 '콘텐츠'로 볼 수 있어 저장본에서는 제거(엄격 모드)
    }
    return {k: v for k, v in item.items() if k not in banned_keys}


def to_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def unique_upper_tickers(items: List[Dict[str, Any]]) -> List[str]:
    s = set()
    for it in items:
        for t in to_list(it.get("tickers")):
            if isinstance(t, str) and t.strip():
                s.add(t.strip().upper())
    return sorted(s)


def unique_entities(items: List[Dict[str, Any]]) -> List[str]:
    s = set()
    for it in items:
        for e in to_list(it.get("entities")):
            if isinstance(e, str) and e.strip():
                s.add(e.strip())
    return sorted(s)


def compute_stats(raw_items: List[Dict[str, Any]], cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    # "signals_total"은 저장 가능한 필드로 정규화된 개수라고 보고 raw_items와 동일 카운트 사용
    tickers = unique_upper_tickers(raw_items)
    entities = unique_entities(raw_items)
    return {
        "collected_total": len(raw_items),
        "signals_total": len(raw_items),
        "cards_total": len(cards),
        "entities_total": len(entities),
        "tickers_total": len(tickers),
    }


def read_sources_config() -> Optional[Dict[str, Any]]:
    if SOURCES_JSON_PATH.exists():
        data = load_json(SOURCES_JSON_PATH)
        if isinstance(data, dict):
            return data
    return None


# -------------------------
# Collector orchestration
# -------------------------
@dataclass
class SourceRunResult:
    source_id: str
    ok: bool
    items: List[Dict[str, Any]]
    error: Optional[str] = None


def run_collector(source_id: str, file_name: str, fn_candidates: List[str], sources_cfg: Optional[Dict[str, Any]]) -> SourceRunResult:
    """
    - source_id: meta.json sources[].id 와 동일 키
    - file_name: scripts/collectors/ 아래 파일
    """
    try:
        path = SCRIPTS_DIR / "collectors" / file_name
        mod = load_module_from_path(f"collectors.{source_id}", path)
        fn = resolve_callable(mod, fn_candidates)
        if fn is None:
            raise AttributeError(f"No callable found in {file_name}. Tried: {fn_candidates}")

        # 함수 시그니처 다양성 흡수: (sources_cfg) 받을 수도/안 받을 수도
        try:
            if sources_cfg is not None:
                out = fn(sources_cfg)
            else:
                out = fn()
        except TypeError:
            out = fn()

        items = []
        for x in to_list(out):
            if isinstance(x, dict):
                items.append(x)
            else:
                # dict 아닌 경우 최소 래핑
                items.append({"url": str(x), "timestamp": utc_now_iso(), "source_name": source_id, "source_type": "unknown"})
        return SourceRunResult(source_id=source_id, ok=True, items=items)

    except Exception as e:
        return SourceRunResult(source_id=source_id, ok=False, items=[], error=f"{e.__class__.__name__}: {e}")


def update_meta_source(meta: Dict[str, Any], source_id: str, ok: bool, count: int) -> None:
    now = utc_now_iso()
    for s in meta.get("sources", []):
        if s.get("id") == source_id:
            s["status"] = "ok" if ok else "error"
            s["items_collected"] = int(count)
            s["last_run_utc"] = now
            break


# -------------------------
# Processor orchestration
# -------------------------
def run_ai_summarizer(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    processors/ai_summarizer.py 를 가능한 이름 후보로 호출.
    실패 시, 최소 fallback 카드 생성.
    """
    try:
        path = SCRIPTS_DIR / "processors" / "ai_summarizer.py"
        mod = load_module_from_path("processors.ai_summarizer", path)
        fn = resolve_callable(mod, ["summarize_news", "summarize_signals", "build_cards", "generate_cards"])
        if fn is None:
            raise AttributeError("No summarizer function found (summarize_news / summarize_signals / build_cards / generate_cards)")

        out = fn(raw_items)
        cards: List[Dict[str, Any]] = []
        for c in to_list(out):
            if isinstance(c, dict):
                cards.append(c)
        return cards

    except Exception:
        # fallback: 원문 없이(키워드/티커 기반) 안전 카드
        cards = []
        for i, it in enumerate(raw_items[:12], start=1):
            tickers = unique_upper_tickers([it])
            kw = to_list(it.get("keywords_top"))[:5]
            url = it.get("url")
            cards.append({
                "id": f"fallback-{i}",
                "tier": "signal",
                "tickers": tickers,
                "headline": f"Signal detected for {', '.join(tickers) if tickers else 'market'}",
                "why_it_matters": "A new high-signal indicator was detected. Verify source link and watch for follow-through.",
                "keywords": kw,
                "sources": [url] if isinstance(url, str) else [],
                "confidence": float(it.get("confidence", 0.5)) if isinstance(it.get("confidence", 0.5), (int, float)) else 0.5
            })
        return cards


def run_risk_checker(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    processors/risk_checker.py 를 가능한 이름 후보로 호출.
    실패 시 pass-through.
    """
    try:
        path = SCRIPTS_DIR / "processors" / "risk_checker.py"
        mod = load_module_from_path("processors.risk_checker", path)
        fn = resolve_callable(mod, ["run_risk_check", "risk_check", "filter_cards", "sanitize_cards"])
        if fn is None:
            raise AttributeError("No risk checker function found (run_risk_check / risk_check / filter_cards / sanitize_cards)")
        out = fn(cards)
        safe_cards: List[Dict[str, Any]] = []
        for c in to_list(out):
            if isinstance(c, dict):
                safe_cards.append(c)
        return safe_cards
    except Exception:
        return cards


# -------------------------
# Output builders (simple, safe)
# -------------------------
def build_hot_cards_payload(cards: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "updated_at_utc": utc_now_iso(),
        "channel": meta.get("channel", "us-stocks-ai"),
        "cards": cards
    }


def build_briefing_payload(cards: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    """
    AM/PM 브리핑은 (핫카드 상위 몇개) + 체크포인트 형태로 가볍게 생성
    """
    top = cards[:7]
    checkpoints = [
        "Check official filings / press releases for confirmation",
        "Watch pre-market / after-hours reaction and volume",
        "Track follow-through keywords velocity in the next cycle"
    ]
    return {
        "updated_at_utc": utc_now_iso(),
        "label": label,
        "top_cards": top,
        "checkpoints": checkpoints
    }


def build_watchlist_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "updated_at_utc": utc_now_iso(),
        "primary_tickers": meta.get("primary_tickers", ["NVDA", "MSFT", "GOOGL", "AMD", "TSLA"]),
        "notes": "Placeholder. Extend with per-ticker thresholds, alert rules, and source mappings."
    }


# -------------------------
# Main
# -------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-type", default="manual", choices=["manual", "hourly-hot", "daily-briefing", "weekly-deep"])
    parser.add_argument("--env", default=os.getenv("ATLAS_ENV", "local"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started_ts = time.time()
    started_utc = utc_now_iso()

    safe_mkdir(DATA_DIR)

    # load meta (or create)
    meta = merge_meta(load_json(META_PATH))

    # run id
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{random.randint(1000,9999)}"

    # meta: start
    meta["last_attempt_utc"] = started_utc
    meta["success"] = False
    meta["run"] = meta.get("run", {})
    meta["run"]["run_id"] = run_id
    meta["run"]["run_type"] = args.run_type
    meta["run"]["environment"] = args.env
    meta["run"]["started_utc"] = started_utc
    meta["run"]["finished_utc"] = None
    meta["run"]["duration_ms"] = None

    all_errors: List[Tuple[str, str]] = []  # (stage, message)

    sources_cfg = read_sources_config()

    # --- 1) Collect ---
    collected: List[Dict[str, Any]] = []
    collector_specs = [
        ("official_rss", "official_rss.py", ["fetch_official_news", "fetch_official_rss", "fetch_items"]),
        ("youtube_rss",  "youtube_rss.py",  ["fetch_youtube_videos", "fetch_youtube_rss", "fetch_items"]),
        ("reddit_api",   "reddit_api.py",   ["fetch_reddit_buzz", "fetch_reddit", "fetch_items"]),
    ]

    for source_id, file_name, fn_candidates in collector_specs:
        res = run_collector(source_id, file_name, fn_candidates, sources_cfg)
        update_meta_source(meta, source_id, res.ok, len(res.items))
        if res.ok:
            collected.extend(res.items)
        else:
            all_errors.append((f"collect:{source_id}", res.error or "unknown error"))

    # normalize signals for storage-safe stats (remove raw content fields)
    storage_safe_signals = [strip_raw_content_fields(it if isinstance(it, dict) else {}) for it in collected]

    # --- 2) Process ---
    process_ok = True
    try:
        draft_cards = run_ai_summarizer(collected)  # summarizer may use raw in-memory fields
        final_cards = run_risk_checker(draft_cards)
    except Exception as e:
        process_ok = False
        final_cards = []
        all_errors.append(("process", f"{e.__class__.__name__}: {e}"))
        # keep traceback for debugging in logs only (meta에는 메시지만)
        print("PROCESS ERROR:\n", traceback.format_exc())

    # --- 3) Save outputs ---
    save_ok = True
    try:
        hot_payload = build_hot_cards_payload(final_cards, meta)
        am_payload = build_briefing_payload(final_cards, "AM")
        pm_payload = build_briefing_payload(final_cards, "PM")
        watch_payload = build_watchlist_meta(meta)

        if not args.dry_run:
            ok1 = atomic_write_json(HOT_PATH, hot_payload)
            ok2 = atomic_write_json(BRIEF_AM_PATH, am_payload)
            ok3 = atomic_write_json(BRIEF_PM_PATH, pm_payload)
            ok4 = atomic_write_json(WATCHLIST_META_PATH, watch_payload)
            save_ok = bool(ok1 and ok2 and ok3 and ok4)
        else:
            print("(dry-run) outputs not written")

    except Exception as e:
        save_ok = False
        all_errors.append(("save", f"{e.__class__.__name__}: {e}"))
        print("SAVE ERROR:\n", traceback.format_exc())

    # --- 4) Finalize meta ---
    finished_utc = utc_now_iso()
    duration_ms = int((time.time() - started_ts) * 1000)

    meta["run"]["finished_utc"] = finished_utc
    meta["run"]["duration_ms"] = duration_ms

    meta["stats"] = compute_stats(storage_safe_signals, final_cards)

    # success rule: saved + processed + at least 1 collected item
    overall_success = bool(save_ok and process_ok and (len(collected) > 0))
    meta["success"] = overall_success
    if overall_success:
        meta["last_success_utc"] = finished_utc

    # attach errors
    for stage, msg in all_errors:
        try:
            append_error(meta, stage, Exception(msg), keep_last=50)
        except Exception:
            pass

    if not args.dry_run:
        okm = atomic_write_json(META_PATH, meta)
        if not okm:
            print("WARN: meta.json write failed")
            overall_success = False

    # Console summary
    print(f"[AtlasAxiom] run_id={run_id} run_type={args.run_type} env={args.env}")
    print(f"[AtlasAxiom] collected={len(collected)} cards={len(final_cards)} save_ok={save_ok} process_ok={process_ok} success={overall_success}")
    if all_errors:
        print("[AtlasAxiom] errors:")
        for s, m in all_errors:
            print(f" - {s}: {m}")

    return 0 if overall_success else 2


if __name__ == "__main__":
    raise SystemExit(main())
