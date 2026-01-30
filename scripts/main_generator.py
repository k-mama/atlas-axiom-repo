# scripts/main_generator.py
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Path setup (stable regardless of working directory)
# ──────────────────────────────────────────────────────────────────────────────
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DATA_DIR = REPO_ROOT / "data"

# Make imports work when running from repo root OR scripts dir
sys.path.insert(0, str(SCRIPTS_DIR))

# ──────────────────────────────────────────────────────────────────────────────
# Optional imports (collectors / processors)
# ──────────────────────────────────────────────────────────────────────────────
def _safe_import():
    collectors = {}
    processors = {}
    errors = []

    try:
        from collectors.official_rss import fetch_official_news  # type: ignore
        collectors["official_rss"] = fetch_official_news
    except Exception as e:
        errors.append(f"collectors.official_rss: {e}")

    try:
        from collectors.reddit_api import fetch_reddit_buzz  # type: ignore
        collectors["reddit_api"] = fetch_reddit_buzz
    except Exception as e:
        errors.append(f"collectors.reddit_api: {e}")

    try:
        from collectors.youtube_rss import fetch_youtube_videos  # type: ignore
        collectors["youtube_rss"] = fetch_youtube_videos
    except Exception as e:
        errors.append(f"collectors.youtube_rss: {e}")

    # processors (optional)
    try:
        from processors.ai_summarizer import summarize_news  # type: ignore
        processors["ai_summarizer"] = summarize_news
    except Exception as e:
        # Not fatal – we’ll fallback to heuristic card text
        errors.append(f"processors.ai_summarizer: {e}")

    try:
        from processors.risk_checker import run_risk_check  # type: ignore
        processors["risk_checker"] = run_risk_check
    except Exception as e:
        errors.append(f"processors.risk_checker: {e}")

    return collectors, processors, errors


# ──────────────────────────────────────────────────────────────────────────────
# Policy constants (Signal-only storage)
# ──────────────────────────────────────────────────────────────────────────────
SCHEMA_VERSION = "2026-01-30"
TARGET_TICKERS = ["NVDA", "MSFT", "GOOGL", "AMD", "TSLA"]
TICKER_ALIASES = {
    "GOOG": "GOOGL",
}
MAX_CARDS_DEFAULT = 12

# fields allowed inside "signal" object (strict)
ALLOWED_SIGNAL_FIELDS = {
    "source_type",       # news/youtube/social/forum
    "source_name",       # CNBC/Reuters/SEC/Reddit etc
    "url",               # link only
    "timestamp",         # ISO8601 Z
    "tickers",           # list
    "entities",          # list
    "keywords_top",      # list (<=5)
    "sentiment",         # pos/neg/neutral
    "intensity",         # 0~100
    "velocity",          # numeric or null
    "topic_cluster",     # string
    "confidence",        # 0~1
}

# Hard remove any potentially “content-like” fields from raw items before any save.
CONTENT_LIKE_KEYS = {
    "title", "headline", "content", "text", "body", "description", "summary",
    "transcript", "raw", "html", "markdown", "article", "post", "message",
    "selftext", "comment", "captions",
}

STOPWORDS = {
    "the","a","an","and","or","to","of","in","for","on","at","by","with","from",
    "is","are","was","were","be","been","it","this","that","as","will","its",
    "we","you","they","their","our","your",
}

TOPIC_RULES = [
    ("earnings", {"earnings","guidance","revenue","eps","margin","q1","q2","q3","q4"}),
    ("rates", {"fed","cpi","inflation","rates","yield","treasury","powell"}),
    ("ai", {"ai","gpu","nvidia","model","inference","training","llm","datacenter","cuda"}),
    ("ev", {"ev","autonomy","fsd","robotaxi","delivery","battery","charging"}),
    ("regulation", {"sec","doj","ftc","antitrust","lawsuit","regulation","ban"}),
    ("m&a", {"acquisition","merger","m&a","buyout","deal"}),
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def iso_z(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def parse_dt_any(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    s = str(value).strip()
    # Try ISO
    try:
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(s).astimezone(timezone.utc)
    except Exception:
        pass
    # Try RFC-ish / loose
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None

def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)

def write_json_if_changed(path: Path, obj: Any) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_txt = stable_json_dumps(obj)
    if path.exists():
        old_txt = path.read_text(encoding="utf-8")
        if old_txt == new_txt:
            return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(new_txt, encoding="utf-8")
    tmp.replace(path)
    return True

def sha256_short(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    # normalize common trackers
    u = re.sub(r"[?&](utm_[^=]+|fbclid|gclid)=[^&]+", "", u, flags=re.I)
    u = u.replace("??", "?").rstrip("?&")
    return u

def extract_tickers(text: str) -> List[str]:
    # $TSLA or TSLA (up to 5 chars)
    candidates = set(re.findall(r"\$?([A-Z]{1,5})\b", text or ""))
    out = []
    for c in candidates:
        if c in TICKER_ALIASES:
            c = TICKER_ALIASES[c]
        if c in TARGET_TICKERS:
            out.append(c)
    out.sort()
    return out

def extract_keywords(text: str, k: int = 5) -> List[str]:
    words = re.findall(r"[A-Za-z0-9]{3,}", (text or "").lower())
    words = [w for w in words if w not in STOPWORDS]
    if not words:
        return []
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for (w, _) in ranked[:k]]

def classify_topic(keywords: List[str], entities: List[str], tickers: List[str]) -> str:
    bag = set([k.lower() for k in keywords] + [e.lower() for e in entities] + [t.lower() for t in tickers])
    for topic, keys in TOPIC_RULES:
        if bag.intersection(keys):
            return topic
    return "general"

def infer_source_type(raw: Dict[str, Any], default: str) -> str:
    v = (raw.get("source_type") or raw.get("type") or raw.get("kind") or "").lower()
    if v in {"news","youtube","social","forum"}:
        return v
    return default

def infer_source_name(raw: Dict[str, Any], default: str) -> str:
    return str(raw.get("source_name") or raw.get("source") or raw.get("site") or raw.get("publisher") or default)

def infer_sentiment(raw: Dict[str, Any]) -> str:
    v = (raw.get("sentiment") or "neutral").lower()
    return v if v in {"pos","neg","neutral"} else "neutral"

def infer_intensity(raw: Dict[str, Any], source_type: str) -> int:
    # If collector already computed:
    if isinstance(raw.get("intensity"), (int, float)):
        return int(max(0, min(100, raw["intensity"])))
    # Otherwise infer from available engagement signals:
    up = raw.get("upvotes") or raw.get("score") or 0
    cm = raw.get("comments") or raw.get("num_comments") or 0
    vw = raw.get("views") or raw.get("view_count") or 0
    try:
        up = float(up); cm = float(cm); vw = float(vw)
    except Exception:
        up = cm = vw = 0.0
    base = 15.0
    if source_type in {"news"}:
        base = 25.0
    elif source_type in {"youtube"}:
        base = 18.0
    elif source_type in {"social","forum"}:
        base = 12.0
    score = base + (up * 0.08) + (cm * 0.30) + (vw * 0.00002)
    return int(max(0, min(100, score)))

def infer_velocity(raw: Dict[str, Any]) -> Optional[float]:
    v = raw.get("velocity")
    if isinstance(v, (int, float)):
        return float(v)
    # try common fields
    for key in ("velocity_pct", "growth", "growth_rate", "spike"):
        vv = raw.get(key)
        if isinstance(vv, (int, float)):
            return float(vv)
    return None

def tier_from_source(source_type: str, source_name: str) -> str:
    # Simple & safe rule:
    if source_type == "news":
        return "A"
    if source_type in {"youtube"}:
        # official channels are Tier A-ish, but we still treat as B unless collector tags it
        return "B"
    if source_type in {"social","forum"}:
        return "B"
    return "C"

def confidence_from_tier(tier: str) -> float:
    if tier == "A":
        return 0.85
    if tier == "B":
        return 0.65
    return 0.40

def make_signal_id(url: str, source_name: str) -> str:
    return sha256_short(f"{normalize_url(url)}|{source_name}")

def sanitize_for_storage(signal: Dict[str, Any]) -> Dict[str, Any]:
    # Keep only allowed signal fields, enforce types, trim sizes.
    out: Dict[str, Any] = {}
    for k in ALLOWED_SIGNAL_FIELDS:
        if k in signal:
            out[k] = signal[k]
    # Enforce constraints:
    out["tickers"] = list(dict.fromkeys(out.get("tickers") or []))[:10]
    out["entities"] = list(dict.fromkeys(out.get("entities") or []))[:20]
    out["keywords_top"] = list(dict.fromkeys(out.get("keywords_top") or []))[:5]
    out["intensity"] = int(max(0, min(100, int(out.get("intensity") or 0))))
    try:
        out["confidence"] = float(out.get("confidence") or 0)
    except Exception:
        out["confidence"] = 0.0
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    out["sentiment"] = out.get("sentiment") if out.get("sentiment") in {"pos","neg","neutral"} else "neutral"
    # velocity can be None
    if out.get("velocity") is not None:
        try:
            out["velocity"] = float(out["velocity"])
        except Exception:
            out["velocity"] = None
    return out

def score_signal(signal: Dict[str, Any], now: datetime) -> float:
    ts = parse_dt_any(signal.get("timestamp"))
    age_h = 24.0
    if ts:
        age_h = max(0.1, (now - ts).total_seconds() / 3600.0)
    recency = 1.0 / (1.0 + (age_h / 12.0))  # faster decay
    intensity = float(signal.get("intensity") or 0) / 100.0
    conf = float(signal.get("confidence") or 0)
    vel = signal.get("velocity")
    vel_boost = 1.0
    if isinstance(vel, (int, float)):
        vel_boost = 1.0 + min(1.5, max(0.0, float(vel) / 300.0))  # 300% spike -> +1.0
    return intensity * conf * recency * vel_boost

def default_checkpoints(topic: str, tickers: List[str]) -> List[str]:
    t = (tickers[0] if tickers else "ticker")
    if topic == "earnings":
        return [
            f"{t} guidance changes 여부(다음 업데이트)",
            "옵션 IV/프리마켓 반응 체크",
            "동일 섹터 동반 움직임(동일 키워드) 확인",
        ]
    if topic == "rates":
        return [
            "오늘/내일 매크로 캘린더(CPI/Fed/채권) 확인",
            "장기금리/달러 변동과 동행 여부",
            "성장주/AI 대형주 동반 리스크 점검",
        ]
    if topic == "ai":
        return [
            "공식 발표(리포/블로그/공시)로 확인",
            "관련 공급망/동종(AMD/MSFT/GOOGL 등) 반응 비교",
            "GPU/데이터센터 수요 신호(가이던스/발주) 추적",
        ]
    if topic == "ev":
        return [
            "생산/인도/가격변경 같은 공식 신호 확인",
            "규제/리콜/사고 이슈의 2차 확산 여부",
            "관련 밸류체인(배터리/충전) 동반 움직임 확인",
        ]
    if topic == "regulation":
        return [
            "공식 문서(기관/법원/공시) 1차 확인",
            "‘합의/조사/소송’ 단계 구분해서 추적",
            "동일 섹터 규제 확장 가능성 점검",
        ]
    if topic == "m&a":
        return [
            "공식 공시/보도자료 존재 여부 확인",
            "조건(가격/지분/규제심사) 핵심 변수 체크",
            "유사 딜 사례와 시장 반응 비교",
        ]
    return [
        "공식 소스에서 1차 확인",
        "커뮤니티 키워드 상승 지속 여부",
        "내일 검증할 체크포인트 3개 유지",
    ]

def heuristic_insight(signal: Dict[str, Any]) -> Dict[str, Any]:
    tickers = signal.get("tickers") or []
    topic = signal.get("topic_cluster") or "general"
    kws = signal.get("keywords_top") or []
    src = signal.get("source_name") or "source"

    # headline should be OUR wording (no source title)
    headline = " · ".join([t for t in tickers[:2]]) if tickers else "Market"
    if kws:
        headline = f"{headline}: {topic} signal ({kws[0]})"
    else:
        headline = f"{headline}: {topic} signal"

    angle = f"{src} 기반 신호 + 커뮤니티/공식 흐름을 분리해 확인"
    checkpoints = default_checkpoints(topic, tickers)

    return {
        "headline": headline,
        "angle": angle,
        "checkpoints": checkpoints[:3],
        "tags": list(dict.fromkeys((tickers + [topic])[:8])),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def call_collector(fn, **kwargs) -> List[Dict[str, Any]]:
    # collectors may accept (tickers=...) or not
    try:
        return fn(**kwargs)  # type: ignore
    except TypeError:
        try:
            return fn()  # type: ignore
        except Exception:
            return []
    except Exception:
        return []

def normalize_raw_item(raw: Dict[str, Any], default_source_type: str, default_source_name: str, collected_at: datetime) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None

    # We can USE text for extraction, but we will NEVER STORE it.
    # For keyword/ticker extraction, we may look at title/headline if present.
    title_like = str(raw.get("title") or raw.get("headline") or "")
    url = str(raw.get("url") or raw.get("link") or raw.get("permalink") or "").strip()
    if not url:
        return None
    url = normalize_url(url)

    source_type = infer_source_type(raw, default_source_type)
    source_name = infer_source_name(raw, default_source_name)
    tier = tier_from_source(source_type, source_name)

    ts = (
        parse_dt_any(raw.get("timestamp"))
        or parse_dt_any(raw.get("published_at"))
        or parse_dt_any(raw.get("published"))
        or parse_dt_any(raw.get("date"))
        or collected_at
    )

    # tickers/entities/keywords extraction
    tickers = raw.get("tickers")
    if not isinstance(tickers, list):
        tickers = extract_tickers(title_like)

    # If collector provides entities, use them. Otherwise derive minimal.
    entities = raw.get("entities")
    if not isinstance(entities, list):
        entities = []
        # Add tickers as entities (safe, not content)
        for t in tickers:
            entities.append(t)

    keywords = raw.get("keywords_top")
    if not isinstance(keywords, list):
        keywords = extract_keywords(title_like, k=5)

    topic = raw.get("topic_cluster")
    if not isinstance(topic, str) or not topic.strip():
        topic = classify_topic(keywords, entities, tickers)

    sentiment = infer_sentiment(raw)
    intensity = infer_intensity(raw, source_type)
    velocity = infer_velocity(raw)
    confidence = float(raw.get("confidence") or confidence_from_tier(tier))

    signal = {
        "source_type": source_type,
        "source_name": source_name,
        "url": url,
        "timestamp": iso_z(ts),
        "tickers": tickers,
        "entities": entities,
        "keywords_top": keywords,
        "sentiment": sentiment,
        "intensity": intensity,
        "velocity": velocity,
        "topic_cluster": topic,
        "confidence": confidence,
    }

    return sanitize_for_storage(signal)

def build_signals(collectors: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    collected_at = utc_now()
    raw_counts = {}
    raw_all: List[Tuple[str, Dict[str, Any]]] = []

    # Collect (Tier A/B sources)
    if "official_rss" in collectors:
        items = call_collector(collectors["official_rss"], tickers=TARGET_TICKERS)
        raw_counts["official_rss"] = len(items)
        for it in items:
            raw_all.append(("news", it))

    if "reddit_api" in collectors:
        items = call_collector(collectors["reddit_api"], tickers=TARGET_TICKERS)
        raw_counts["reddit_api"] = len(items)
        for it in items:
            raw_all.append(("forum", it))

    if "youtube_rss" in collectors:
        items = call_collector(collectors["youtube_rss"], tickers=TARGET_TICKERS)
        raw_counts["youtube_rss"] = len(items)
        for it in items:
            raw_all.append(("youtube", it))

    # Normalize + dedupe by URL
    dedup: Dict[str, Dict[str, Any]] = {}
    for default_type, raw in raw_all:
        default_name = {
            "news": "Official RSS",
            "forum": "Reddit",
            "youtube": "YouTube RSS",
        }.get(default_type, "Source")

        norm = normalize_raw_item(raw, default_type, default_name, collected_at)
        if not norm:
            continue

        # keep only our target tickers OR market-wide topics if empty tickers (optional)
        tickers = norm.get("tickers") or []
        if tickers:
            # already filtered by extract_tickers, but keep safe
            tickers = [t for t in tickers if t in TARGET_TICKERS]
            norm["tickers"] = tickers

        key = normalize_url(norm["url"])
        # If duplicate, keep the higher intensity one
        if key in dedup:
            if (norm.get("intensity") or 0) > (dedup[key].get("intensity") or 0):
                dedup[key] = norm
        else:
            dedup[key] = norm

    signals = list(dedup.values())
    stats = {
        "collected_at_utc": iso_z(collected_at),
        "raw_counts": raw_counts,
        "signals_count": len(signals),
    }
    return signals, stats

def build_cards(
    signals: List[Dict[str, Any]],
    processors: Dict[str, Any],
    max_cards: int
) -> List[Dict[str, Any]]:
    now = utc_now()

    scored = []
    for s in signals:
        scored.append((score_signal(s, now), s))
    scored.sort(key=lambda x: x[0], reverse=True)

    top = [s for _, s in scored[:max_cards]]

    # 1) Create base cards (heuristic)
    cards = []
    for s in top:
        card_id = make_signal_id(s["url"], s["source_name"])
        cards.append({
            "id": card_id,
            "signal": s,
            "insight": heuristic_insight(s),
            "score": round(score_signal(s, now), 6),
        })

    # 2) Optional AI summarizer can overwrite/augment insight safely
    if "ai_summarizer" in processors:
        try:
            # Provide ONLY signal fields (no raw content)
            safe_payload = [c["signal"] for c in cards]
            ai_result = processors["ai_summarizer"](safe_payload)  # type: ignore

            # If ai_summarizer returns list aligned with inputs:
            if isinstance(ai_result, list) and len(ai_result) == len(cards):
                for i, r in enumerate(ai_result):
                    if isinstance(r, dict):
                        # allow only these insight keys
                        allowed = {"headline", "angle", "checkpoints", "tags"}
                        merged = dict(cards[i]["insight"])
                        for k in allowed:
                            if k in r:
                                merged[k] = r[k]
                        # normalize checkpoints/tags
                        if isinstance(merged.get("checkpoints"), list):
                            merged["checkpoints"] = merged["checkpoints"][:3]
                        if isinstance(merged.get("tags"), list):
                            merged["tags"] = merged["tags"][:8]
                        cards[i]["insight"] = merged
        except Exception:
            pass

    # 3) Optional risk checker (final sanitization/filters)
    if "risk_checker" in processors:
        try:
            checked = processors["risk_checker"](cards)  # type: ignore
            if isinstance(checked, list):
                cards = checked
        except Exception:
            pass

    return cards

def build_briefing(cards: List[Dict[str, Any]], slot: str) -> Dict[str, Any]:
    # Minimal AM/PM wrapper so front-end can fetch
    return {
        "schema_version": SCHEMA_VERSION,
        "slot": slot,
        "generated_at_utc": iso_z(utc_now()),
        "top": cards[:7],  # daily 1-page brief target
    }

def make_meta(run_kind: str, stats: Dict[str, Any], cards_count: int, success: bool, extra_errors: List[str]) -> Dict[str, Any]:
    now = utc_now()
    return {
        "schema_version": SCHEMA_VERSION,
        "run_kind": run_kind,
        "success": bool(success),
        "last_success_utc": iso_z(now) if success else None,
        "generated_at_utc": iso_z(now),
        "generated_epoch_ms": int(now.timestamp() * 1000),
        "items": {
            "signals": int(stats.get("signals_count") or 0),
            "cards": int(cards_count),
        },
        "collectors": stats.get("raw_counts") or {},
        "git": {
            "sha": os.getenv("GITHUB_SHA"),
            "run_id": os.getenv("GITHUB_RUN_ID"),
            "workflow": os.getenv("GITHUB_WORKFLOW"),
        },
        "notes": "signal-only storage (no original content persisted)",
        "errors": extra_errors[:12],
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI / Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AtlasAxiom main generator (signal-only).")
    parser.add_argument("mode", choices=["hot", "briefing", "all"], help="what to generate")
    parser.add_argument("--slot", choices=["am", "pm"], default="am", help="briefing slot when mode=briefing")
    parser.add_argument("--max-cards", type=int, default=MAX_CARDS_DEFAULT, help="max cards for hot feed")
    args = parser.parse_args()

    collectors, processors, import_errors = _safe_import()

    # Run
    success = True
    extra_errors = list(import_errors)

    signals: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    try:
        signals, stats = build_signals(collectors)
    except Exception as e:
        success = False
        extra_errors.append(f"build_signals: {e}")
        signals, stats = [], {"raw_counts": {}, "signals_count": 0}

    # Cards
    cards: List[Dict[str, Any]] = []
    if success:
        try:
            cards = build_cards(signals, processors, max_cards=args.max_cards)
        except Exception as e:
            success = False
            extra_errors.append(f"build_cards: {e}")
            cards = []

    # Output paths
    hot_path = DATA_DIR / "hot_cards.json"
    meta_path = DATA_DIR / "meta.json"
    briefing_am_path = DATA_DIR / "briefing_am.json"
    briefing_pm_path = DATA_DIR / "briefing_pm.json"

    # Always write meta (even if fail) so UI can reflect status
    run_kind = args.mode if args.mode != "briefing" else f"briefing_{args.slot}"
    meta = make_meta(run_kind, stats, len(cards), success, extra_errors)

    # Generate outputs per mode
    wrote_any = False

    if args.mode in {"hot", "all"}:
        hot_payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": iso_z(utc_now()),
            "updated_at_utc": meta.get("last_success_utc"),  # what UI should use
            "cards": cards,
        }
        wrote_any |= write_json_if_changed(hot_path, hot_payload)

    if args.mode in {"briefing", "all"}:
        slot = args.slot
        briefing = build_briefing(cards, slot=slot)
        if slot == "am":
            wrote_any |= write_json_if_changed(briefing_am_path, briefing)
        else:
            wrote_any |= write_json_if_changed(briefing_pm_path, briefing)

    wrote_any |= write_json_if_changed(meta_path, meta)

    # Console summary
    print("🚀 AtlasAxiom Generator")
    print(f" - mode: {args.mode} (slot={args.slot})")
    print(f" - signals: {stats.get('signals_count', 0)} | cards: {len(cards)} | success: {success}")
    print(f" - wrote files: {'YES' if wrote_any else 'NO (unchanged)'}")
    if extra_errors:
        # show only first few
        print(" - notes/errors:")
        for m in extra_errors[:6]:
            print(f"   • {m}")

    # Exit code for CI
    if not success:
        sys.exit(2)

if __name__ == "__main__":
    main()
