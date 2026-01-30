# processors/ai_summarizer.py
from __future__ import annotations

import os
import re
import time
import json
import hashlib
import importlib
from typing import Any, Dict, List, Optional

from urllib.parse import urlsplit, urlunsplit

from .risk_checker import RiskChecker, NoOriginalTextError


def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _canonicalize_url(url: str) -> str:
    """
    URL query/fragment 저장 금지:
    저장 payload에는 query/fragment 제거된 canonical URL만 남김
    """
    u = (url or "").strip()
    if not u:
        return ""
    try:
        parts = urlsplit(u)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:
        return u


def _safe_str(x: Any, n: int = 80) -> str:
    s = str(x or "").strip()
    if len(s) <= n:
        return s
    return s[:n].rstrip() + "…"


def _domain(url: str) -> str:
    try:
        u = urlsplit((url or "").strip())
        return (u.netloc or "").lower()
    except Exception:
        return ""


class _NullAudit:
    def record_blocked(self, *args, **kwargs) -> None:
        return

    def snapshot(self) -> Dict[str, Any]:
        return {"total_blocked": 0, "by_code": {}, "by_source": {}}


class _JsonlAudit:
    """
    block_audit.py 가 없거나 import가 깨져도,
    최소한의 '원문 없는' 운영 로그를 남기기 위한 안전 폴백.
    """

    def __init__(self, path: str = "data/block_audit.jsonl") -> None:
        self.path = path
        self.total = 0
        self.by_code: Dict[str, int] = {}
        self.by_source: Dict[str, int] = {}

    def record_blocked(
        self,
        item_id: str,
        source: str,
        url: str,
        code: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.total += 1
        code = _safe_str(code, 60) or "unknown"
        src = _safe_str(source, 60) or "unknown"
        dom = _domain(url)
        url_hash = _sha(url)[:16] if url else ""

        self.by_code[code] = self.by_code.get(code, 0) + 1
        self.by_source[src] = self.by_source.get(src, 0) + 1

        record = {
            "at": _now_iso(),
            "code": code,
            "source": src,
            "domain": dom,
            "item_id": _safe_str(item_id, 24),
            "url_hash": url_hash,
            "meta": self._sanitize_meta(meta or {}),
        }

        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            return

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_blocked": self.total,
            "by_code": dict(sorted(self.by_code.items(), key=lambda x: x[1], reverse=True)),
            "by_source": dict(sorted(self.by_source.items(), key=lambda x: x[1], reverse=True)),
        }

    def _sanitize_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in list(meta.items())[:8]:
            kk = _safe_str(k, 40)
            if isinstance(v, (int, float, bool)) or v is None:
                out[kk] = v
            elif isinstance(v, str):
                out[kk] = _safe_str(v, 120)
            elif isinstance(v, list):
                out[kk] = [_safe_str(x, 40) for x in v[:10]]
            elif isinstance(v, dict):
                out[kk] = {_safe_str(k2, 30): _safe_str(v2, 60) for k2, v2 in list(v.items())[:8]}
            else:
                out[kk] = _safe_str(v, 60)
        return out


class AISummarizer:
    """
    투자자급 (핫카드/브리핑) 생성 엔진

    - 입력에는 본문이 들어올 수 있어도, 출력(저장 payload)에는 절대 남기지 않음
    - 출력은 risk_checker로 한 번 더 강제 필터링
    - 원문 위험으로 차단된 아이템은 (원문 없이) 별도 카운트/로그 남김
    """

    def __init__(
        self,
        model: Optional[str] = None,
        strict_no_original: bool = True,
        max_context_chars: int = 1200,
    ) -> None:
        self.model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
        self.strict_no_original = strict_no_original
        self.max_context_chars = max_context_chars

        self._risk = RiskChecker(strict=True)
        self._audit = self._init_audit()

        self._openai_client = self._init_openai_client()

    # ---------- public ----------

    def summarize(self, item: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._build_context(item)

        if self._openai_client is not None and os.getenv("OPENAI_API_KEY"):
            out = self._summarize_with_openai(ctx)
        else:
            out = self._summarize_fallback(ctx)

        payload = {
            "id": ctx["id"],
            "url": ctx.get("url"),
            "source": ctx.get("source"),
            "published_at": ctx.get("published_at"),
            "title": ctx.get("title"),
            "tickers": out.get("tickers", []),
            "tags": out.get("tags", []),
            "sentiment": out.get("sentiment", "neutral"),
            "horizon": out.get("horizon", "near-term"),
            "confidence": out.get("confidence", 0.55),
            "hot_cards": out.get("hot_cards", []),
            "briefing": out.get("briefing", []),
            "risk_flags": out.get("risk_flags", []),
            "_generated": {
                "by": "ai_summarizer",
                "version": "2026-01-31",
                "at": _now_iso(),
                "model": self.model,
            },
        }

        try:
            safe_payload, _ = self._risk.enforce_no_original_text(payload)
            return safe_payload
        except NoOriginalTextError as e:
            # (원문 없이) 운영자용 기록
            self._audit.record_blocked(
                item_id=ctx.get("id", ""),
                source=ctx.get("source", ""),
                url=ctx.get("url", ""),
                code=e.code,
                meta=e.meta,
            )
            raise

    def summarize_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for it in items:
            try:
                out.append(self.summarize(it))
            except NoOriginalTextError:
                continue
            except Exception:
                continue
        return out

    # ---------- init helpers ----------

    def _init_audit(self):
        """
        block_audit.py 가 있어도/없어도, 정적 분석/런타임 모두 안정적으로.
        """
        try:
            pkg = __package__ or "processors"
            mod = importlib.import_module(f"{pkg}.block_audit")
            BlockAudit = getattr(mod, "BlockAudit", None)
            if callable(BlockAudit):
                return BlockAudit(enabled=True)
        except Exception:
            pass

        try:
            return _JsonlAudit(path=os.getenv("BLOCK_AUDIT_PATH") or "data/block_audit.jsonl")
        except Exception:
            return _NullAudit()

    def _init_openai_client(self):
        """
        OpenAI SDK 버전 차이로 생기는 'Problems'를 없애기 위해
        importlib + getattr 로 완전 동적 처리.
        """
        try:
            mod = importlib.import_module("openai")
        except Exception:
            return None

        key = os.getenv("OPENAI_API_KEY") or ""

        OpenAI = getattr(mod, "OpenAI", None)
        if callable(OpenAI):
            try:
                return OpenAI(api_key=key)
            except Exception:
                return None

        # 구버전 모듈 방식
        try:
            setattr(mod, "api_key", key)
        except Exception:
            pass
        return mod

    # ---------- context build ----------

    def _build_context(self, item: Dict[str, Any]) -> Dict[str, Any]:
        title = (item.get("title") or item.get("headline") or "").strip()
        url_raw = (item.get("url") or item.get("link") or "").strip()
        url = _canonicalize_url(url_raw)
        source = (item.get("source") or item.get("publisher") or item.get("site") or "").strip()
        published_at = (item.get("published_at") or item.get("published") or item.get("date") or "").strip()

        snippet = (
            item.get("snippet")
            or item.get("summary")
            or item.get("description")
            or item.get("excerpt")
            or ""
        )
        snippet = str(snippet).strip()

        if not snippet:
            maybe_body = item.get("content") or item.get("body") or item.get("full_text") or item.get("text") or ""
            snippet = str(maybe_body or "").strip()

        snippet = self._clean_snippet(snippet)[: self.max_context_chars].strip()

        stable_id = url or (title + "|" + source + "|" + published_at)
        return {
            "id": _sha(stable_id)[:24],
            "title": title[:220],
            "url": url,
            "source": source[:120],
            "published_at": published_at[:40],
            "snippet": snippet,
        }

    def _clean_snippet(self, s: str) -> str:
        s = re.sub(r"<[^>]+>", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ---------- OpenAI ----------

    def _summarize_with_openai(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        system = (
            "You are an elite buy-side investment brief writer.\n"
            "Hard rules:\n"
            "- Never quote or reproduce the original article text.\n"
            "- Do NOT include long passages. Output must be short, distilled, and original.\n"
            "- Write in crisp investor tone. No hype.\n"
            "- Return ONLY valid JSON.\n"
            "Schema:\n"
            "{"
            "\"hot_cards\": [\"...\"], "
            "\"briefing\": [\"...\"], "
            "\"tickers\": [\"...\"], "
            "\"tags\": [\"...\"], "
            "\"sentiment\": \"bullish|bearish|neutral\", "
            "\"horizon\": \"near-term|mid-term|long-term\", "
            "\"confidence\": 0.0, "
            "\"risk_flags\": [\"...\"]"
            "}\n"
            "Constraints:\n"
            "- hot_cards: 3 to 5 bullets, each <= 160 chars.\n"
            "- briefing: 2 to 4 sentences, each <= 200 chars.\n"
            "- If tickers unknown, keep empty.\n"
        )

        user = (
            "Context (do not quote):\n"
            f"- title: {ctx.get('title','')}\n"
            f"- source: {ctx.get('source','')}\n"
            f"- published_at: {ctx.get('published_at','')}\n"
            f"- url: {ctx.get('url','')}\n"
            f"- snippet: {ctx.get('snippet','')}\n"
        )

        try:
            resp = self._openai_call(system, user)
            text = self._extract_openai_text(resp)
        except Exception:
            return self._summarize_fallback(ctx)

        parsed = self._safe_json(text)
        return self._normalize_output(parsed)

    def _openai_call(self, system: str, user: str):
        c = self._openai_client

        # (new client or new module) chat.completions.create
        chat = getattr(c, "chat", None)
        completions = getattr(chat, "completions", None) if chat is not None else None
        create_fn = getattr(completions, "create", None) if completions is not None else None
        if callable(create_fn):
            return create_fn(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
            )

        # (old module) ChatCompletion.create
        ChatCompletion = getattr(c, "ChatCompletion", None)
        create_old = getattr(ChatCompletion, "create", None) if ChatCompletion is not None else None
        if callable(create_old):
            return create_old(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
            )

        raise RuntimeError("openai_client_unavailable")

    def _extract_openai_text(self, resp: Any) -> str:
        # object style: resp.choices[0].message.content
        try:
            choices = getattr(resp, "choices", None)
            if choices and len(choices) > 0:
                msg = getattr(choices[0], "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if isinstance(content, str):
                        return content
        except Exception:
            pass

        # dict style: resp["choices"][0]["message"]["content"]
        try:
            if isinstance(resp, dict):
                return str(resp.get("choices", [{}])[0].get("message", {}).get("content", "{}") or "{}")
        except Exception:
            pass

        return "{}"

    def _safe_json(self, text: str) -> Dict[str, Any]:
        t = (text or "").strip()
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)

        dec = json.JSONDecoder()
        for i, ch in enumerate(t):
            if ch != "{":
                continue
            try:
                obj, _ = dec.raw_decode(t[i:])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return {}

    # ---------- fallback ----------

    def _summarize_fallback(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        title = ctx.get("title", "").strip()
        snippet = ctx.get("snippet", "").strip()

        hot_cards = []
        if title:
            hot_cards.append(f"Key update: {self._clip(title, 150)}")
        if snippet:
            hot_cards.append(f"Why it matters: {self._clip(self._infer_why(snippet), 150)}")
        hot_cards.append("Watch: second-order effects on sentiment, guidance, and near-term catalysts.")

        briefing = [
            self._clip(self._make_brief_sentence(title, snippet), 190),
            "Interpretation: treat as signal, not a trade. Monitor confirmation from filings/earnings/official releases.",
        ]

        return self._normalize_output(
            {
                "hot_cards": hot_cards[:5],
                "briefing": briefing[:4],
                "tickers": self._extract_tickers(title + " " + snippet),
                "tags": self._basic_tags(title, snippet),
                "sentiment": "neutral",
                "horizon": "near-term",
                "confidence": 0.55,
                "risk_flags": ["unverified-details", "headline-driven"],
            }
        )

    # ---------- normalize & helpers ----------

    def _normalize_output(self, d: Dict[str, Any]) -> Dict[str, Any]:
        def _list_str(x, max_items=8):
            if not isinstance(x, list):
                return []
            out = []
            for v in x[:max_items]:
                if isinstance(v, str):
                    vv = v.strip()
                    if vv:
                        out.append(vv)
            return out

        hot_cards = _list_str(d.get("hot_cards"))
        briefing = _list_str(d.get("briefing"))
        tickers = _list_str(d.get("tickers"))
        tags = _list_str(d.get("tags"))
        risk_flags = _list_str(d.get("risk_flags"))

        sentiment = d.get("sentiment") if isinstance(d.get("sentiment"), str) else "neutral"
        if sentiment not in ("bullish", "bearish", "neutral"):
            sentiment = "neutral"

        horizon = d.get("horizon") if isinstance(d.get("horizon"), str) else "near-term"
        if horizon not in ("near-term", "mid-term", "long-term"):
            horizon = "near-term"

        confidence = d.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.55
        confidence = max(0.0, min(1.0, confidence))

        hot_cards = [self._clip(x, 160) for x in hot_cards]
        briefing = [self._clip(x, 200) for x in briefing]

        return {
            "hot_cards": hot_cards[:5],
            "briefing": briefing[:4],
            "tickers": tickers[:8],
            "tags": tags[:8],
            "sentiment": sentiment,
            "horizon": horizon,
            "confidence": confidence,
            "risk_flags": risk_flags[:8],
        }

    def _clip(self, s: str, n: int) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[:n].rstrip() + "…"

    def _extract_tickers(self, s: str) -> List[str]:
        candidates = re.findall(r"\b[A-Z]{1,5}\b", s or "")
        stop = {"A", "AN", "THE", "AND", "OR", "TO", "IN", "ON", "FOR", "WITH", "BY", "AI", "US", "UK"}
        out = []
        for c in candidates:
            if c in stop:
                continue
            if c not in out:
                out.append(c)
        return out[:8]

    def _basic_tags(self, title: str, snippet: str) -> List[str]:
        t = (title + " " + snippet).lower()
        tags = []
        for key, tag in [
            ("earnings", "earnings"),
            ("guidance", "guidance"),
            ("sec", "regulatory"),
            ("lawsuit", "legal"),
            ("antitrust", "regulatory"),
            ("merger", "m&a"),
            ("acquisition", "m&a"),
            ("layoff", "labor"),
            ("rate", "macro"),
            ("fed", "macro"),
            ("inflation", "macro"),
            ("chip", "semis"),
            ("ai", "ai"),
            ("cloud", "cloud"),
            ("crypto", "crypto"),
        ]:
            if key in t and tag not in tags:
                tags.append(tag)
        return tags[:8]

    def _infer_why(self, snippet: str) -> str:
        if not snippet:
            return "Potential impact depends on confirmation and second-order effects."
        parts = re.split(r"[.!?]\s+", snippet)
        first = parts[0].strip() if parts else snippet.strip()
        return first or "Potential impact depends on confirmation and second-order effects."

    def _make_brief_sentence(self, title: str, snippet: str) -> str:
        if title and snippet:
            return f"{title}. Market relevance: {self._infer_why(snippet)}"
        if title:
            return f"{title}. Market relevance: monitor whether this changes expectations or pricing."
        if snippet:
            return f"Update: {self._infer_why(snippet)}"
        return "Update: limited context. Treat as a watch item until confirmed."
