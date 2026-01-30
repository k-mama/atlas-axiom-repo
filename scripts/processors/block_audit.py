# processors/risk_checker.py
from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit


class NoOriginalTextError(Exception):
    """
    원문/본문/긴 문장/HTML/URL query 등 "원문 위험"이 감지되면 저장을 중단시키는 예외.
    예외 메시지/메타에는 절대 원문/본문 텍스트가 들어가지 않도록 설계.
    """

    def __init__(self, code: str, meta: Dict[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.meta = meta or {}


@dataclass
class _Hit:
    code: str
    where: str
    detail: Dict[str, Any]


def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def _canonicalize_url(url: str) -> Tuple[str, bool]:
    """
    URL query/fragment(추적 파라미터 포함) 저장 금지:
    - 저장은 query/fragment 제거된 canonical URL만 허용
    - 변경이 발생했으면 (sanitized=True)
    """
    u = (url or "").strip()
    if not u:
        return "", False
    try:
        parts = urlsplit(u)
        # query + fragment 제거
        sanitized = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        changed = (sanitized != u)
        return sanitized, changed
    except Exception:
        # 파싱 실패는 위험으로 간주(저장 금지)
        return u, False


class RiskChecker:
    """
    (진짜로) 원문 저장 금지 강제 필터

    잡아내는 것:
    - 본문/원문/HTML 같은 "원문 흔적"
    - 지나치게 긴 문장(투자자용 핫카드/브리핑 기준)
    - 텍스트 안의 URL
    - URL query/fragment 저장(추적 파라미터 포함)
    """

    # payload에서 절대 저장되면 안 되는 키(혹시 섞여 들어오면 제거/차단)
    FORBIDDEN_KEYS = {
        "content",
        "body",
        "full_text",
        "text",
        "raw",
        "html",
        "article",
        "original",
        "original_text",
        "excerpt",
        "snippet",  # 입력 유입 방지용(요약 payload에는 없어야 정상)
        "description",
        "transcript",
    }

    # 텍스트에서 위험 패턴
    _RE_HTML_TAG = re.compile(r"<[^>]{1,80}>")
    _RE_HTML_ENTITY = re.compile(r"&(?:lt|gt|nbsp|quot|amp);", re.IGNORECASE)
    _RE_URL_IN_TEXT = re.compile(r"https?://", re.IGNORECASE)
    _RE_TRACKING_TOKENS = re.compile(r"\b(?:utm_|fbclid=|gclid=|ref=|source=)\b", re.IGNORECASE)
    _RE_MULTI_NEWLINE = re.compile(r"[\r\n]")
    _RE_WORD = re.compile(r"\b[\w']+\b")

    def __init__(
        self,
        strict: bool = True,
        max_hotcard_chars: int = 170,
        max_briefing_chars: int = 210,
        max_words_per_line: int = 30,
        max_quote_words: int = 7,
    ) -> None:
        self.strict = strict
        self.max_hotcard_chars = max_hotcard_chars
        self.max_briefing_chars = max_briefing_chars
        self.max_words_per_line = max_words_per_line
        self.max_quote_words = max_quote_words

    def enforce_no_original_text(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        return: (safe_payload, report)
        strict=True 인 경우, 위험 감지 시 NoOriginalTextError 발생(저장 금지).
        report는 원문 없이 "사유 코드/위치/길이" 같은 안전 정보만 포함.
        """
        p = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        hits: List[_Hit] = []

        # 1) 금지 키 제거/감지(혹시라도 섞여오면 강제 차단)
        removed = []
        self._remove_forbidden_keys(p, path="$", removed=removed)
        if removed:
            hits.append(_Hit(code="forbidden_field_present", where="$", detail={"count": len(removed)}))

        # 2) URL query/fragment 저장 금지: canonicalize + 변경 여부 기록
        url = p.get("url")
        if isinstance(url, str) and url.strip():
            canon, changed = _canonicalize_url(url)
            if changed:
                # query/fragment 있었던 것(추적 파라미터 가능) => 원칙적으로는 "위험"으로 잡는다
                hits.append(_Hit(code="url_query_or_fragment_present", where="url", detail={"sanitized": True}))
            p["url"] = canon
        elif url is not None and not isinstance(url, str):
            hits.append(_Hit(code="invalid_url_type", where="url", detail={"type": str(type(url))}))
            p["url"] = None

        # 3) hot_cards / briefing 텍스트 규칙 강화(투자자 서비스 기준)
        self._check_lines(p, key="hot_cards", max_chars=self.max_hotcard_chars, hits=hits)
        self._check_lines(p, key="briefing", max_chars=self.max_briefing_chars, hits=hits)

        # 4) payload 전체를 한번 더 스캔(예: 누군가 실수로 "content" 같은 걸 넣은 경우)
        self._scan_payload_for_html_or_urls(p, hits=hits)

        report = {
            "blocked": bool(hits),
            "hit_count": len(hits),
            "hits": [
                {"code": h.code, "where": h.where, "detail": h.detail}
                for h in hits
            ],
            # 운영자가 “원문 없이” 감만 잡게 하는 핑거프린트(텍스트 포함 금지)
            "fingerprint": _sha(f"{p.get('id','')}{p.get('source','')}{p.get('published_at','')}")[:16],
        }

        if hits and self.strict:
            # 절대 원문 텍스트/문장/타이틀을 meta에 넣지 말 것
            codes = sorted(list({h.code for h in hits}))
            meta = {"codes": codes, "hit_count": len(hits)}
            raise NoOriginalTextError("original_risk_blocked", meta=meta)

        return p, report

    # ----------------- internals -----------------

    def _remove_forbidden_keys(self, obj: Any, path: str, removed: List[str]) -> None:
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                new_path = f"{path}.{k}"
                if k in self.FORBIDDEN_KEYS:
                    removed.append(new_path)
                    obj.pop(k, None)
                    continue
                self._remove_forbidden_keys(obj.get(k), new_path, removed)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._remove_forbidden_keys(v, f"{path}[{i}]", removed)

    def _check_lines(self, p: Dict[str, Any], key: str, max_chars: int, hits: List[_Hit]) -> None:
        val = p.get(key, [])
        if val is None:
            return
        if not isinstance(val, list):
            hits.append(_Hit(code="invalid_list_type", where=key, detail={"type": str(type(val))}))
            p[key] = []
            return

        clean: List[str] = []
        for i, raw in enumerate(val):
            if not isinstance(raw, str):
                hits.append(_Hit(code="non_string_line", where=f"{key}[{i}]", detail={"type": str(type(raw))}))
                continue

            s = raw.strip()

            # 줄바꿈 금지(원문 복붙 흔적)
            if self._RE_MULTI_NEWLINE.search(s):
                hits.append(_Hit(code="newline_in_text", where=f"{key}[{i}]", detail={"len": len(s)}))
                s = re.sub(r"[\r\n]+", " ", s).strip()

            # HTML 태그/엔티티 금지
            if self._RE_HTML_TAG.search(s) or self._RE_HTML_ENTITY.search(s):
                hits.append(_Hit(code="html_detected", where=f"{key}[{i}]", detail={"len": len(s)}))

            # 텍스트 안의 URL 금지
            if self._RE_URL_IN_TEXT.search(s):
                hits.append(_Hit(code="url_in_text", where=f"{key}[{i}]", detail={"len": len(s)}))

            # 추적 토큰 같은 query 파편이 텍스트로 들어오는 경우 금지
            if self._RE_TRACKING_TOKENS.search(s):
                hits.append(_Hit(code="tracking_token_in_text", where=f"{key}[{i}]", detail={"len": len(s)}))

            # “긴 문장” 금지(단어수 + 글자수 둘 다)
            words = self._RE_WORD.findall(s)
            if len(words) > self.max_words_per_line:
                hits.append(_Hit(code="too_many_words", where=f"{key}[{i}]", detail={"words": len(words)}))

            if len(s) > max_chars:
                hits.append(_Hit(code="too_long", where=f"{key}[{i}]", detail={"len": len(s), "max": max_chars}))
                s = s[:max_chars].rstrip() + "…"

            # “인용문 느낌” 과다 금지 (큰따옴표 안 단어가 너무 많으면 원문 복붙 위험)
            q_words = self._count_words_inside_quotes(s)
            if q_words >= self.max_quote_words:
                hits.append(_Hit(code="quote_like_passage", where=f"{key}[{i}]", detail={"quote_words": q_words}))

            clean.append(s)

        p[key] = clean

    def _count_words_inside_quotes(self, s: str) -> int:
        # "..." 또는 “...” 내 단어 수만 세기 (원문 복붙 흔적 감지용)
        # 원문은 저장하지 않고 "단어 수"만 반환
        patterns = [
            r"\"([^\"]+)\"",
            r"“([^”]+)”",
            r"‘([^’]+)’",
            r"'([^']+)'",
        ]
        total = 0
        for pat in patterns:
            for m in re.finditer(pat, s):
                inside = m.group(1)
                total = max(total, len(self._RE_WORD.findall(inside)))
        return total

    def _scan_payload_for_html_or_urls(self, obj: Any, hits: List[_Hit], path: str = "$") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._scan_payload_for_html_or_urls(v, hits, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._scan_payload_for_html_or_urls(v, hits, f"{path}[{i}]")
        elif isinstance(obj, str):
            s = obj.strip()
            # 혹시 어디든 HTML/URL이 들어오면 잡는다
            if self._RE_HTML_TAG.search(s) or self._RE_HTML_ENTITY.search(s):
                hits.append(_Hit(code="html_detected_anywhere", where=path, detail={"len": len(s)}))
            if self._RE_URL_IN_TEXT.search(s):
                hits.append(_Hit(code="url_in_text_anywhere", where=path, detail={"len": len(s)}))
