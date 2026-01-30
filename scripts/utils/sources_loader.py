from __future__ import annotations
import json
import os
from typing import Any, Dict


def _repo_root() -> str:
    # scripts/utils/* 기준으로 레포 루트 추정
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def load_sources() -> Dict[str, Any]:
    root = _repo_root()
    candidates = [
        os.path.join(root, "src", "sources.json"),          # ✅ 지금 구조
        os.path.join(root, "sources", "sources.json"),      # ✅ 예전 구조(있으면 지원)
        os.path.join(root, "sources.json"),
    ]

    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError(
            "sources.json을 찾을 수 없습니다. 다음 중 한 곳에 두세요:\n"
            f"- {candidates[0]}\n- {candidates[1]}\n- {candidates[2]}"
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "sources" not in data or not isinstance(data["sources"], list):
        raise ValueError("sources.json 형식 오류: { watchlist:[], sources:[] } 형태여야 합니다.")
    data.setdefault("watchlist", [])
    return data
