import json
import os
import sys
from datetime import datetime

# --- 1. 경로 설정 및 모듈 불러오기 ---
# 현재 파일 위치를 기준으로 폴더 경로를 확실하게 잡습니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 우리가 만든 로봇들을 가져옵니다.
    from collectors.official_rss import fetch_official_news
    from collectors.reddit_api import fetch_reddit_buzz
    from collectors.youtube_rss import fetch_youtube_videos  # <--- 유튜브 추가됨
    from processors.ai_summarizer import summarize_news
    from processors.risk_checker import run_risk_check
except ImportError as e:
    print(f"❌ 모듈 로딩 실패: {e}")
    print("폴더 구조(scripts > collectors, processors)를 다시 확인해주세요.")
    sys.exit(1)

# 데이터 저장 경로
DATA_PATH = os.path.join("data", "hot_cards.json")

def main():
    print("🚀 AtlasAxiom 엔진 가동")
    
    # --- 1. 수집 (Collect) ---
    print("--- [1단계] 데이터 수집 시작 ---")
    
    # 1-1. 각 로봇 출동
    official_news = fetch_official_news()   # 공식 뉴스
    reddit_buzz = fetch_reddit_buzz()       # 레딧 여론
    youtube_vids = fetch_youtube_videos()   # 유튜브 영상 (<--- 추가됨)
    
    # 1-2. 모든 데이터 하나로 합치기
    raw_news = official_news + reddit_buzz + youtube_vids
    print(f"📊 총 {len(raw_news)}개의 원시 데이터를 수집했습니다.")

    # --- 2. 분석 및 요약 (Process) ---
    print("--- [2단계] AI 분석 및 리스크 필터링 ---")
    
    # 2-1. AI 요약 (초안 작성)
    draft_cards = summarize_news(raw_news)
    
    # 2-2. 법적 리스크 검사 (Risk Check)
    final_cards = run_risk_check(draft_cards)
    
    # --- 3. 저장 (Save) ---
    print("--- [3단계] 데이터 저장 ---")
    output_data = {
        "updated_at": datetime.now().isoformat(),
        "cards": final_cards
    }
    
    # data 폴더가 없으면 만들고 저장
    if not os.path.exists("data"):
        os.makedirs("data")
        
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ 모든 작업 완료! 저장된 파일: {DATA_PATH}")

if __name__ == "__main__":
    main()