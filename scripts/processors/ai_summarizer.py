import json

def summarize_news(news_list):
    print("🧠 AI Analyzing News... (Mock Mode)")
    
    processed_cards = []
    
    # 가져온 뉴스를 AtlasAxiom 카드 형식으로 변환 (Bilingual Support)
    for i, news in enumerate(news_list):
        card = {
            "id": f"auto_{i}",
            "type": "INFO", 
            "ticker": "MKT", 
            # 영어 버전 (기본)
            "headline_en": news['title'],
            "summary_en": f"1. Breaking news reported by {news['source']}.\n2. Click link to verify original source.\n3. AI analysis module connecting soon.",
            
            # 한국어 버전 (토글용 - 지금은 단순 번역 시늉만 냄)
            "headline_kr": f"[속보] {news['title']} (AI 번역 대기중)",
            "summary_kr": f"1. {news['source']}에서 보도된 속보입니다.\n2. 클릭하여 원문을 확인하세요.\n3. AI 분석 모듈이 곧 연결됩니다.",
            
            "source_links": [news['link']],
            "trust_score": 80
        }
        processed_cards.append(card)
        
    return processed_cards