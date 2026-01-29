import feedparser
from datetime import datetime

# 감시할 공식 뉴스 채널들
RSS_SOURCES = {
    "SEC_Press": "https://www.sec.gov/news/pressreleases.rss",
    "CNBC_Tech": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19854910",
    "Investing_News": "https://www.investing.com/rss/news.rss"
}

def fetch_official_news():
    results = []
    print("📡 공식 뉴스 수집 시작...")
    
    for name, url in RSS_SOURCES.items():
        try:
            # RSS 피드 읽어오기
            feed = feedparser.parse(url)
            # 최신 뉴스 2개만 가져오기
            for entry in feed.entries[:2]:
                results.append({
                    "source": name,
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", str(datetime.now()))
                })
        except Exception as e:
            print(f"⚠️ {name} 수집 실패: {e}")
            
    print(f"✅ 총 {len(results)}개의 뉴스 수집 완료")
    return results