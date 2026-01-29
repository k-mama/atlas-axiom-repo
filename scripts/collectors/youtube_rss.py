import feedparser
from datetime import datetime

# 감시할 유튜브 채널 ID 목록
# (원하는 채널의 ID를 찾아서 여기에 추가하면 됩니다)
YOUTUBE_CHANNELS = {
    "CNBC_Television": "UCvJJ_dzjViJCoLf5uKUTwoA",    # CNBC
    "Bloomberg_Tech": "UCrM7B7SL_g1edFOnmj-SDKg",     # Bloomberg Technology
    "Ark_Invest": "UCQI-Ym2r8RhinhGW8TEgMWg"           # ARK Invest (캐시우드)
}

def fetch_youtube_videos():
    results = []
    print("📺 유튜브(YouTube) 최신 영상 수집 시작...")
    
    for name, channel_id in YOUTUBE_CHANNELS.items():
        try:
            # 유튜브 RSS 주소 생성
            rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
            
            feed = feedparser.parse(rss_url)
            
            # 최신 영상 2개만 가져오기
            for entry in feed.entries[:2]:
                results.append({
                    "source": f"YouTube ({name})",
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", str(datetime.now())),
                    # 썸네일 이미지는 보통 media_thumbnail에 있습니다
                    "thumbnail": entry.media_thumbnail[0]['url'] if 'media_thumbnail' in entry else "" 
                })
        except Exception as e:
            print(f"⚠️ {name} 유튜브 수집 실패: {e}")
            
    print(f"✅ 총 {len(results)}개의 유튜브 영상 데이터 수집 완료")
    return results

# 테스트용 실행 코드
if __name__ == "__main__":
    videos = fetch_youtube_videos()
    print(videos)