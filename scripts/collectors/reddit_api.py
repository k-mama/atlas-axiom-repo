import requests
from datetime import datetime

# 감시할 서브레딧 (주식/투자 관련)
SUBREDDITS = [
    "wallstreetbets",
    "investing",
    "stocks",
    "StockMarket"
]

def fetch_reddit_buzz():
    results = []
    print("🤖 레딧(Reddit) 여론 수집 시작...")
    
    # Reddit은 봇 차단을 막기 위해 독특한 User-Agent가 필수입니다.
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for sub in SUBREDDITS:
        try:
            # 공식 JSON 엔드포인트 사용 (API 키 불필요)
            url = f"https://www.reddit.com/r/{sub}/hot.json?limit=3"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                posts = data.get("data", {}).get("children", [])
                
                for post in posts:
                    post_data = post.get("data", {})
                    
                    # '고정된 공지글(stickied)'은 제외하고 진짜 유저 글만 수집
                    if post_data.get("stickied"):
                        continue
                        
                    results.append({
                        "source": f"Reddit (r/{sub})",
                        "title": post_data.get("title"),
                        "link": f"https://www.reddit.com{post_data.get('permalink')}",
                        "upvotes": post_data.get("score"),
                        "published": str(datetime.now()) # 실시간 수집 시각
                    })
            else:
                print(f"⚠️ r/{sub} 접속 제한 (Status: {response.status_code})")
                
        except Exception as e:
            print(f"⚠️ r/{sub} 수집 중 에러: {e}")
            
    print(f"✅ 총 {len(results)}개의 커뮤니티 핫 토픽 수집 완료")
    return results

# 테스트 실행용
if __name__ == "__main__":
    buzz = fetch_reddit_buzz()
    print(buzz)