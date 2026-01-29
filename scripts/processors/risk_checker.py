import re
import json

class RiskEngine:
    def __init__(self):
        # 감시할 금지어 리스트 (정규식) - 영어 & 한국어
        self.forbidden_patterns = [
            # English Patterns
            r"guaranteed return",  # 확정 수익
            r"buy now",            # 당장 사라
            r"sell immediately",   # 당장 팔아라
            r"sure thing",         # 확실하다
            r"100% win",           # 100% 승률
            r"risk-free",          # 무위험
            
            # Korean Patterns
            r"무조건 매수",
            r"확정 수익",
            r"전재산",
            r"몰빵",
            r"무조건 오릅니다",
            r"보장합니다"
        ]

    def audit(self, cards):
        print("⚖️ Legal Risk Audit in progress... (법적 리스크 감수 중)")
        validated_cards = []

        for card in cards:
            # 영어와 한국어 텍스트를 모두 가져와서 검사 (없으면 빈칸 처리)
            text_en = f"{card.get('headline_en', '')} {card.get('summary_en', '')}"
            text_kr = f"{card.get('headline_kr', '')} {card.get('summary_kr', '')}"
            
            # 1. 금지어 검사
            if self._has_risk(text_en) or self._has_risk(text_kr):
                print(f"⚠️ Risk Flagged in card {card.get('id', '?')}: Potential financial advice detected.")
                # 리스크가 발견되면 '주의' 태그를 붙임
                card['risk_flag'] = True
                card['compliance_note'] = "Automated Flag: Review required for investment advice language."
            else:
                card['risk_flag'] = False
            
            # 2. 표준 면책 조항(Disclaimer) 자동 부착
            # (법적 보호를 위해 모든 카드에 붙입니다)
            card['disclaimer_en'] = "This content is for informational purposes only and does not constitute financial advice."
            card['disclaimer_kr'] = "본 정보는 투자 참고용이며, 투자 권유가 아닙니다. 투자의 책임은 본인에게 있습니다."
            
            validated_cards.append(card)
            
        return validated_cards

    def _has_risk(self, text):
        """텍스트에 금지된 패턴이 있는지 확인합니다."""
        for pattern in self.forbidden_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

# 외부(main_generator.py)에서 쉽게 부를 수 있는 함수
def run_risk_check(cards):
    engine = RiskEngine()
    return engine.audit(cards)

# 테스트용 코드 (이 파일을 직접 실행했을 때만 작동)
if __name__ == "__main__":
    sample_cards = [
        {
            "id": "test_1", 
            "headline_en": "Buy Tesla Now! Guaranteed Profit!", 
            "summary_en": "It is a sure thing.",
            "headline_kr": "테슬라 무조건 매수!",
            "summary_kr": "확정 수익 보장합니다."
        },
        {
            "id": "test_2", 
            "headline_en": "Market Trends Analysis", 
            "summary_en": "Stocks are volatile today.",
            "headline_kr": "시장 동향 분석",
            "summary_kr": "오늘 주식 시장 변동성이 큽니다."
        }
    ]
    
    # 테스트 실행
    checked = run_risk_check(sample_cards)
    print(json.dumps(checked, indent=2, ensure_ascii=False))