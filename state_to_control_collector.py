"""
PNWER State-to-Control-Countries Data Collector v9
采集25个州对非USMCA对照国的出口数据

用于州内DDD分析：
- 州对CA/MX出口 (treatment) vs 州对JP/KR/UK/DE出口 (control)

对照国选择标准：
- 稳定可比的大经济体
- 避免同期被贸易战/制裁强冲击的国家（排除CN）
"""

import requests
import json
import time
from pathlib import Path
import datetime

# ============================================================================
# 配置
# ============================================================================

API_KEY = "f51f8af17882fc49a8c6a2eec80c9b9d522562fd"

# 25个州（与v8一致）
PNWER_STATES = ["WA", "OR", "ID", "MT", "AK"]
CONTROL_STATES = [
    "MI", "MN", "ND", "WI", "NY",  # 北部边境
    "CA", "NV", "UT", "CO", "WY",  # 西部
    "TX", "LA", "OK", "NE", "KS",  # 能源/农业
    "FL", "GA", "NC", "SC", "VA"   # 东南部
]
ALL_STATES = PNWER_STATES + CONTROL_STATES

STATE_NAMES = {
    "WA": "Washington", "OR": "Oregon", "ID": "Idaho", 
    "MT": "Montana", "AK": "Alaska",
    "MI": "Michigan", "MN": "Minnesota", "ND": "North Dakota",
    "WI": "Wisconsin", "NY": "New York",
    "CA": "California", "NV": "Nevada", "UT": "Utah",
    "CO": "Colorado", "WY": "Wyoming",
    "TX": "Texas", "LA": "Louisiana", "OK": "Oklahoma",
    "NE": "Nebraska", "KS": "Kansas",
    "FL": "Florida", "GA": "Georgia", "NC": "North Carolina",
    "SC": "South Carolina", "VA": "Virginia"
}

# 对照国（排除CN，避免贸易战干扰）
CONTROL_COUNTRIES = {
    "5880": {"code": "JP", "name": "Japan"},
    "5800": {"code": "KR", "name": "South Korea"},
    "4120": {"code": "UK", "name": "United Kingdom"},
    "4280": {"code": "DE", "name": "Germany"},
}

YEARS = list(range(2017, 2026))
YEAR_2025_MONTH = "11"

STATE_EXPORT_API = "https://api.census.gov/data/timeseries/intltrade/exports/statehs"


# ============================================================================
# HS2 → 产业映射
# ============================================================================

def get_industry_from_hs2(hs2_code: str) -> str:
    try:
        hs2 = int(hs2_code)
    except (ValueError, TypeError):
        return None
    
    if 1 <= hs2 <= 24:
        return "agriculture"
    elif hs2 == 27:
        return "energy"
    elif 44 <= hs2 <= 49:
        return "forestry"
    elif hs2 in [26, 72, 73, 74, 75, 76]:
        return "minerals"
    elif hs2 in [84, 85, 86, 87, 88, 89, 90]:
        return "manufacturing"
    else:
        return "other"


# ============================================================================
# 数据采集
# ============================================================================

def fetch_state_exports(state: str, country_code: str, year: int, month: str) -> dict:
    """获取州对某国的出口数据"""
    
    params = {
        "get": f"STATE,CTY_CODE,E_COMMODITY,ALL_VAL_YR",
        "time": f"{year}-{month}",
        "STATE": state,
        "CTY_CODE": country_code,
        "key": API_KEY
    }
    
    try:
        response = requests.get(STATE_EXPORT_API, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 1:
                headers = data[0]
                val_idx = headers.index("ALL_VAL_YR")
                hs_idx = headers.index("E_COMMODITY")
                
                industry_totals = {}
                total = 0
                
                for row in data[1:]:
                    try:
                        hs_code = row[hs_idx]
                        val_str = row[val_idx]
                        
                        # 只取HS2级别
                        if not hs_code or len(hs_code) != 2:
                            continue
                        if not val_str or val_str == '-':
                            continue
                        
                        value = int(val_str)
                        industry = get_industry_from_hs2(hs_code)
                        
                        if industry:
                            industry_totals[industry] = industry_totals.get(industry, 0) + value
                            total += value
                    except (ValueError, IndexError):
                        pass
                
                return {"total": total, "by_industry": industry_totals}
        return {"total": 0, "by_industry": {}}
    except Exception as e:
        return {"total": 0, "by_industry": {}}


def collect_all_data() -> dict:
    """采集所有州对对照国的出口数据"""
    
    print("\n" + "=" * 60)
    print(f"📊 州→对照国 出口数据采集")
    print(f"   州: {len(ALL_STATES)}个")
    print(f"   对照国: {list(CONTROL_COUNTRIES.values())}")
    print(f"   年份: {YEARS[0]}-{YEARS[-1]}")
    print("=" * 60)
    
    # 请求数: 25州 × 4国 × 9年 = 900
    total_requests = len(ALL_STATES) * len(CONTROL_COUNTRIES) * len(YEARS)
    current = 0
    
    state_data = {}
    
    for state in ALL_STATES:
        state_data[state] = {
            "name": STATE_NAMES[state],
            "group": "pnwer" if state in PNWER_STATES else "control"
        }
        
        for cty_census_code, cty_info in CONTROL_COUNTRIES.items():
            cty_code = cty_info["code"]
            state_data[state][cty_code] = {}
            
            for year in YEARS:
                month = YEAR_2025_MONTH if year == 2025 else "12"
                
                current += 1
                print(f"\r[{current}/{total_requests}] {state} → {cty_code} ({year})...", 
                      end="", flush=True)
                
                exports = fetch_state_exports(state, cty_census_code, year, month)
                
                state_data[state][cty_code][str(year)] = {
                    "exports": exports["total"],
                    "by_industry": exports["by_industry"]
                }
                
                time.sleep(0.1)
    
    print(f"\n✅ 采集完成!")
    return state_data


def build_output(state_data: dict) -> dict:
    """构建输出JSON"""
    
    return {
        "metadata": {
            "version": "9.0",
            "description": "State exports to control countries (for DDD analysis)",
            "source": "U.S. Census Bureau API",
            "generated_at": datetime.datetime.now().isoformat(),
            "years": YEARS,
            "notes": [
                "用于州内DDD分析的对照国数据",
                "对照国: JP, KR, UK, DE (排除CN避免贸易战干扰)",
                "与v8州数据配合使用"
            ]
        },
        
        "control_countries": {
            code: info["name"] for code, info in 
            {v["code"]: v for v in CONTROL_COUNTRIES.values()}.items()
        },
        
        "states": {
            "pnwer": PNWER_STATES,
            "control": CONTROL_STATES
        },
        
        "industries": {
            "agriculture": "HS 01-24",
            "energy": "HS 27",
            "forestry": "HS 44-49",
            "minerals": "HS 26, 72-76",
            "manufacturing": "HS 84-90",
            "other": "others"
        },
        
        "state_exports_to_control": state_data
    }


def print_summary(state_data: dict):
    """打印摘要"""
    print("\n" + "=" * 70)
    print("📊 数据摘要: 各州对对照国出口 (2019)")
    print("=" * 70)
    
    print(f"\n{'州':<5} {'组别':<8} {'→JP':<12} {'→KR':<12} {'→UK':<12} {'→DE':<12}")
    print("-" * 65)
    
    for state in ALL_STATES[:10]:  # 只显示前10个
        data = state_data[state]
        group = data["group"]
        
        jp = data.get("JP", {}).get("2019", {}).get("exports", 0) / 1e9
        kr = data.get("KR", {}).get("2019", {}).get("exports", 0) / 1e9
        uk = data.get("UK", {}).get("2019", {}).get("exports", 0) / 1e9
        de = data.get("DE", {}).get("2019", {}).get("exports", 0) / 1e9
        
        print(f"{state:<5} {group:<8} ${jp:>8.2f}B  ${kr:>8.2f}B  ${uk:>8.2f}B  ${de:>8.2f}B")
    
    print("... (更多州省略)")


def save_data(data: dict, path: str = "data/state_to_control_countries.json"):
    """保存数据"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 数据已保存: {path}")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       State → Control Countries Export Collector            ║
    ║       25州 × 4对照国 (JP/KR/UK/DE)                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"⚠️  预计请求数: ~900个，耗时约8-10分钟\n")
    
    # 采集
    state_data = collect_all_data()
    
    # 摘要
    print_summary(state_data)
    
    # 保存
    output = build_output(state_data)
    save_data(output, "data/state_to_control_countries.json")
    
    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("   此数据用于州内DDD分析")
    print("   需配合 pnwer_analysis_data_v8.json 使用")
    print("=" * 60)


if __name__ == "__main__":
    main()
