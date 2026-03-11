"""
National Trade Data Collector
采集美国对各贸易伙伴的国家层面出口数据

用于层1 DID分析：
- Treatment: US对CA/MX出口 (USMCA)
- Control: US对JP/KR/UK/DE出口 (稳定可比大经济体)

注意：排除CN，因为同期被贸易战强冲击
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

# 贸易伙伴
COUNTRIES = {
    # USMCA (Treatment)
    "1220": {"code": "CA", "name": "Canada", "group": "usmca"},
    "2010": {"code": "MX", "name": "Mexico", "group": "usmca"},
    # 对照国 (Control) - 稳定可比大经济体
    "5880": {"code": "JP", "name": "Japan", "group": "control"},
    "5800": {"code": "KR", "name": "South Korea", "group": "control"},
    "4120": {"code": "UK", "name": "United Kingdom", "group": "control"},
    "4280": {"code": "DE", "name": "Germany", "group": "control"},
}

YEARS = list(range(2017, 2026))
YEAR_2025_MONTH = "11"

NATIONAL_EXPORT_API = "https://api.census.gov/data/timeseries/intltrade/exports/hs"
NATIONAL_IMPORT_API = "https://api.census.gov/data/timeseries/intltrade/imports/hs"


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

def fetch_national_trade(country_code: str, year: int, month: str, 
                         is_export: bool = True) -> dict:
    """获取全美对某国的贸易数据（含产业分解）"""
    
    if is_export:
        api_url = NATIONAL_EXPORT_API
        val_field = "ALL_VAL_YR"
        hs_field = "E_COMMODITY"
    else:
        api_url = NATIONAL_IMPORT_API
        val_field = "GEN_VAL_YR"
        hs_field = "I_COMMODITY"
    
    params = {
        "get": f"CTY_CODE,{hs_field},{val_field}",
        "time": f"{year}-{month}",
        "CTY_CODE": country_code,
        "key": API_KEY
    }
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 1:
                headers = data[0]
                val_idx = headers.index(val_field)
                hs_idx = headers.index(hs_field)
                
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
    """采集所有国家层面数据"""
    
    print("\n" + "=" * 60)
    print("📊 国家层面贸易数据采集")
    print(f"   USMCA: CA, MX")
    print(f"   对照: JP, KR, UK, DE")
    print(f"   年份: {YEARS[0]}-{YEARS[-1]}")
    print("=" * 60)
    
    # 请求数: 6国 × 9年 × 2(出口+进口) = 108
    total_requests = len(COUNTRIES) * len(YEARS) * 2
    current = 0
    
    national_data = {}
    
    for cty_census_code, cty_info in COUNTRIES.items():
        code = cty_info["code"]
        national_data[code] = {
            "name": cty_info["name"],
            "group": cty_info["group"],
            "years": {}
        }
        
        for year in YEARS:
            month = YEAR_2025_MONTH if year == 2025 else "12"
            
            # 出口
            current += 1
            print(f"\r[{current}/{total_requests}] US → {code} ({year})...", 
                  end="", flush=True)
            exports = fetch_national_trade(cty_census_code, year, month, is_export=True)
            
            # 进口
            current += 1
            print(f"\r[{current}/{total_requests}] US ← {code} ({year})...", 
                  end="", flush=True)
            imports = fetch_national_trade(cty_census_code, year, month, is_export=False)
            
            national_data[code]["years"][str(year)] = {
                "exports": exports["total"],
                "imports": imports["total"],
                "total_trade": exports["total"] + imports["total"],
                "balance": exports["total"] - imports["total"],
                "exports_by_industry": exports["by_industry"],
                "imports_by_industry": imports["by_industry"]
            }
            
            time.sleep(0.15)
    
    print(f"\n✅ 采集完成!")
    return national_data


def build_output(national_data: dict) -> dict:
    """构建输出JSON"""
    
    return {
        "metadata": {
            "version": "1.0",
            "description": "US national trade with USMCA and control countries",
            "source": "U.S. Census Bureau API",
            "generated_at": datetime.datetime.now().isoformat(),
            "years": YEARS,
            "notes": [
                "用于层1 DID分析",
                "Treatment: CA, MX (USMCA)",
                "Control: JP, KR, UK, DE (稳定可比大经济体)",
                "排除CN (贸易战干扰)",
                "Post期建议从2021开始 (剔除2020过渡年)"
            ]
        },
        
        "analysis_design": {
            "treatment": ["CA", "MX"],
            "control": ["JP", "KR", "UK", "DE"],
            "pre_period": "2017-2019",
            "post_period": "2021-2025 (recommended)",
            "transition_year": 2020
        },
        
        "industries": {
            "agriculture": "HS 01-24",
            "energy": "HS 27",
            "forestry": "HS 44-49",
            "minerals": "HS 26, 72-76",
            "manufacturing": "HS 84-90",
            "other": "others"
        },
        
        "national_trade": national_data
    }


def print_summary(national_data: dict):
    """打印摘要"""
    print("\n" + "=" * 70)
    print("📊 数据摘要: US出口 (billions USD)")
    print("=" * 70)
    
    print(f"\n{'国家':<8} {'组别':<10} {'2019':<12} {'2024':<12} {'增长':<10}")
    print("-" * 55)
    
    for code, data in national_data.items():
        exp_19 = data["years"]["2019"]["exports"] / 1e9
        exp_24 = data["years"]["2024"]["exports"] / 1e9
        growth = (exp_24 / exp_19 - 1) * 100 if exp_19 > 0 else 0
        
        print(f"{code:<8} {data['group']:<10} ${exp_19:>8.1f}B    ${exp_24:>8.1f}B    {growth:>+6.1f}%")
    
    # 计算组平均
    print("-" * 55)
    for group in ["usmca", "control"]:
        pre_total = sum(d["years"]["2019"]["exports"] for d in national_data.values() 
                       if d["group"] == group)
        post_total = sum(d["years"]["2024"]["exports"] for d in national_data.values() 
                        if d["group"] == group)
        growth = (post_total / pre_total - 1) * 100 if pre_total > 0 else 0
        print(f"{group.upper():<8} {'平均':<10} ${pre_total/1e9:>8.1f}B    ${post_total/1e9:>8.1f}B    {growth:>+6.1f}%")


def save_data(data: dict, path: str = "data/national_trade.json"):
    """保存数据"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 数据已保存: {path}")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       National Trade Data Collector                         ║
    ║       US → USMCA(CA/MX) + Control(JP/KR/UK/DE)              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"⚠️  预计请求数: ~108个，耗时约2分钟\n")
    
    # 采集
    national_data = collect_all_data()
    
    # 摘要
    print_summary(national_data)
    
    # 保存
    output = build_output(national_data)
    save_data(output, "data/national_trade.json")
    
    print("\n" + "=" * 60)
    print("✅ 完成!")
    print("   用于层1 DID: USMCA整体效应")
    print("=" * 60)


if __name__ == "__main__":
    main()
