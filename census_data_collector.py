"""
PNWER Trade Analysis Data Collector v8
采集25个州的数据用于更robust的DID分析

PNWER州 (5): WA, OR, ID, MT, AK
对照州 (20): 
  - 北部边境州: MI, MN, ND, WI, NY (靠近加拿大)
  - 西部州: CA, NV, UT, CO, WY (地理相近)
  - 能源/农业州: TX, LA, OK, NE, KS (产业相似)
  - 原对照州: FL, GA, NC, SC, VA (保留作为对比)
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

# PNWER州
PNWER_STATES = ["WA", "OR", "ID", "MT", "AK"]

# 对照州 - 选择与PNWER更相似的州
CONTROL_STATES = [
    # 北部边境州 (靠近加拿大，贸易结构相似)
    "MI", "MN", "ND", "WI", "NY",
    # 西部州 (地理位置相近)
    "CA", "NV", "UT", "CO", "WY",
    # 能源/农业州 (产业结构相似)
    "TX", "LA", "OK", "NE", "KS",
    # 原对照州 (保留作为对比)
    "FL", "GA", "NC", "SC", "VA"
]

ALL_STATES = PNWER_STATES + CONTROL_STATES

STATE_NAMES = {
    # PNWER
    "WA": "Washington", "OR": "Oregon", "ID": "Idaho", 
    "MT": "Montana", "AK": "Alaska",
    # 北部边境州
    "MI": "Michigan", "MN": "Minnesota", "ND": "North Dakota",
    "WI": "Wisconsin", "NY": "New York",
    # 西部州
    "CA": "California", "NV": "Nevada", "UT": "Utah",
    "CO": "Colorado", "WY": "Wyoming",
    # 能源/农业州
    "TX": "Texas", "LA": "Louisiana", "OK": "Oklahoma",
    "NE": "Nebraska", "KS": "Kansas",
    # 东南部州
    "FL": "Florida", "GA": "Georgia", "NC": "North Carolina",
    "SC": "South Carolina", "VA": "Virginia"
}

# 国家配置 (保持不变)
COUNTRIES = {
    "1220": {"code": "CA", "name": "Canada", "group": "usmca"},
    "2010": {"code": "MX", "name": "Mexico", "group": "usmca"},
    "5700": {"code": "CN", "name": "China", "group": "control"},
    "5880": {"code": "JP", "name": "Japan", "group": "control"},
    "4280": {"code": "DE", "name": "Germany", "group": "control"},
    "4120": {"code": "UK", "name": "United Kingdom", "group": "control"},
    "5800": {"code": "KR", "name": "South Korea", "group": "control"},
}

YEARS = list(range(2017, 2026))
YEAR_2025_MONTH = "11"

# API端点
STATE_EXPORT_API = "https://api.census.gov/data/timeseries/intltrade/exports/statehs"
STATE_IMPORT_API = "https://api.census.gov/data/timeseries/intltrade/imports/statehs"
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

def fetch_state_trade(state: str, country_code: str, year: int, month: str, 
                      is_export: bool = True) -> dict:
    """获取州级贸易数据（按产业）"""
    
    if is_export:
        api_url = STATE_EXPORT_API
        val_field = "ALL_VAL_YR"
        hs_field = "E_COMMODITY"
    else:
        api_url = STATE_IMPORT_API
        val_field = "GEN_VAL_YR"
        hs_field = "I_COMMODITY"
    
    params = {
        "get": f"STATE,CTY_CODE,{hs_field},{val_field}",
        "time": f"{year}-{month}",
        "STATE": state,
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


def fetch_national_trade(country_code: str, year: int, month: str, 
                         is_export: bool = True) -> int:
    """获取全美对某国的贸易总额"""
    
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
                
                total = 0
                for row in data[1:]:
                    try:
                        hs_code = row[hs_idx]
                        val_str = row[val_idx]
                        
                        if hs_code and len(hs_code) == 2 and val_str and val_str != '-':
                            total += int(val_str)
                    except (ValueError, IndexError):
                        pass
                return total
        return 0
    except Exception as e:
        return 0


def collect_state_data() -> dict:
    """采集所有州的数据"""
    print("\n" + "=" * 60)
    print(f"📊 州级数据采集 ({len(ALL_STATES)}个州)")
    print(f"   PNWER: {PNWER_STATES}")
    print(f"   对照: {len(CONTROL_STATES)}个州")
    print("=" * 60)
    
    # 请求数: 25州 × 2国 × 9年 × 2(出口+进口) = 900
    total_requests = len(ALL_STATES) * 2 * len(YEARS) * 2
    current = 0
    
    state_data = {}
    
    for state in ALL_STATES:
        state_data[state] = {
            "name": STATE_NAMES[state],
            "group": "pnwer" if state in PNWER_STATES else "control",
            "CA": {},
            "MX": {}
        }
        
        for cty_code, partner in [("1220", "CA"), ("2010", "MX")]:
            for year in YEARS:
                month = YEAR_2025_MONTH if year == 2025 else "12"
                
                # 出口
                current += 1
                print(f"\r[{current}/{total_requests}] {state} → {partner} ({year})...", end="", flush=True)
                exports = fetch_state_trade(state, cty_code, year, month, is_export=True)
                
                # 进口
                current += 1
                print(f"\r[{current}/{total_requests}] {state} ← {partner} ({year})...", end="", flush=True)
                imports = fetch_state_trade(state, cty_code, year, month, is_export=False)
                
                state_data[state][partner][str(year)] = {
                    "total": {
                        "exports": exports["total"],
                        "imports": imports["total"],
                        "balance": exports["total"] - imports["total"]
                    },
                    "by_industry": {}
                }
                
                all_industries = set(exports["by_industry"].keys()) | set(imports["by_industry"].keys())
                for ind in all_industries:
                    exp_val = exports["by_industry"].get(ind, 0)
                    imp_val = imports["by_industry"].get(ind, 0)
                    state_data[state][partner][str(year)]["by_industry"][ind] = {
                        "exports": exp_val,
                        "imports": imp_val,
                        "balance": exp_val - imp_val
                    }
                
                time.sleep(0.1)
    
    print(f"\n✅ 州级数据采集完成!")
    return state_data


def collect_national_data() -> dict:
    """采集国家级数据"""
    print("\n" + "=" * 60)
    print("📊 国家级数据采集")
    print("=" * 60)
    
    total_requests = len(COUNTRIES) * len(YEARS) * 2
    current = 0
    
    national_data = {}
    
    for cty_code, cty_info in COUNTRIES.items():
        code = cty_info["code"]
        national_data[code] = {
            "name": cty_info["name"],
            "group": cty_info["group"],
            "years": {}
        }
        
        for year in YEARS:
            month = YEAR_2025_MONTH if year == 2025 else "12"
            
            current += 1
            print(f"\r[{current}/{total_requests}] US → {code} ({year})...", end="", flush=True)
            exports = fetch_national_trade(cty_code, year, month, is_export=True)
            
            current += 1
            print(f"\r[{current}/{total_requests}] US ← {code} ({year})...", end="", flush=True)
            imports = fetch_national_trade(cty_code, year, month, is_export=False)
            
            national_data[code]["years"][str(year)] = {
                "exports": exports,
                "imports": imports,
                "total_trade": exports + imports,
                "balance": exports - imports
            }
            
            time.sleep(0.15)
    
    print(f"\n✅ 国家级数据采集完成!")
    return national_data


def build_output(state_data: dict, national_data: dict) -> dict:
    """构建输出"""
    
    # 按组分类对照州
    control_groups = {
        "border_north": ["MI", "MN", "ND", "WI", "NY"],
        "west": ["CA", "NV", "UT", "CO", "WY"],
        "energy_ag": ["TX", "LA", "OK", "NE", "KS"],
        "southeast": ["FL", "GA", "NC", "SC", "VA"]
    }
    
    return {
        "metadata": {
            "version": "8.0",
            "description": "PNWER Trade Analysis Data - Extended",
            "source": "U.S. Census Bureau API",
            "generated_at": datetime.datetime.now().isoformat(),
            "years": YEARS,
            "notes": [
                "25个州: 5 PNWER + 20 对照",
                "对照州选择标准: 地理位置、产业结构相似性",
                "层1 DID: US对CA/MX vs US对CN/JP/DE/UK/KR",
                "层2 DID: PNWER州 vs 对照州 (州×产业面板)"
            ]
        },
        
        "analysis_design": {
            "layer1_did": {
                "treatment": ["CA", "MX"],
                "control": ["CN", "JP", "DE", "UK", "KR"],
                "policy_event": "USMCA (effective 2020-07)"
            },
            "layer2_did": {
                "treatment": PNWER_STATES,
                "control": CONTROL_STATES,
                "control_groups": control_groups
            }
        },
        
        "industries": {
            "agriculture": {"hs_codes": "01-24"},
            "energy": {"hs_codes": "27"},
            "forestry": {"hs_codes": "44-49"},
            "minerals": {"hs_codes": "26, 72-76"},
            "manufacturing": {"hs_codes": "84-90"},
            "other": {"hs_codes": "others"}
        },
        
        "national_trade": national_data,
        "state_trade": state_data
    }


def print_summary(state_data: dict):
    """打印摘要"""
    print("\n" + "=" * 70)
    print("📊 数据摘要: 对CA出口 (2019 vs 2024)")
    print("=" * 70)
    
    print(f"\n{'州':<5} {'组别':<10} {'2019出口':<15} {'2024出口':<15} {'增长':<10}")
    print("-" * 60)
    
    growths = {"pnwer": [], "control": []}
    
    for state in ALL_STATES:
        data = state_data[state]
        group = data["group"]
        
        exp_19 = data["CA"].get("2019", {}).get("total", {}).get("exports", 0)
        exp_24 = data["CA"].get("2024", {}).get("total", {}).get("exports", 0)
        
        if exp_19 > 0:
            growth = (exp_24 / exp_19 - 1) * 100
            growths[group].append(growth)
        else:
            growth = 0
        
        print(f"{state:<5} {group:<10} ${exp_19/1e9:>10.2f}B    ${exp_24/1e9:>10.2f}B    {growth:>+6.1f}%")
    
    print("-" * 60)
    pnwer_avg = sum(growths["pnwer"]) / len(growths["pnwer"]) if growths["pnwer"] else 0
    control_avg = sum(growths["control"]) / len(growths["control"]) if growths["control"] else 0
    
    print(f"\nPNWER平均增长:  {pnwer_avg:+.1f}%")
    print(f"对照组平均增长: {control_avg:+.1f}%")
    print(f"简单DID:        {pnwer_avg - control_avg:+.1f}%")


def save_data(data: dict, path: str = "data/pnwer_analysis_data_v8.json"):
    """保存数据"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n📁 数据已保存: {path}")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       PNWER Trade Data Collector v8                         ║
    ║       25个州 (5 PNWER + 20 对照)                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print(f"⚠️  预计请求数: ~900个，耗时约8-10分钟\n")
    
    # 只采集州级数据
    state_data = collect_state_data()
    
    # 摘要
    print_summary(state_data)
    
    # 保存
    output = {
        "metadata": {
            "version": "8.0",
            "description": "PNWER Trade Analysis Data - Extended (States Only)",
            "source": "U.S. Census Bureau API",
            "generated_at": datetime.datetime.now().isoformat(),
            "years": YEARS,
            "notes": [
                "25个州: 5 PNWER + 20 对照",
                "对照州选择标准: 地理位置、产业结构相似性",
                "DID: PNWER州 vs 对照州 (州×产业面板)"
            ]
        },
        
        "analysis_design": {
            "did": {
                "treatment": PNWER_STATES,
                "control": CONTROL_STATES,
                "control_groups": {
                    "border_north": ["MI", "MN", "ND", "WI", "NY"],
                    "west": ["CA", "NV", "UT", "CO", "WY"],
                    "energy_ag": ["TX", "LA", "OK", "NE", "KS"],
                    "southeast": ["FL", "GA", "NC", "SC", "VA"]
                }
            }
        },
        
        "industries": {
            "agriculture": {"hs_codes": "01-24"},
            "energy": {"hs_codes": "27"},
            "forestry": {"hs_codes": "44-49"},
            "minerals": {"hs_codes": "26, 72-76"},
            "manufacturing": {"hs_codes": "84-90"},
            "other": {"hs_codes": "others"}
        },
        
        "state_trade": state_data
    }
    
    save_data(output, "data/pnwer_analysis_data_v8.json")
    
    print("\n" + "=" * 60)
    print("✅ 数据采集完成!")
    print(f"   州数: {len(ALL_STATES)} (PNWER: 5, 对照: 20)")
    print(f"   预期面板规模: {len(ALL_STATES)} × 6产业 × 9年 = {len(ALL_STATES)*6*9}观测值")
    print(f"   Cluster数: {len(ALL_STATES)} (df={len(ALL_STATES)-1})")
    print("=" * 60)


if __name__ == "__main__":
    main()