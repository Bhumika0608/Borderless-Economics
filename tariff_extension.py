"""
PNWER Tariff Impact — Extensions
=================================
1. Oil Price Adjustment: Decomposes energy trade changes into price vs tariff effects
2. Export Impact Model: Mirrors import analysis for PNWER exports (CA/MX retaliatory tariffs)
3. Bilateral Summary: Combines import + export for full trade relationship picture
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict


# ============================================================================
# 1. OIL PRICE ADJUSTMENT
# ============================================================================

# WTI annual averages ($/bbl)
# Source: EIA / Statista
WTI_PRICES = {
    "2022": 94.53,
    "2023": 77.61,
    "2024": 77.13,
    "2025": 65.00,  # Estimate: Aug 2025 avg $64.86, full year ~$65
}

# Energy import value ≈ volume × price
# When oil price drops, energy import VALUE drops even if physical volume is stable
# We need to separate: ΔValue = ΔVolume + ΔPrice (approximately)
# ΔPrice/Price ≈ ΔWTI/WTI (strong correlation for crude-dominated energy trade)

def compute_oil_price_adjustment(state_trade: dict, pnwer_states: list) -> dict:
    """
    Decompose energy import changes into oil-price vs tariff components.
    
    Returns adjustment data for each state and aggregate PNWER.
    """
    wti_24 = WTI_PRICES["2024"]
    wti_25 = WTI_PRICES["2025"]
    wti_change_pct = (wti_25 - wti_24) / wti_24  # ~ -15.7%
    
    results = {"wti_2024": wti_24, "wti_2025": wti_25, "wti_change_pct": round(wti_change_pct * 100, 1)}
    results["by_state"] = {}
    
    total_energy_24 = 0
    total_energy_25 = 0
    
    for state in pnwer_states:
        for partner in ["CA", "MX"]:
            e24 = state_trade.get(state, {}).get(partner, {}).get("2024", {}).get(
                "by_industry", {}).get("energy", {}).get("imports", 0)
            e25 = state_trade.get(state, {}).get(partner, {}).get("2025", {}).get(
                "by_industry", {}).get("energy", {}).get("imports", 0)
            
            if e24 <= 0:
                continue
            
            actual_change = e25 - e24
            actual_pct = actual_change / e24 * 100
            
            # Price-driven change: if physical volume was constant, value would change by wti_change_pct
            price_driven = e24 * wti_change_pct
            price_driven_pct = wti_change_pct * 100
            
            # Tariff + volume residual
            tariff_residual = actual_change - price_driven
            tariff_residual_pct = actual_pct - price_driven_pct
            
            key = f"{state}_{partner}"
            results["by_state"][key] = {
                "energy_2024_M": round(e24 / 1e6, 1),
                "energy_2025_M": round(e25 / 1e6, 1),
                "actual_change_M": round(actual_change / 1e6, 1),
                "actual_change_pct": round(actual_pct, 1),
                "price_driven_M": round(price_driven / 1e6, 1),
                "price_driven_pct": round(price_driven_pct, 1),
                "tariff_residual_M": round(tariff_residual / 1e6, 1),
                "tariff_residual_pct": round(tariff_residual_pct, 1),
            }
            
            total_energy_24 += e24
            total_energy_25 += e25
    
    # PNWER aggregate
    if total_energy_24 > 0:
        total_actual = total_energy_25 - total_energy_24
        total_price = total_energy_24 * wti_change_pct
        total_tariff = total_actual - total_price
        results["pnwer_total"] = {
            "energy_2024_M": round(total_energy_24 / 1e6, 1),
            "energy_2025_M": round(total_energy_25 / 1e6, 1),
            "actual_change_M": round(total_actual / 1e6, 1),
            "actual_change_pct": round(total_actual / total_energy_24 * 100, 1),
            "price_driven_M": round(total_price / 1e6, 1),
            "price_driven_pct": round(wti_change_pct * 100, 1),
            "tariff_residual_M": round(total_tariff / 1e6, 1),
            "tariff_residual_pct": round(total_tariff / total_energy_24 * 100, 1),
        }
    
    return results


# ============================================================================
# 2. EXPORT IMPACT MODEL
# ============================================================================

# Canada retaliatory tariffs (effective March 2025):
# - 25% on C$30B (~US$22B) of US goods (Phase 1)
# - Targeting: steel, aluminum, consumer goods, agriculture
# - USMCA-compliant goods: mostly exempt from Sept 2025 onward
# Mexico retaliatory: more limited, primarily agricultural/steel

RETALIATORY_TARIFFS = {
    "CA": {
        "agriculture": 0.25,      # CA targeted US ag exports
        "energy": 0.10,           # Limited (mutual energy dependency)
        "forestry": 0.15,         # Partial targeting
        "minerals": 0.25,         # Steel/aluminum retaliation
        "manufacturing": 0.25,    # Consumer goods, machinery
        "other": 0.20,            # Mixed
    },
    "MX": {
        "agriculture": 0.15,      # Limited retaliation on US ag
        "energy": 0.05,           # Minimal (MX needs US energy)
        "forestry": 0.10,         # Minimal
        "minerals": 0.15,         # Some steel/aluminum
        "manufacturing": 0.20,    # Auto parts, machinery
        "other": 0.15,            # Mixed
    },
}

# Export elasticities (how sensitive are US exports to foreign tariffs)
# Generally lower than import elasticities because PNWER exports are
# often specialized (agricultural commodities, energy, aerospace)
EXPORT_ELASTICITIES = {
    "agriculture": -1.2,     # Moderate: alternative buyers exist but logistics constrain
    "energy": -0.3,          # Very inelastic: pipeline infrastructure locks trade patterns
    "forestry": -1.0,        # Moderate
    "minerals": -0.6,        # Low: specialized mineral trade
    "manufacturing": -1.5,   # Highest: most substitutable
    "other": -1.2,           # Moderate
}

# Export I-O multipliers (similar to import but from production side)
EXPORT_IO_MULTIPLIERS = {
    "agriculture": 2.0,
    "energy": 1.6,
    "forestry": 2.1,
    "minerals": 1.7,
    "manufacturing": 2.3,
    "other": 1.8,
}

EXPORT_JOBS_PER_MILLION = {
    "agriculture": 9.0,
    "energy": 2.0,
    "forestry": 7.5,
    "minerals": 4.5,
    "manufacturing": 6.0,
    "other": 5.5,
}


def analyze_export_impact(state_trade: dict, pnwer_states: list,
                          industries: list) -> dict:
    """
    Analyze impact of retaliatory tariffs on PNWER exports.
    Also computes actual 2024→2025 change for backtest.
    """
    results = {
        "methodology": {
            "description": "Impact of CA/MX retaliatory tariffs on PNWER exports",
            "retaliatory_tariff_schedule": RETALIATORY_TARIFFS,
            "export_elasticities": EXPORT_ELASTICITIES,
            "formula": "ΔExports = Exports_2024 × elasticity × retaliatory_tariff",
        },
        "by_state": {},
        "pnwer_total": {},
    }
    
    pnwer = {
        "exports_ca_24": 0, "exports_ca_25": 0,
        "exports_mx_24": 0, "exports_mx_25": 0,
        "model_change_ca": 0, "model_change_mx": 0,
        "gdp_impact": 0, "jobs_impact": 0,
        "by_industry": {},
    }
    
    for state in pnwer_states:
        st = {
            "exports_ca_24_M": 0, "exports_mx_24_M": 0,
            "model_change_ca_M": 0, "model_change_mx_M": 0,
            "actual_change_ca_M": 0, "actual_change_mx_M": 0,
            "gdp_impact_M": 0, "jobs_impact": 0,
            "by_industry": {},
        }
        
        for partner in ["CA", "MX"]:
            pk = partner.lower()
            retaliatory = RETALIATORY_TARIFFS[partner]
            
            for ind in industries:
                # Get 2024 baseline exports
                exp_24 = state_trade.get(state, {}).get(partner, {}).get(
                    "2024", {}).get("by_industry", {}).get(ind, {}).get("exports", 0)
                exp_25 = state_trade.get(state, {}).get(partner, {}).get(
                    "2025", {}).get("by_industry", {}).get(ind, {}).get("exports", 0)
                
                if exp_24 <= 0:
                    continue
                
                # Model prediction
                tariff = retaliatory.get(ind, 0.15)
                elast = EXPORT_ELASTICITIES.get(ind, -1.0)
                model_change = exp_24 * elast * tariff
                
                # Actual change
                actual_change = exp_25 - exp_24
                
                # GDP and jobs from export loss
                mult = EXPORT_IO_MULTIPLIERS.get(ind, 1.8)
                jpm = EXPORT_JOBS_PER_MILLION.get(ind, 5.0)
                
                gdp = abs(model_change) * mult if model_change < 0 else 0
                jobs = (gdp / 1e6) * jpm
                
                st[f"exports_{pk}_24_M"] += exp_24 / 1e6
                st[f"model_change_{pk}_M"] += model_change / 1e6
                st[f"actual_change_{pk}_M"] += actual_change / 1e6
                st["gdp_impact_M"] += gdp / 1e6
                st["jobs_impact"] += jobs
                
                pnwer[f"exports_{pk}_24"] += exp_24
                pnwer[f"exports_{pk}_25"] += exp_25
                pnwer[f"model_change_{pk}"] += model_change
                pnwer["gdp_impact"] += gdp
                pnwer["jobs_impact"] += jobs
                
                if ind not in st["by_industry"]:
                    st["by_industry"][ind] = {"exports_24_M": 0, "model_change_M": 0,
                                               "actual_change_M": 0}
                st["by_industry"][ind]["exports_24_M"] += exp_24 / 1e6
                st["by_industry"][ind]["model_change_M"] += model_change / 1e6
                st["by_industry"][ind]["actual_change_M"] += actual_change / 1e6
                
                if ind not in pnwer["by_industry"]:
                    pnwer["by_industry"][ind] = {"exports_24_M": 0, "model_M": 0,
                                                  "actual_M": 0}
                pnwer["by_industry"][ind]["exports_24_M"] += exp_24 / 1e6
                pnwer["by_industry"][ind]["model_M"] += model_change / 1e6
                pnwer["by_industry"][ind]["actual_M"] += actual_change / 1e6
        
        # Round state results
        for k in st:
            if isinstance(st[k], float):
                st[k] = round(st[k], 1)
        results["by_state"][state] = st
    
    # PNWER totals
    total_exp_24 = pnwer["exports_ca_24"] + pnwer["exports_mx_24"]
    total_exp_25 = pnwer["exports_ca_25"] + pnwer["exports_mx_25"]
    total_model = pnwer["model_change_ca"] + pnwer["model_change_mx"]
    total_actual = (total_exp_25 - total_exp_24)
    
    results["pnwer_total"] = {
        "exports_2024_M": round(total_exp_24 / 1e6, 1),
        "exports_2025_M": round(total_exp_25 / 1e6, 1),
        "model_change_total_M": round(total_model / 1e6, 1),
        "model_change_ca_M": round(pnwer["model_change_ca"] / 1e6, 1),
        "model_change_mx_M": round(pnwer["model_change_mx"] / 1e6, 1),
        "model_change_pct": round(total_model / total_exp_24 * 100, 1) if total_exp_24 > 0 else 0,
        "actual_change_M": round(total_actual / 1e6, 1),
        "actual_change_pct": round(total_actual / total_exp_24 * 100, 1) if total_exp_24 > 0 else 0,
        "gdp_impact_M": round(pnwer["gdp_impact"] / 1e6, 1),
        "jobs_impact": round(pnwer["jobs_impact"]),
        "by_industry": {
            ind: {k: round(v, 1) for k, v in vals.items()}
            for ind, vals in pnwer["by_industry"].items()
        },
    }
    
    return results


# ============================================================================
# 3. BILATERAL SUMMARY
# ============================================================================

def compute_bilateral_summary(state_trade: dict, pnwer_states: list,
                              import_results: dict, export_results: dict,
                              oil_adjustment: dict) -> dict:
    """
    Combine import + export + oil adjustment into a full bilateral picture.
    """
    # Import side (from v3 model base scenario)
    imp = import_results.get("scenarios", {}).get("base", {}).get("totals", {})
    imp_bands = import_results.get("bands", {})
    
    # Export side
    exp = export_results.get("pnwer_total", {})
    
    # Oil adjustment
    oil = oil_adjustment.get("pnwer_total", {})
    
    # Actual bilateral totals from raw data
    actual = {}
    for partner in ["CA", "MX"]:
        exp_24, exp_25, imp_24, imp_25 = 0, 0, 0, 0
        for state in pnwer_states:
            sd = state_trade.get(state, {}).get(partner, {})
            exp_24 += sd.get("2024", {}).get("total", {}).get("exports", 0)
            exp_25 += sd.get("2025", {}).get("total", {}).get("exports", 0)
            imp_24 += sd.get("2024", {}).get("total", {}).get("imports", 0)
            imp_25 += sd.get("2025", {}).get("total", {}).get("imports", 0)
        actual[partner] = {
            "exports_24_B": round(exp_24 / 1e9, 1),
            "exports_25_B": round(exp_25 / 1e9, 1),
            "exports_change_pct": round((exp_25 - exp_24) / exp_24 * 100, 1) if exp_24 > 0 else 0,
            "imports_24_B": round(imp_24 / 1e9, 1),
            "imports_25_B": round(imp_25 / 1e9, 1),
            "imports_change_pct": round((imp_25 - imp_24) / imp_24 * 100, 1) if imp_24 > 0 else 0,
            "total_trade_24_B": round((exp_24 + imp_24) / 1e9, 1),
            "total_trade_25_B": round((exp_25 + imp_25) / 1e9, 1),
            "total_change_pct": round(((exp_25 + imp_25) - (exp_24 + imp_24)) / (exp_24 + imp_24) * 100, 1) if (exp_24 + imp_24) > 0 else 0,
        }
    
    # Combined GDP and jobs impact
    import_gdp = imp.get("gdp_net_impact_M", 0)
    export_gdp = exp.get("gdp_impact_M", 0)
    import_jobs = imp.get("jobs_at_risk", 0)
    export_jobs = exp.get("jobs_impact", 0)
    
    return {
        "title": "PNWER Bilateral Trade Impact — Full Picture",
        "actual_bilateral": actual,
        "model_summary": {
            "import_side": {
                "description": "US tariffs on CA/MX → PNWER imports decline",
                "base_change_M": imp.get("change_total_M", 0),
                "base_change_pct": imp.get("change_pct", 0),
                "band_low_pct": imp_bands.get("change_pct", {}).get("low", 0),
                "band_high_pct": imp_bands.get("change_pct", {}).get("high", 0),
                "gdp_impact_M": import_gdp,
                "jobs_at_risk": import_jobs,
            },
            "export_side": {
                "description": "CA/MX retaliatory tariffs → PNWER exports decline",
                "model_change_M": exp.get("model_change_total_M", 0),
                "model_change_pct": exp.get("model_change_pct", 0),
                "actual_change_M": exp.get("actual_change_M", 0),
                "actual_change_pct": exp.get("actual_change_pct", 0),
                "gdp_impact_M": export_gdp,
                "jobs_impact": export_jobs,
            },
            "oil_price_adjustment": {
                "description": "Of the energy import decline, most is oil price not tariff",
                "total_energy_decline_M": oil.get("actual_change_M", 0),
                "price_driven_M": oil.get("price_driven_M", 0),
                "tariff_residual_M": oil.get("tariff_residual_M", 0),
                "price_share_pct": round(
                    abs(oil.get("price_driven_M", 0)) / abs(oil.get("actual_change_M", 1)) * 100, 0
                ) if oil.get("actual_change_M", 0) != 0 else 0,
            },
            "combined_impact": {
                "total_gdp_impact_M": round(import_gdp + export_gdp, 1),
                "total_jobs_at_risk": round(import_jobs + export_jobs),
                "total_trade_disrupted_M": round(
                    abs(imp.get("change_total_M", 0)) + abs(exp.get("model_change_total_M", 0)), 1
                ),
            },
        },
    }


# ============================================================================
# MAIN — Run Extensions
# ============================================================================

def run_extensions(state_data_path: str, import_results_path: str,
                   output_path: str = "tariff_bilateral_results.json"):
    """Run oil adjustment + export model + bilateral summary"""
    
    with open(state_data_path, 'r', encoding='utf-8') as f:
        state_data = json.load(f)
    state_trade = state_data.get("state_trade", {})
    
    with open(import_results_path, 'r', encoding='utf-8') as f:
        import_results = json.load(f)
    
    pnwer_states = ["WA", "OR", "ID", "MT", "AK"]
    industries = ["agriculture", "energy", "forestry", "minerals", "manufacturing", "other"]
    
    print("=" * 80)
    print("  PNWER TARIFF IMPACT — EXTENSIONS")
    print("=" * 80)
    
    # 1. Oil Price Adjustment
    print("\n  [1] Oil Price Adjustment...")
    oil = compute_oil_price_adjustment(state_trade, pnwer_states)
    ot = oil["pnwer_total"]
    print(f"      WTI: ${oil['wti_2024']:.0f} → ${oil['wti_2025']:.0f} ({oil['wti_change_pct']}%)")
    print(f"      Energy import decline: ${ot['actual_change_M']}M ({ot['actual_change_pct']}%)")
    print(f"        Price-driven:        ${ot['price_driven_M']}M ({ot['price_driven_pct']}%)")
    print(f"        Tariff residual:     ${ot['tariff_residual_M']}M ({ot['tariff_residual_pct']}%)")
    
    # 2. Export Impact
    print("\n  [2] Export Impact (Retaliatory Tariffs)...")
    exp = analyze_export_impact(state_trade, pnwer_states, industries)
    et = exp["pnwer_total"]
    print(f"      PNWER exports 2024: ${et['exports_2024_M']:,.0f}M")
    print(f"      Model predicted change: ${et['model_change_total_M']:,.0f}M ({et['model_change_pct']}%)")
    print(f"      Actual 2025 change:     ${et['actual_change_M']:,.0f}M ({et['actual_change_pct']}%)")
    print(f"      GDP impact: ${et['gdp_impact_M']:,.0f}M | Jobs: {et['jobs_impact']:,}")
    
    # Export backtest by industry
    print(f"\n      {'Industry':<15} {'Model%':>8} {'Actual%':>8} {'Gap':>8}")
    print(f"      {'-'*39}")
    for ind in industries:
        bi = et["by_industry"].get(ind, {})
        base = bi.get("exports_24_M", 0)
        if base > 0:
            mp = bi.get("model_M", 0) / base * 100
            ap = bi.get("actual_M", 0) / base * 100
            print(f"      {ind:<15} {mp:>7.1f}% {ap:>7.1f}% {mp-ap:>7.1f}")
    
    # 3. Bilateral Summary
    print("\n  [3] Bilateral Summary...")
    bilateral = compute_bilateral_summary(
        state_trade, pnwer_states, import_results, exp, oil
    )
    
    ba = bilateral["actual_bilateral"]
    print(f"\n      ACTUAL BILATERAL TRADE (2024 → 2025):")
    print(f"      {'':15} {'Exports':>12} {'Imports':>12} {'Total':>12}")
    for p in ["CA", "MX"]:
        d = ba[p]
        print(f"      {p:<15} ${d['exports_24_B']}B→${d['exports_25_B']}B  "
              f"${d['imports_24_B']}B→${d['imports_25_B']}B  "
              f"${d['total_trade_24_B']}B→${d['total_trade_25_B']}B ({d['total_change_pct']}%)")
    
    ms = bilateral["model_summary"]
    ci = ms["combined_impact"]
    print(f"\n      COMBINED IMPACT (Import + Export):")
    print(f"        Total GDP at risk:    ${ci['total_gdp_impact_M']:,.0f}M")
    print(f"        Total jobs at risk:   {ci['total_jobs_at_risk']:,}")
    print(f"        Trade disrupted:      ${ci['total_trade_disrupted_M']:,.0f}M")
    
    oa = ms["oil_price_adjustment"]
    print(f"\n      OIL PRICE NOTE:")
    print(f"        Energy import decline: ${oa['total_energy_decline_M']}M")
    print(f"        Oil price accounts for {oa['price_share_pct']:.0f}% of the decline")
    print(f"        Tariff-attributable:   ${oa['tariff_residual_M']}M only")
    
    # Save
    combined = {
        "oil_price_adjustment": oil,
        "export_impact": exp,
        "bilateral_summary": bilateral,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {output_path}")
    print("=" * 80)
    
    return combined


if __name__ == "__main__":
    data_paths = ["data/pnwer_analysis_data_v8.json", "pnwer_analysis_data_v8.json"]
    result_paths = ["data/tariff_impact_results_v3.json", "tariff_impact_results_v3.json"]
    
    dp = next((p for p in data_paths if Path(p).exists()), None)
    rp = next((p for p in result_paths if Path(p).exists()), None)
    
    if dp and rp:
        run_extensions(dp, rp)
    else:
        print("ERROR: Missing data files")
        print(f"  State data: {dp}")
        print(f"  Import results: {rp}")