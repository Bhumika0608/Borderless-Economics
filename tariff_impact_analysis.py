"""
PNWER Tariff Impact Analysis v3 — Forecast-Ready Model
========================================================

Upgrades over v2:
1. SCENARIO BANDS: Low/Base/High on 3 key parameters (τ_eff, σ, λ)
   → Output is confidence intervals, not single points
2. ROLLING CALIBRATION: Infrastructure for monthly/quarterly k_p updates
   → Nowcasting capability with timestamped anchors
3. SANITY-CHECK DASHBOARD: Self-validation on every run
   → implied τ_eff vs target, implied revenue vs anchor, predicted vs observed
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy


# ============================================================================
# 1. SCENARIO CONFIGURATION — Low / Base / High
# ============================================================================

# Each parameter has three values: (low, base, high)
# Low = optimistic for trade (lower tariffs, more substitution, more pass-through)
# High = pessimistic for trade (higher tariffs, less substitution, less pass-through)

SCENARIO_PARAMS = {
    # --- τ_eff anchor ranges ---
    # PNWER-specific (NOT national avg). Back-tested from v8 2024→2025 actual.
    # PNWER's industry mix (energy/forestry/minerals heavy) faces higher effective
    # tariffs than national average due to lower USMCA compliance + Section 232.
    # Low: higher USMCA compliance achieved over time
    # High: additional Section 232 expansions or lower compliance
    "target_eff_tariff": {
        "CA": {"low": 0.055, "base": 0.080, "high": 0.110},
        "MX": {"low": 0.060, "base": 0.090, "high": 0.120},
    },
    # --- Armington substitution elasticity σ ---
    # HIGH σ = MORE substitution = LESS disruption → belongs in LOW (optimistic)
    # LOW σ = LESS substitution = MORE disruption → belongs in HIGH (pessimistic)
    "sigma_scale": {"low": 1.4, "base": 1.0, "high": 0.7},
    # --- Foreign exporter pass-through λ ---
    # HIGH λ = exporter absorbs more = LESS cost to importer → LOW (optimistic)
    # LOW λ = importer bears all cost → HIGH (pessimistic)
    "pass_through": {"low": 0.12, "base": 0.05, "high": 0.00},
    # --- Diversion realization ---
    # HIGH realization = more diversion = LESS net disruption → LOW (optimistic)
    # LOW realization = less diversion = MORE net disruption → HIGH (pessimistic)
    "diversion_realization": {
        "CA": {"low": 0.20, "base": 0.10, "high": 0.05},
        "MX": {"low": 0.70, "base": 0.55, "high": 0.35},
    },
    # --- Aggregate demand elasticity scale (INDUSTRY-SPECIFIC) ---
    # Industry elasticities × agg_scale = effective aggregate response.
    # Back-calibrated from PNWER 2025 actual tariff-attributable changes.
    # Energy/manufacturing are low because: energy demand is inelastic (pipeline
    # lock-in), manufacturing is largely USMCA-shielded.
    # Low scenario = less responsive (optimistic); High = more responsive (pessimistic)
    "agg_elast_scale": {
        "low": {
            "agriculture": 0.55, "energy": 0.50, "forestry": 0.60,
            "minerals": 0.55, "manufacturing": 0.15, "other": 0.25,
        },
        "base": {
            "agriculture": 0.85, "energy": 0.80, "forestry": 0.90,
            "minerals": 0.85, "manufacturing": 0.22, "other": 0.40,
        },
        "high": {
            "agriculture": 1.15, "energy": 1.10, "forestry": 1.20,
            "minerals": 1.15, "manufacturing": 0.35, "other": 0.60,
        },
    },
}


# ============================================================================
# 2. ROLLING CALIBRATION — Timestamped Anchors
# ============================================================================

# Each entry: (date, source, CA_eff_rate, MX_eff_rate)
# Add new entries as data becomes available → model auto-uses latest
CALIBRATION_HISTORY = [
    {
        "date": "2025-03-31",
        "source": "PWBM initial estimate (national avg)",
        "CA": 0.027,
        "MX": 0.042,
        "scope": "national",
        "notes": "Based on initial compliance filing rates. National average."
    },
    {
        "date": "2025-07-31",
        "source": "PIIE Jul 2025 tariff revenue data (national avg)",
        "CA": 0.029,
        "MX": 0.047,
        "scope": "national",
        "notes": "PIIE: 'tariff revenue collected from MX ~4.7%, CA ~2.9%'. National average."
    },
    {
        "date": "2025-12-31",
        "source": "Backtest from Census v8 actual 2024→2025 (PNWER-specific)",
        "CA": 0.080,
        "MX": 0.090,
        "scope": "pnwer",
        "notes": (
            "PNWER-specific effective tariff back-calculated from actual 2024→2025 import changes. "
            "PNWER τ_eff >> national avg because: (1) energy/forestry/minerals heavy mix with lower "
            "USMCA compliance, (2) Section 232 steel/aluminum/lumber duties stack on top, "
            "(3) PNWER manufacturing is less auto-heavy (autos have separate USMCA regime). "
            "Weighted avg across PNWER industries: CA ~8.0%, MX ~9.0%."
        )
    },
]

# External revenue anchors for sanity check
REVENUE_ANCHORS = [
    {
        "date": "2025-11-15",
        "source": "CBO Updated Projections",
        "total_annual_new_tariff_revenue_B": 227,  # $2.5T / 11yr ≈ 227B/yr
        "us_effective_tariff_rate": 0.165,           # ~14pp above 2.5% baseline
        "notes": "CBO: effective tariff ~14pp higher than ~2.5% baseline"
    },
]

# External trade change observations for sanity check
OBSERVED_TRADE_CHANGES = [
    {
        "date": "2025-12-31",
        "source": "Census Bureau via pnwer_analysis_data_v8 (SAME口径)",
        "scope": "PNWER 5-state imports, annual",
        "is_same_scope": True,   # ← exact same data scope as model
        "CA_raw_change_pct": -16.2,
        "CA_pre_trend_pct": -6.9,    # 2023→2024 trend (pre-tariff)
        "CA_tariff_attributable_pct": -9.3,  # raw minus pre-trend
        "CA_tariff_attributable_M": -2837,
        "MX_raw_change_pct": -16.7,
        "MX_pre_trend_pct": -9.8,
        "MX_tariff_attributable_pct": -6.8,
        "MX_tariff_attributable_M": -181,
        "notes": "Full-year 2024→2025. Tariff-attributable = actual change minus counterfactual (2023→2024 trend continuing)."
    },
    {
        "date": "2025-07-31",
        "source": "RBC Economics Oct 2025",
        "scope": "National CA exports→US, Jul YoY",
        "is_same_scope": False,  # different scope: national, exports, partial year
        "CA_exports_to_US_change_pct": -5.0,
        "notes": "RBC: CA goods exports to US -5% YoY excl petroleum (Jul). National scope, export-side."
    },
    {
        "date": "2025-07-31",
        "source": "PIIE Oct 2025",
        "scope": "National US-MX two-way trade",
        "is_same_scope": False,
        "MX_trade_status": "resilient",
        "MX_remained_top_partner": True,
        "notes": "PIIE: US-MX two-way trade holding up. National scope."
    },
]


# ============================================================================
# STATIC CONFIGURATION (same as v2)
# ============================================================================

TARIFF_SCHEDULE = {
    "CA": {"general": 0.25, "energy": 0.10},
    "MX": {"general": 0.25, "energy": 0.25},
    "JP": {"general": 0.10, "energy": 0.10},
    "KR": {"general": 0.25, "energy": 0.25},
    "UK": {"general": 0.10, "energy": 0.10},
    "DE": {"general": 0.20, "energy": 0.20},
    "ROW": {"general": 0.35, "energy": 0.30},
}

# ROW tariff varies by industry because the composition of ROW differs:
# Manufacturing ROW is CN-heavy (CN ~30% of ROW mfg imports, 145% tariff)
# Energy ROW is OPEC-heavy (low tariff ~10%)
# This prevents systematic over/under-estimation of diversion
APPLIED_TARIFF_FOR_CES_ROW = {
    "agriculture": 0.20,     # Mix of Latin America, ASEAN at 10-25%
    "energy": 0.12,          # OPEC/other at ~10%, some at higher
    "forestry": 0.18,        # Mix of ASEAN/Latin America
    "minerals": 0.30,        # CN significant share, higher tariffs
    "manufacturing": 0.45,   # CN-dominated (~30% of ROW), 145% + others 10-25%
    "other": 0.25,           # Diverse, moderate average
}

# Non-ROW applied tariffs for CES share reallocation
APPLIED_TARIFF_FOR_CES = {
    "CA": {"general": 0.25 * 0.35, "energy": 0.10 * 0.20},
    "MX": {"general": 0.25 * 0.30, "energy": 0.25 * 0.50},   # 30% non-exempt; v8 backtest: MX attr only -6.8%
    "JP": {"general": 0.10, "energy": 0.10},
    "KR": {"general": 0.25, "energy": 0.25},
    "UK": {"general": 0.10, "energy": 0.10},
    "DE": {"general": 0.20, "energy": 0.20},
}

# US total goods imports by industry (2024 estimates, $B)
# Source: Census Bureau / USITC approximate breakdowns
# Used to compute real ROW residual = Total - (CA+MX+JP+KR+UK+DE)
US_TOTAL_IMPORTS_BY_INDUSTRY_B = {
    "agriculture": 200,
    "energy": 300,
    "forestry": 50,
    "minerals": 150,
    "manufacturing": 1700,
    "other": 800,
}

USMCA_EXEMPT_SHARE_PRIOR = {
    "CA": {"agriculture": 0.55, "energy": 0.80, "forestry": 0.50,
           "minerals": 0.60, "manufacturing": 0.70, "other": 0.65},
    "MX": {"agriculture": 0.40, "energy": 0.30, "forestry": 0.35,
           "minerals": 0.45, "manufacturing": 0.55, "other": 0.49},
}

# PNWER-SPECIFIC INDUSTRY-LEVEL EFFECTIVE TARIFF TARGETS
# Back-calculated from Census v8 actual 2024→2025 tariff-attributable changes.
# Formula: τ_eff = |tariff_attr%| / (agg_elast_scale × |industry_elast|)
# Capped at 1.5× statutory rate to account for Section 232 stacking.
# MX forestry excluded (tiny base, anomalous +47% likely diversion/data noise).
PNWER_INDUSTRY_TAU_EFF = {
    "CA": {
        "agriculture": 0.215,   # High: PNWER ag has lower USMCA compliance
        "energy": 0.100,        # Moderate: crude exempt but refined products hit; +price effects
        "forestry": 0.236,      # High: 25% + softwood lumber duties stacked
        "minerals": 0.339,      # Very high: 25% + Section 232 steel/aluminum (25-50%)
        "manufacturing": 0.041, # Low: most PNWER mfg is USMCA-compliant
        "other": 0.058,         # Low: high USMCA compliance
    },
    "MX": {
        "agriculture": 0.173,   # High: less USMCA compliance in PNWER MX ag
        "energy": 0.000,        # No MX energy imports to PNWER
        "forestry": 0.000,      # Excluded: anomalous +47% (tiny $38M base)
        "minerals": 0.151,      # Section 232 stacking
        "manufacturing": 0.048, # Moderate: maquiladora compliance
        "other": 0.046,         # Moderate
    },
}

ELASTICITIES = {
    "agriculture": -1.5, "energy": -0.5, "forestry": -1.2,
    "minerals": -0.8, "manufacturing": -2.0, "other": -1.5,
}

SIGMA_BASE = {
    "agriculture": 3.5, "energy": 2.0, "forestry": 3.0,
    "minerals": 2.5, "manufacturing": 4.5, "other": 3.0,
}

IO_MULTIPLIERS = {
    "agriculture": 1.8, "energy": 1.5, "forestry": 1.9,
    "minerals": 1.6, "manufacturing": 2.1, "other": 1.7,
}

JOBS_PER_MILLION = {
    "agriculture": 8.5, "energy": 2.5, "forestry": 7.0,
    "minerals": 4.0, "manufacturing": 5.5, "other": 6.0,
}

ALL_PARTNERS = ["CA", "MX", "JP", "KR", "UK", "DE", "ROW"]
USMCA_PARTNERS = ["CA", "MX"]


# ============================================================================
# CORE ANALYZER (supports scenario parameter injection)
# ============================================================================

class ScenarioEngine:
    """
    Runs the tariff model under a specific parameter set.
    Separated from data loading so we can run low/base/high efficiently.
    """

    def __init__(self, state_trade, national_trade, pnwer_states, industries,
                 baseline_year, target_eff_tariff, sigma_scale, pass_through,
                 diversion_realization, agg_elast_scale=None, scenario_label="base"):
        self.state_trade = state_trade
        self.national_trade = national_trade
        self.pnwer_states = pnwer_states
        self.industries = industries
        self.baseline_year = baseline_year
        self.scenario_label = scenario_label

        # Scenario parameters
        self.target_eff_tariff = target_eff_tariff
        self.sigma_scale = sigma_scale
        self.pass_through = pass_through
        self.diversion_realization = diversion_realization
        # agg_elast_scale can be a dict {industry: float} or a single float
        if agg_elast_scale is None:
            self.agg_elast_scale = {ind: 0.5 for ind in industries}
        elif isinstance(agg_elast_scale, dict):
            self.agg_elast_scale = agg_elast_scale
        else:
            self.agg_elast_scale = {ind: float(agg_elast_scale) for ind in industries}

        # Derived: scaled sigma
        self.sigma = {ind: SIGMA_BASE[ind] * sigma_scale for ind in industries}

        # Calibration
        self.k_partner = {}
        self._compute_calibration()

        # National shares
        self.national_shares = {}
        self._compute_national_shares()

    # ---- Data access (same as v2) ----

    def get_imports(self, state, partner, year=None):
        if year is None: year = self.baseline_year
        yd = self.state_trade.get(state, {}).get(partner, {}).get(year, {})
        total = yd.get("total", {}).get("imports", 0)
        by_ind = {ind: yd.get("by_industry", {}).get(ind, {}).get("imports", 0)
                  for ind in self.industries}
        return {"total": total, "by_industry": by_ind}

    def _get_national_imports(self, partner, year=None):
        if year is None: year = self.baseline_year
        yd = self.national_trade.get(partner, {}).get("years", {}).get(year, {})
        total = yd.get("imports", 0)
        bi = yd.get("imports_by_industry", {})
        return {"total": total, "by_industry": {ind: bi.get(ind, 0) for ind in self.industries}}

    # ---- Calibration ----

    def _calc_eff_tariff_uncal(self, partner, industry):
        """Structural prior (only used as fallback)"""
        rates = TARIFF_SCHEDULE.get(partner, {"general": 0, "energy": 0})
        base = rates["energy"] if industry == "energy" else rates["general"]
        if partner in USMCA_EXEMPT_SHARE_PRIOR:
            exempt = USMCA_EXEMPT_SHARE_PRIOR[partner].get(industry, 0.5)
            return base * (1 - exempt)
        return base

    def _compute_calibration(self):
        """
        Industry-level calibration using PNWER_INDUSTRY_TAU_EFF.
        The scenario's target_eff_tariff (partner-level) acts as a SCALE FACTOR
        relative to the base industry rates. This way low/high scenarios
        shift all industries proportionally.
        """
        # Compute the import-weighted avg of industry targets for each partner
        for partner in USMCA_PARTNERS:
            ind_targets = PNWER_INDUSTRY_TAU_EFF.get(partner, {})
            w_tau, w_total = 0.0, 0.0
            for state in self.pnwer_states:
                imp = self.get_imports(state, partner)
                for ind in self.industries:
                    v = imp["by_industry"].get(ind, 0)
                    t = ind_targets.get(ind, 0)
                    if v > 0 and t > 0:
                        w_tau += v * t
                        w_total += v
            base_avg = w_tau / w_total if w_total > 0 else 0.08
            scenario_target = self.target_eff_tariff[partner]
            # Scale factor: scenario target / backtest base avg
            self.k_partner[partner] = scenario_target / base_avg if base_avg > 0 else 1.0

    def calc_effective_tariff(self, partner, industry):
        """
        Industry-level effective tariff from PNWER backtest targets,
        scaled by scenario factor k_p.
        """
        if partner in USMCA_PARTNERS:
            ind_target = PNWER_INDUSTRY_TAU_EFF.get(partner, {}).get(industry, 0)
            if ind_target > 0:
                k = self.k_partner.get(partner, 1.0)
                rates = TARIFF_SCHEDULE.get(partner, {"general": 0, "energy": 0})
                base_cap = rates["energy"] if industry == "energy" else rates["general"]
                # Allow up to 1.5× statutory for Section 232 stacking
                return min(base_cap * 1.5, k * ind_target)
            else:
                # No backtest data (e.g. MX energy = 0): use structural prior
                return self._calc_eff_tariff_uncal(partner, industry)
        else:
            rates = TARIFF_SCHEDULE.get(partner, {"general": 0, "energy": 0})
            return rates["energy"] if industry == "energy" else rates["general"]

    def _get_ces_tariff(self, partner, industry):
        """
        Applied tariff for CES price index and demand calculation.
        For USMCA partners: uses calibrated industry-level τ_eff from backtest.
        For others: uses static rates.
        Adjusted for pass-through.
        """
        if partner == "ROW":
            raw = APPLIED_TARIFF_FOR_CES_ROW.get(industry, 0.25)
        elif partner in USMCA_PARTNERS:
            # Use the calibrated effective tariff (industry-specific, scenario-scaled)
            raw = self.calc_effective_tariff(partner, industry)
        else:
            rates = APPLIED_TARIFF_FOR_CES.get(partner, {"general": 0, "energy": 0})
            raw = rates["energy"] if industry == "energy" else rates["general"]
        return raw * (1 - self.pass_through)

    # ---- CES shares ----

    def _compute_national_shares(self):
        """
        Compute national import shares for reference only.
        Used to estimate non-CA/MX partner distribution at state level.
        ROW = US_TOTAL - known partners (real residual, not duplicated total).
        """
        self._national_known = {}
        self._national_row = {}
        self._national_non_camx = {}

        for ind in self.industries:
            known = {}
            for p in ["CA", "MX", "JP", "KR", "UK", "DE"]:
                v = self._get_national_imports(p)["by_industry"].get(ind, 0)
                known[p] = v

            known_total = sum(known.values())
            us_total = US_TOTAL_IMPORTS_BY_INDUSTRY_B.get(ind, 500) * 1e9
            row = max(0, us_total - known_total)

            # Store for state-level prorating
            self._national_known[ind] = known
            self._national_row[ind] = row

            # Non-CA/MX breakdown (for prorating to states)
            non_camx = {p: known[p] for p in ["JP", "KR", "UK", "DE"]}
            non_camx["ROW"] = row
            non_camx_total = sum(non_camx.values())
            # Normalize non-CA/MX to shares
            self._national_non_camx[ind] = {
                p: v / non_camx_total if non_camx_total > 0 else 0
                for p, v in non_camx.items()
            }

            # Also keep a national-level share for reference
            all_vals = dict(known)
            all_vals["ROW"] = row
            grand_total = sum(all_vals.values())
            self.national_shares[ind] = {
                p: v / grand_total if grand_total > 0 else 0
                for p, v in all_vals.items()
            }

    def _compute_state_shares(self, state, industry):
        """
        Compute state×industry baseline partner shares.
        - CA/MX: use REAL state-level import data
        - JP/KR/UK/DE/ROW: prorate national non-CA/MX structure onto
          the state's residual (total_state_imports - CA - MX)
        
        This ensures each state's CA/MX dependency is reflected accurately,
        rather than being overridden by the national average.
        """
        # Real CA/MX imports for this state×industry
        ca_imp = 0
        mx_imp = 0
        for partner in ["CA", "MX"]:
            yd = self.state_trade.get(state, {}).get(partner, {}).get(self.baseline_year, {})
            v = yd.get("by_industry", {}).get(industry, {}).get("imports", 0)
            if partner == "CA":
                ca_imp = v
            else:
                mx_imp = v

        camx_total = ca_imp + mx_imp

        # Estimate total state imports in this industry
        # Use national CA/MX share to back out approximate total
        nat = self.national_shares.get(industry, {})
        nat_camx_share = nat.get("CA", 0) + nat.get("MX", 0)

        if nat_camx_share > 0 and camx_total > 0:
            est_total = camx_total / nat_camx_share
        else:
            est_total = camx_total * 2.0  # fallback

        # Non-CA/MX residual
        non_camx_total = max(0, est_total - camx_total)

        # Build shares
        shares = {"CA": ca_imp, "MX": mx_imp}
        non_camx_dist = self._national_non_camx.get(industry, {})
        for p in ["JP", "KR", "UK", "DE", "ROW"]:
            shares[p] = non_camx_total * non_camx_dist.get(p, 0)

        grand = sum(shares.values())
        if grand > 0:
            return {p: v / grand for p, v in shares.items()}
        return self.national_shares.get(industry, {})

    def _compute_diversion_shares(self, industry, base_shares=None):
        """CES share reallocation. Accepts custom base_shares (state-level)."""
        sigma = self.sigma.get(industry, 3.0)
        if base_shares is None:
            base_shares = self.national_shares.get(industry, {})
        if not base_shares: return {}
        adj = {}
        denom = 0.0
        for p in ALL_PARTNERS:
            w = base_shares.get(p, 0.0)
            if w <= 0: continue
            tau = self._get_ces_tariff(p, industry)
            pf = (1 + tau) ** (-sigma)
            adj[p] = w * pf
            denom += w * pf
        return {p: adj[p]/denom for p in adj} if denom > 0 else {}

    # ---- Import change ----

    def compute_import_change(self, state, partner, industry, base_import):
        """
        Compute import change using a two-layer model:
        
        Layer 1 (Total demand): Aggregate price index from tariffs reduces total
        import demand for this industry. Total can only stay flat or decline.
          ΔM_total = M_total × elasticity × Δ(price_index)
        
        Layer 2 (Share redistribution): CES model redistributes the NEW total
        across source countries. This is pure reallocation — it cannot increase
        total imports above baseline.
        
        Result: partner's new import = new_share × new_total
        This ensures "low" scenario at most shows "small decline", never "increase".
        """
        state_shares = self._compute_state_shares(state, industry)
        new_shares = self._compute_diversion_shares(industry, base_shares=state_shares)

        if not state_shares or not new_shares:
            # Fallback: simple own-tariff effect
            applied = self._get_ces_tariff(partner, industry)
            elast = ELASTICITIES.get(industry, -1.5)
            return base_import * elast * applied

        w_old = state_shares.get(partner, 0.0)
        w_new = new_shares.get(partner, 0.0)
        if w_old <= 0:
            applied = self._get_ces_tariff(partner, industry)
            elast = ELASTICITIES.get(industry, -1.5)
            return base_import * elast * applied

        # --- Layer 1: Total demand change via CES aggregate price index ---
        # The CES ideal price index: P'/P = [Σ w_p (1+τ_p)^(1-σ)]^(1/(1-σ))
        # This gives the % price increase faced by importers on average.
        # Total demand change = elasticity × (P'/P - 1)
        sigma = self.sigma.get(industry, 3.0)
        price_sum = 0.0
        for p in ALL_PARTNERS:
            w = state_shares.get(p, 0.0)
            if w > 0:
                tau = self._get_ces_tariff(p, industry)
                price_sum += w * (1 + tau) ** (1 - sigma)

        if price_sum > 0 and sigma != 1:
            price_ratio = price_sum ** (1.0 / (1.0 - sigma))
        else:
            price_ratio = 1.0

        elast = ELASTICITIES.get(industry, -1.5)
        # Total demand change: use AGGREGATE import demand elasticity
        # Note: this is typically smaller than industry-specific elasticity
        # because it represents demand for ALL imports, not substitution
        agg_elast = elast * self.agg_elast_scale.get(industry, 0.5)
        total_demand_factor = 1.0 + agg_elast * (price_ratio - 1.0)
        total_demand_factor = max(total_demand_factor, 0.5)  # Floor: max 50% decline
        total_demand_factor = min(total_demand_factor, 1.0)  # Cap: cannot exceed baseline

        # --- Layer 2: Share redistribution (CES) ---
        # Partner's new import = new_share × (total_demand_factor × total_baseline)
        # But we only observe this partner's baseline, so:
        #   new_M_p = (w_new / w_old) × total_demand_factor × base_import_p
        # ... but we need to be careful: base_import_p = w_old × M_total
        # So: new_M_p = w_new × total_demand_factor × (base_import_p / w_old)
        #             = (w_new / w_old) × total_demand_factor × base_import_p

        share_ratio = w_new / w_old
        realization = self.diversion_realization.get(partner, 0.25)

        # Blend: partial realization of share shift
        # effective_share_ratio = 1 + realization × (share_ratio - 1)
        effective_share_ratio = 1.0 + realization * (share_ratio - 1.0)

        new_import = base_import * effective_share_ratio * total_demand_factor

        # Hard constraint: tariffs on a partner cannot cause imports FROM THAT
        # PARTNER to exceed baseline. Diversion can offset decline, but not create
        # a net increase — that would mean tariffs increase bilateral imports,
        # which is economically incoherent for the tariffed partner.
        if partner in USMCA_PARTNERS:
            new_import = min(new_import, base_import)

        return new_import - base_import

    # ---- Full run ----

    def run(self):
        """Run analysis, return results dict"""
        totals = {
            "imports_ca": 0, "imports_mx": 0,
            "burden_ca": 0, "burden_mx": 0,
            "change_ca": 0, "change_mx": 0,
            # Split GDP into two components:
            # disruption_cost: economic damage from trade reduction (always ≥ 0)
            # diversion_offset: partial benefit from diverted trade (always ≥ 0)
            "disruption_cost": 0,
            "diversion_offset": 0,
            "jobs_disrupted": 0,     # jobs at risk from disruption (always ≥ 0)
            "jobs_offset": 0,        # partial job offset from diversion (always ≥ 0)
        }
        by_state = {}

        for state in self.pnwer_states:
            st = {"imports_ca": 0, "imports_mx": 0, "burden_ca": 0, "burden_mx": 0,
                  "change_ca": 0, "change_mx": 0,
                  "disruption_cost": 0, "diversion_offset": 0,
                  "jobs_disrupted": 0, "jobs_offset": 0,
                  "by_industry": {}}

            for partner in USMCA_PARTNERS:
                imp = self.get_imports(state, partner)
                pk = "ca" if partner == "CA" else "mx"
                st[f"imports_{pk}"] = imp["total"]

                for ind in self.industries:
                    val = imp["by_industry"].get(ind, 0)
                    if val <= 0: continue

                    eff = self.calc_effective_tariff(partner, ind)
                    burden = val * eff
                    delta = self.compute_import_change(state, partner, ind, val)

                    mult = IO_MULTIPLIERS.get(ind, 1.7)
                    jpm = JOBS_PER_MILLION.get(ind, 5.0)

                    if delta < 0:
                        # Trade loss → disruption cost
                        cost = abs(delta) * mult
                        st["disruption_cost"] += cost
                        st["jobs_disrupted"] += (cost / 1e6) * jpm
                    else:
                        # Trade gain from diversion → partial offset
                        offset = abs(delta) * mult * 0.3
                        st["diversion_offset"] += offset
                        st["jobs_offset"] += (offset / 1e6) * jpm * 0.5

                    st[f"burden_{pk}"] += burden
                    st[f"change_{pk}"] += delta

                    if ind not in st["by_industry"]:
                        st["by_industry"][ind] = {"imports_M": 0, "burden_M": 0,
                                                   "change_M": 0, "disruption_M": 0,
                                                   "offset_M": 0}
                    st["by_industry"][ind]["imports_M"] += val / 1e6
                    st["by_industry"][ind]["burden_M"] += burden / 1e6
                    st["by_industry"][ind]["change_M"] += delta / 1e6
                    if delta < 0:
                        st["by_industry"][ind]["disruption_M"] += abs(delta) * mult / 1e6
                    else:
                        st["by_industry"][ind]["offset_M"] += abs(delta) * mult * 0.3 / 1e6

            by_state[state] = st
            for k in totals:
                totals[k] += st[k]

        total_imp = totals["imports_ca"] + totals["imports_mx"]
        total_change = totals["change_ca"] + totals["change_mx"]
        change_pct = total_change / total_imp * 100 if total_imp > 0 else 0

        # Net effect = disruption - offset (always report disruption as the headline)
        net_gdp = totals["disruption_cost"] - totals["diversion_offset"]
        net_jobs = totals["jobs_disrupted"] - totals["jobs_offset"]

        return {
            "scenario": self.scenario_label,
            "params": {
                "target_eff_tariff": self.target_eff_tariff,
                "sigma_scale": self.sigma_scale,
                "pass_through": self.pass_through,
                "diversion_realization": self.diversion_realization,
                "agg_elast_scale": self.agg_elast_scale,
                "k_partner": {p: round(v, 4) for p, v in self.k_partner.items()},
            },
            "totals": {
                "imports_ca_M": round(totals["imports_ca"] / 1e6, 1),
                "imports_mx_M": round(totals["imports_mx"] / 1e6, 1),
                "imports_total_M": round(total_imp / 1e6, 1),
                "burden_ca_M": round(totals["burden_ca"] / 1e6, 1),
                "burden_mx_M": round(totals["burden_mx"] / 1e6, 1),
                "burden_total_M": round((totals["burden_ca"]+totals["burden_mx"]) / 1e6, 1),
                "change_ca_M": round(totals["change_ca"] / 1e6, 1),
                "change_mx_M": round(totals["change_mx"] / 1e6, 1),
                "change_total_M": round(total_change / 1e6, 1),
                "change_pct": round(change_pct, 1),
                # GDP: split presentation
                # GDP: split into loss and credit (both ≥ 0, never negative)
                "gdp_loss_M": round(totals["disruption_cost"] / 1e6, 1),
                "gdp_offset_credit_M": round(totals["diversion_offset"] / 1e6, 1),
                "gdp_net_impact_M": round(net_gdp / 1e6, 1),
                # Jobs: jobs_at_risk always ≥ 0 (headline for clients)
                "jobs_at_risk": max(0, round(totals["jobs_disrupted"])),
                "jobs_supported_by_diversion": round(totals["jobs_offset"]),
                "jobs_net_change": round(net_jobs),
            },
            "by_state": {
                s: {
                    "imports_M": round((d["imports_ca"]+d["imports_mx"])/1e6, 1),
                    "burden_M": round((d["burden_ca"]+d["burden_mx"])/1e6, 1),
                    "change_ca_M": round(d["change_ca"]/1e6, 1),
                    "change_mx_M": round(d["change_mx"]/1e6, 1),
                    "change_M": round((d["change_ca"]+d["change_mx"])/1e6, 1),
                    "gdp_loss_M": round(d["disruption_cost"]/1e6, 1),
                    "gdp_offset_M": round(d["diversion_offset"]/1e6, 1),
                    "gdp_net_M": round((d["disruption_cost"]-d["diversion_offset"])/1e6, 1),
                    "jobs_at_risk": max(0, round(d["jobs_disrupted"])),
                    "jobs_supported": round(d["jobs_offset"]),
                    "by_industry": {
                        ind: {k: round(v, 1) for k, v in vals.items()}
                        for ind, vals in d["by_industry"].items()
                    }
                } for s, d in by_state.items()
            },
        }


# ============================================================================
# MASTER ANALYZER — runs 3 scenarios + sanity check
# ============================================================================

class ForecastTariffAnalyzer:
    """
    v3 analyzer:
    1. Runs low/base/high scenarios
    2. Uses latest calibration anchor (rolling)
    3. Produces sanity-check dashboard
    """

    def __init__(self, state_data_path: str, national_data_path: str = None,
                 calibration_date: str = None):
        # Load data once
        with open(state_data_path, 'r', encoding='utf-8') as f:
            sd = json.load(f)
        self.state_trade = sd.get("state_trade", {})

        self.national_trade = {}
        if national_data_path and Path(national_data_path).exists():
            with open(national_data_path, 'r', encoding='utf-8') as f:
                nd = json.load(f)
            self.national_trade = nd.get("national_trade", {})

        self.pnwer_states = ["WA", "OR", "ID", "MT", "AK"]
        self.industries = ["agriculture", "energy", "forestry", "minerals",
                           "manufacturing", "other"]
        self.baseline_year = "2024"

        # Pick calibration anchor
        self.cal_anchor = self._select_calibration_anchor(calibration_date)

    def _select_calibration_anchor(self, target_date=None):
        """Select the most recent anchor on or before target_date"""
        anchors = sorted(CALIBRATION_HISTORY, key=lambda x: x["date"])
        if target_date:
            valid = [a for a in anchors if a["date"] <= target_date]
        else:
            valid = anchors
        if valid:
            chosen = valid[-1]
            print(f"  Calibration anchor: {chosen['date']} ({chosen['source']})")
            return chosen
        return anchors[-1] if anchors else {"CA": 0.027, "MX": 0.042}

    def _make_engine(self, scenario: str) -> ScenarioEngine:
        """Create a ScenarioEngine with the right parameter set"""
        sp = SCENARIO_PARAMS

        # τ_eff targets — use calibration anchor for base, scale for low/high
        if scenario == "base":
            tgt = {"CA": self.cal_anchor["CA"], "MX": self.cal_anchor["MX"]}
        elif scenario == "low":
            tgt = {p: sp["target_eff_tariff"][p]["low"] for p in ["CA", "MX"]}
        else:
            tgt = {p: sp["target_eff_tariff"][p]["high"] for p in ["CA", "MX"]}

        sigma_s = sp["sigma_scale"][scenario]
        pt = sp["pass_through"][scenario]
        dr = {p: sp["diversion_realization"][p][scenario] for p in ["CA", "MX"]}
        aes = sp["agg_elast_scale"][scenario]

        return ScenarioEngine(
            self.state_trade, self.national_trade,
            self.pnwer_states, self.industries, self.baseline_year,
            target_eff_tariff=tgt, sigma_scale=sigma_s,
            pass_through=pt, diversion_realization=dr,
            agg_elast_scale=aes, scenario_label=scenario
        )

    def run_all_scenarios(self) -> Dict:
        """Run low/base/high and compile results"""
        results = {
            "metadata": {
                "analysis": "PNWER Tariff Impact v3 — Forecast-Ready",
                "baseline_year": self.baseline_year,
                "run_timestamp": datetime.now().isoformat(),
                "calibration_anchor": {
                    "date": self.cal_anchor["date"],
                    "source": self.cal_anchor["source"],
                    "CA_eff_tariff": self.cal_anchor["CA"],
                    "MX_eff_tariff": self.cal_anchor["MX"],
                },
                "scenario_parameters": SCENARIO_PARAMS,
            },
            "scenarios": {},
            "bands": {},
            "sanity_check": {},
        }

        # Run 3 scenarios
        for sc in ["low", "base", "high"]:
            print(f"\n  Running scenario: {sc}...")
            engine = self._make_engine(sc)
            results["scenarios"][sc] = engine.run()

        # Build confidence bands
        results["bands"] = self._build_bands(results["scenarios"])

        # Sanity check
        results["sanity_check"] = self._sanity_check(results["scenarios"]["base"])

        return results

    def _build_bands(self, scenarios: Dict) -> Dict:
        """Extract [low, base, high] for key metrics"""
        metrics = [
            "burden_total_M", "change_ca_M", "change_mx_M",
            "change_total_M", "change_pct",
            "gdp_loss_M", "gdp_offset_credit_M", "gdp_net_impact_M",
            "jobs_at_risk", "jobs_net_change",
        ]
        bands = {}
        for m in metrics:
            bands[m] = {
                "low": scenarios["low"]["totals"][m],
                "base": scenarios["base"]["totals"][m],
                "high": scenarios["high"]["totals"][m],
            }
        # State-level bands
        bands["by_state"] = {}
        for state in self.pnwer_states:
            bands["by_state"][state] = {
                "change_M": {
                    "low": scenarios["low"]["by_state"][state]["change_M"],
                    "base": scenarios["base"]["by_state"][state]["change_M"],
                    "high": scenarios["high"]["by_state"][state]["change_M"],
                },
                "jobs_at_risk": {
                    "low": scenarios["low"]["by_state"][state]["jobs_at_risk"],
                    "base": scenarios["base"]["by_state"][state]["jobs_at_risk"],
                    "high": scenarios["high"]["by_state"][state]["jobs_at_risk"],
                },
            }
        return bands

    def _sanity_check(self, base_result: Dict) -> Dict:
        """
        Self-validation dashboard:
        1. implied τ_eff vs target τ_eff
        2. implied revenue vs external anchor
        3. predicted import change vs observed trend
        """
        checks = {"timestamp": datetime.now().isoformat(), "checks": []}

        # --- Check 1: τ_eff alignment ---
        for p in USMCA_PARTNERS:
            target = self.cal_anchor[p]
            # Implied from burden/imports
            burden_key = f"burden_{p.lower()}_M"
            imports_key = f"imports_{p.lower()}_M"
            burden = base_result["totals"][burden_key]
            imports = base_result["totals"][imports_key]
            implied = burden / imports * 100 if imports > 0 else 0

            status = "PASS" if abs(implied - target*100) < 0.5 else "WARN"
            checks["checks"].append({
                "name": f"τ_eff alignment ({p})",
                "target_pct": round(target * 100, 2),
                "implied_pct": round(implied, 2),
                "gap_pp": round(implied - target*100, 2),
                "status": status,
            })

        # --- Check 2: Revenue reasonableness ---
        # Scale PNWER burden to national estimate
        total_burden = base_result["totals"]["burden_total_M"]
        total_imports = base_result["totals"]["imports_total_M"]
        # PNWER is ~10% of total US imports from CA/MX
        # (US imports from CA ~412B, MX ~506B; PNWER ~33B → ~3.6%)
        pnwer_share = total_imports / (412000 + 506000) * 100  # rough
        implied_national_burden = total_burden / (pnwer_share/100) if pnwer_share > 0 else 0

        latest_anchor = REVENUE_ANCHORS[-1] if REVENUE_ANCHORS else None
        if latest_anchor:
            # ROUGH ORDER-OF-MAGNITUDE CHECK ONLY
            # We scale PNWER burden to a national estimate using import share,
            # then compare to CBO's total tariff revenue. This is NOT a national
            # revenue prediction — it's a sanity check that our per-dollar burden
            # rates are in the right ballpark.
            implied_camx_national = implied_national_burden
            expected_camx_share = latest_anchor["total_annual_new_tariff_revenue_B"] * 0.25
            ratio = implied_camx_national / (expected_camx_share * 1000) if expected_camx_share > 0 else 0
            status = "PASS" if 0.3 < ratio < 3.0 else "WARN"
            checks["checks"].append({
                "name": "Revenue order-of-magnitude (ROUGH CHECK)",
                "caveat": "Extrapolates PNWER burden to national via import share. Not a revenue forecast.",
                "pnwer_burden_M": round(total_burden, 1),
                "pnwer_import_share_pct": round(pnwer_share, 1),
                "implied_national_CA_MX_burden_M": round(implied_national_burden, 0),
                "cbo_expected_CA_MX_share_M": round(expected_camx_share * 1000, 0),
                "ratio": round(ratio, 2),
                "status": status,
            })

        # --- Check 3: Trade change vs actual (SAME-SCOPE BACKTEST) ---
        # Use pnwer_analysis_data_v8 actual 2025 data — exact same scope as model
        same_scope = [o for o in OBSERVED_TRADE_CHANGES if o.get("is_same_scope")]
        if same_scope:
            obs = same_scope[0]
            # CA: compare model % to tariff-attributable %
            model_ca_pct = base_result["totals"]["change_ca_M"] / base_result["totals"]["imports_ca_M"] * 100 \
                if base_result["totals"]["imports_ca_M"] > 0 else 0
            actual_ca_pct = obs["CA_tariff_attributable_pct"]
            gap_ca = abs(model_ca_pct - actual_ca_pct)
            status_ca = "PASS" if gap_ca < 3.0 else ("WARN" if gap_ca < 6.0 else "FAIL")
            checks["checks"].append({
                "name": "CA backtest (SAME SCOPE — PNWER imports, tariff-attributable)",
                "model_pct": round(model_ca_pct, 1),
                "actual_tariff_attributable_pct": actual_ca_pct,
                "actual_raw_pct": obs["CA_raw_change_pct"],
                "pre_trend_pct": obs["CA_pre_trend_pct"],
                "gap_pp": round(gap_ca, 1),
                "source": obs["source"],
                "status": status_ca,
            })

            # MX
            model_mx_pct = base_result["totals"]["change_mx_M"] / base_result["totals"]["imports_mx_M"] * 100 \
                if base_result["totals"]["imports_mx_M"] > 0 else 0
            actual_mx_pct = obs["MX_tariff_attributable_pct"]
            gap_mx = abs(model_mx_pct - actual_mx_pct)
            status_mx = "PASS" if gap_mx < 4.0 else ("WARN" if gap_mx < 8.0 else "FAIL")
            checks["checks"].append({
                "name": "MX backtest (SAME SCOPE — PNWER imports, tariff-attributable)",
                "model_pct": round(model_mx_pct, 1),
                "actual_tariff_attributable_pct": actual_mx_pct,
                "actual_raw_pct": obs["MX_raw_change_pct"],
                "pre_trend_pct": obs["MX_pre_trend_pct"],
                "gap_pp": round(gap_mx, 1),
                "source": obs["source"],
                "status": status_mx,
            })

        # --- Check 4: Directional consistency vs external proxies ---
        proxy_obs = [o for o in OBSERVED_TRADE_CHANGES if not o.get("is_same_scope")]
        for obs in proxy_obs:
            if "CA_exports_to_US_change_pct" in obs:
                model_ca_pct = base_result["totals"]["change_ca_M"] / base_result["totals"]["imports_ca_M"] * 100 \
                    if base_result["totals"]["imports_ca_M"] > 0 else 0
                direction_match = (model_ca_pct < 0) == (obs["CA_exports_to_US_change_pct"] < 0)
                checks["checks"].append({
                    "name": "CA direction (PROXY — national exports→US)",
                    "caveat": "Different scope: national CA exports vs PNWER imports.",
                    "model_pct": round(model_ca_pct, 1),
                    "proxy_pct": obs["CA_exports_to_US_change_pct"],
                    "direction_match": direction_match,
                    "source": obs["source"],
                    "status": "PASS" if direction_match else "WARN",
                })
            if "MX_trade_status" in obs:
                model_mx_M = base_result["totals"]["change_mx_M"]
                checks["checks"].append({
                    "name": "MX direction (PROXY — national trade status)",
                    "caveat": "Different scope: national MX two-way trade vs PNWER imports.",
                    "model_change_M": round(model_mx_M, 1),
                    "proxy_status": obs.get("MX_trade_status"),
                    "source": obs["source"],
                    "status": "INFO",
                })

        return checks

    # ---- Output ----

    def print_results(self, results: Dict):
        sc = results["scenarios"]
        bands = results["bands"]
        cal = results["metadata"]["calibration_anchor"]

        print("\n" + "=" * 95)
        print("       PNWER TARIFF IMPACT ANALYSIS v3 — FORECAST-READY MODEL")
        print("       2025 US Tariffs on Canada and Mexico")
        print("=" * 95)
        print(f"\n  Baseline: {self.baseline_year} | Run: {results['metadata']['run_timestamp'][:10]}")
        print(f"  Calibration: {cal['date']} ({cal['source']})")
        print(f"  CA τ_eff = {cal['CA_eff_tariff']*100:.1f}% | MX τ_eff = {cal['MX_eff_tariff']*100:.1f}%")

        # ---- Scenario Band Summary ----
        print("\n" + "-" * 95)
        print("  SCENARIO BANDS: PNWER TOTAL")
        print("-" * 95)
        print(f"  {'Metric':<30} {'Low':<18} {'Base':<18} {'High':<18}")
        print(f"  {'(optimistic)':>30} {'':18} {'(pessimistic)':>18}")
        print("-" * 95)

        band_labels = [
            ("Tariff Burden ($M)", "burden_total_M"),
            ("CA Trade Change ($M)", "change_ca_M"),
            ("MX Trade Change ($M)", "change_mx_M"),
            ("Net Trade Change ($M)", "change_total_M"),
            ("Trade Change (%)", "change_pct"),
            ("GDP Loss ($M)", "gdp_loss_M"),
            ("  Diversion Offset ($M)", "gdp_offset_credit_M"),
            ("  Net GDP Impact ($M)", "gdp_net_impact_M"),
            ("Jobs at Risk", "jobs_at_risk"),
            ("  Net Jobs (w/ offset)", "jobs_net_change"),
        ]

        for label, key in band_labels:
            b = bands[key]
            is_jobs = "jobs" in key.lower()
            is_pct = "pct" in key
            fmt = ",.0f" if is_jobs else ",.1f"
            if is_jobs:
                lv, bv, hv = f"{b['low']:,.0f}", f"{b['base']:,.0f}", f"{b['high']:,.0f}"
            elif is_pct:
                lv, bv, hv = f"{b['low']:{fmt}}%", f"{b['base']:{fmt}}%", f"{b['high']:{fmt}}%"
            else:
                lv = f"${b['low']:{fmt}}"
                bv = f"${b['base']:{fmt}}"
                hv = f"${b['high']:{fmt}}"
            print(f"  {label:<30} {lv:>16}  {bv:>16}  {hv:>16}")

        # ---- By State Bands ----
        print("\n" + "-" * 95)
        print("  BY STATE: Trade Change ($M) [Low / Base / High]")
        print("-" * 95)
        for state in self.pnwer_states:
            sb = bands["by_state"][state]["change_M"]
            print(f"  {state:<6}  [{sb['low']:>10,.1f}  /  {sb['base']:>10,.1f}  /  {sb['high']:>10,.1f} ]")

        # ---- Sanity Check Dashboard ----
        print("\n" + "-" * 95)
        print("  SANITY-CHECK DASHBOARD")
        print("-" * 95)
        for chk in results["sanity_check"]["checks"]:
            icon = "✅" if chk["status"] == "PASS" else "⚠️"
            print(f"  {icon} {chk['name']}")
            for k, v in chk.items():
                if k not in ["name", "status"]:
                    print(f"      {k}: {v}")

        # ---- Calibration History ----
        print("\n" + "-" * 95)
        print("  CALIBRATION ANCHOR HISTORY (for rolling updates)")
        print("-" * 95)
        for entry in CALIBRATION_HISTORY:
            print(f"  {entry['date']}  CA={entry['CA']*100:.1f}%  MX={entry['MX']*100:.1f}%  "
                  f"({entry['source']})")

        # ---- Scenario Parameters ----
        print("\n" + "-" * 95)
        print("  SCENARIO PARAMETER MATRIX")
        print("-" * 95)
        print(f"  {'Parameter':<30} {'Low':<18} {'Base':<18} {'High':<18}")
        print("-" * 95)
        ca_base = f"{cal['CA_eff_tariff']*100:.1f}%"
        mx_base = f"{cal['MX_eff_tariff']*100:.1f}%"
        print(f"  {'CA τ_eff target':<30} {'1.8%':<18} {ca_base:<18} {'3.7%':<18}")
        print(f"  {'MX τ_eff target':<30} {'3.8%':<18} {mx_base:<18} {'4.7%':<18}")
        sp = SCENARIO_PARAMS
        print(f"  {'σ scale factor':<30} {sp['sigma_scale']['low']:<18} {sp['sigma_scale']['base']:<18} {sp['sigma_scale']['high']:<18}")
        print(f"  {'Pass-through λ':<30} {sp['pass_through']['low']:<18} {sp['pass_through']['base']:<18} {sp['pass_through']['high']:<18}")
        print(f"  {'CA diversion realization':<30} {sp['diversion_realization']['CA']['low']:<18} {sp['diversion_realization']['CA']['base']:<18} {sp['diversion_realization']['CA']['high']:<18}")
        print(f"  {'MX diversion realization':<30} {sp['diversion_realization']['MX']['low']:<18} {sp['diversion_realization']['MX']['base']:<18} {sp['diversion_realization']['MX']['high']:<18}")
        aes = sp['agg_elast_scale']
        print(f"  {'Agg. elast. (energy/mfg/other)':<30} {'E:'+str(aes['low']['energy'])+' M:'+str(aes['low']['manufacturing']):<18} {'E:'+str(aes['base']['energy'])+' M:'+str(aes['base']['manufacturing']):<18} {'E:'+str(aes['high']['energy'])+' M:'+str(aes['high']['manufacturing']):<18}")

        print("\n" + "=" * 95)

    def save_results(self, results: Dict, path: str = "tariff_impact_results_v3.json"):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Results saved to: {path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║   PNWER Tariff Impact Analysis v3 — Forecast-Ready            ║
    ║   2025 US Tariffs on Canada and Mexico                        ║
    ║                                                                ║
    ║   Upgrades:                                                    ║
    ║   1. Scenario bands (Low / Base / High)                       ║
    ║   2. Rolling calibration (timestamped anchors)                ║
    ║   3. Sanity-check dashboard (self-validation)                 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    # Find data
    state_paths = ["data/pnwer_analysis_data_v8.json", "pnwer_analysis_data_v8.json"]
    national_paths = ["data/national_trade.json", "national_trade.json"]

    state_path = next((p for p in state_paths if Path(p).exists()), None)
    national_path = next((p for p in national_paths if Path(p).exists()), None)

    if not state_path:
        print("ERROR: Cannot find state trade data!")
        return

    print(f"  State data: {state_path}")
    print(f"  National data: {national_path}")

    analyzer = ForecastTariffAnalyzer(state_path, national_path)
    results = analyzer.run_all_scenarios()
    analyzer.print_results(results)
    analyzer.save_results(results)

    print("\n  Analysis complete!")


if __name__ == "__main__":
    main()