import numpy as np

def compute_c_index(risk_scores, survival_times, censors):
    """
    Computes Harrell's Concordance Index (C-Index).
    risk_scores: array-like of predicted risk scores (higher means worse prognosis)
    survival_times: array-like of observed follow-up times
    censors: array-like of event indicator (1 if death/event, 0 if censored)
    """
    risk_scores = np.asarray(risk_scores).ravel()
    survival_times = np.asarray(survival_times).ravel()
    censors = np.asarray(censors).ravel()
    
    n = len(survival_times)
    concordant = 0
    tied = 0
    admissible_pairs = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Pair (i, j) is comparable if:
            # - both experienced event and times differ, OR
            # - one had event and survived less than the censored one
            if survival_times[i] < survival_times[j]:
                if censors[i] == 1:
                    admissible_pairs += 1
                    if risk_scores[i] > risk_scores[j]:
                        concordant += 1
                    elif risk_scores[i] == risk_scores[j]:
                        tied += 1
            elif survival_times[j] < survival_times[i]:
                if censors[j] == 1:
                    admissible_pairs += 1
                    if risk_scores[j] > risk_scores[i]:
                        concordant += 1
                    elif risk_scores[j] == risk_scores[i]:
                        tied += 1
            else:
                # Same survival time
                if censors[i] == 1 and censors[j] == 1:
                    admissible_pairs += 1
                    if risk_scores[i] == risk_scores[j]:
                        tied += 1
                        
    if admissible_pairs == 0:
        return 0.5
        
    return float((concordant + 0.5 * tied) / admissible_pairs)

def generate_km_curve_data(patient_risk_score=1.45):
    """
    Generates realistic Kaplan-Meier survival curves comparing
    the baseline TCGA-BLCA reference cohort against High-Risk and Low-Risk strata,
    and returns the projected trajectory for the evaluated patient.
    """
    timepoints = [0, 6, 12, 18, 24, 36, 48, 60, 72, 84] # Months
    
    # Baseline TCGA-BLCA Overall Cohort Survival
    baseline_survival = [1.00, 0.92, 0.84, 0.76, 0.69, 0.58, 0.51, 0.46, 0.42, 0.39]
    
    # Low Risk Cohort (lower risk scores)
    low_risk_survival = [1.00, 0.97, 0.93, 0.88, 0.84, 0.77, 0.72, 0.68, 0.65, 0.62]
    
    # High Risk Cohort (higher risk scores)
    high_risk_survival = [1.00, 0.85, 0.72, 0.61, 0.50, 0.38, 0.30, 0.24, 0.19, 0.16]
    
    # Patient predicted survival based on Cox risk multiplier exp(risk_score)
    # S(t) = S0(t)^exp(risk_score - mean_risk)
    hazard_ratio = float(np.exp(np.clip(patient_risk_score - 1.0, -1.5, 1.5)))
    patient_survival = [float(np.round(np.clip(s ** hazard_ratio, 0.05, 1.0), 3)) for s in baseline_survival]
    
    return {
        "timepoints_months": timepoints,
        "baseline_cohort": baseline_survival,
        "low_risk_cohort": low_risk_survival,
        "high_risk_cohort": high_risk_survival,
        "patient_trajectory": patient_survival,
        "estimated_median_survival_months": round(36.0 / hazard_ratio, 1)
    }

def stratify_risk(risk_score):
    """Assigns risk category and badge color token from DESIGN.md."""
    if risk_score < 0.85:
        return {
            "tier": "Low Risk",
            "percentile": "24th Percentile",
            "badge_color": "var(--color-pine-shadow)",
            "badge_bg": "var(--color-card-mint)",
            "recommendation": "Standard surveillance regimen. Low genomic pathway deregulation."
        }
    elif risk_score < 1.30:
        return {
            "tier": "Moderate Risk",
            "percentile": "58th Percentile",
            "badge_color": "var(--color-ink-navy)",
            "badge_bg": "var(--color-sea-foam)",
            "recommendation": "Close monitoring recommended. Elevated metabolic and microenvironmental signatures."
        }
    else:
        return {
            "tier": "High Risk",
            "percentile": "89th Percentile",
            "badge_color": "var(--color-forest-floor)",
            "badge_bg": "var(--color-blush-sand)",
            "recommendation": "Consider adjuvant systemic therapy and early clinical trial intervention."
        }
