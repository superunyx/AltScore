import math
from datetime import datetime, timedelta

def compute_ratio_features(sms_logs, app_usage, window_start, window_end):
    """
    Computes 8-value feature vector from 30-day window.
    sms_logs: list of dicts with 'timestamp', 'amount', 'type'
    app_usage: dict of {date_str: {'sessions', 'hours_active'}}
    window_start: datetime
    window_end: datetime (exclusive)
    
    Returns:
        [IRI, ISI, EIR, SR, SF, TD, EC, low_confidence]
    """
    # Filter SMS logs
    window_sms = []
    for sms in sms_logs:
        dt = datetime.strptime(sms['timestamp'], "%Y-%m-%dT%H:%M:%S")
        if window_start <= dt < window_end:
            window_sms.append((dt, sms))
    
    # Sort just in case
    window_sms.sort(key=lambda x: x[0])
    
    incomes = []
    expenses = []
    income_times = []
    
    for dt, sms in window_sms:
        amt = abs(sms['amount'])
        if sms['type'] == 'credit':
            incomes.append(amt)
            income_times.append(dt)
        elif sms['type'] == 'debit':
            expenses.append(amt)

    # Defaults (imputation as specified: 0.5 for CV-based, 0.0 for others)
    IRI = 0.5
    ISI = 0.5
    EIR = 0.0
    SR = 0.0
    SF = 0.0
    TD = 0.0
    EC = 0.5
    low_confidence = 0.0

    sum_income = sum(incomes)
    sum_expense = sum(expenses)
    
    # Cold-start fallback
    if len(incomes) < 4:
        low_confidence = 1.0
        # Compute Engagement Consistency (EC)
        hours = []
        for i in range((window_end - window_start).days):
            curr_date = window_start + timedelta(days=i)
            date_str = curr_date.strftime("%Y-%m-%d")
            usage = app_usage.get(date_str, {})
            hours.append(usage.get('hours_active', 0.0))
        
        mean_hours = sum(hours) / len(hours) if hours else 0.0
        if mean_hours > 0:
            var_hours = sum((h - mean_hours)**2 for h in hours) / len(hours)
            std_hours = math.sqrt(var_hours)
            cv_hours = std_hours / mean_hours
            EC = 1.0 / (1.0 + cv_hours)
        else:
            EC = 0.0
    else:
        # 1. Income Regularity (IRI)
        gaps = [(income_times[i] - income_times[i-1]).total_seconds() / 86400.0 for i in range(1, len(income_times))]
        if len(gaps) > 0:
            mean_gap = sum(gaps) / len(gaps)
            if mean_gap > 0:
                var_gap = sum((g - mean_gap)**2 for g in gaps) / len(gaps)
                std_gap = math.sqrt(var_gap)
                cv_gap = std_gap / mean_gap
                IRI = 1.0 / (1.0 + cv_gap)
        
        # 2. Income Stability (ISI)
        mean_income = sum_income / len(incomes)
        if mean_income > 0:
            var_income = sum((i - mean_income)**2 for i in incomes) / len(incomes)
            std_income = math.sqrt(var_income)
            cv_income = std_income / mean_income
            ISI = 1.0 / (1.0 + cv_income)
            
        # 3. Expense-to-Income Ratio (EIR)
        if sum_income > 0:
            eir_raw = sum_expense / sum_income
            # Clip to [0, 2] to cap extreme outliers (e.g. huge expense, tiny income)
            EIR = max(0.0, min(2.0, eir_raw))
            
        # 4. Savings Rate (SR)
        if sum_income > 0:
            SR = (sum_income - sum_expense) / sum_income
            
        # 6. Trend (TD)
        mid_point = window_start + timedelta(days=15)
        
        early_incomes = 0.0
        early_expenses = 0.0
        late_incomes = 0.0
        late_expenses = 0.0
        
        for dt, sms in window_sms:
            amt = abs(sms['amount'])
            if window_start <= dt < mid_point:
                if sms['type'] == 'credit': early_incomes += amt
                elif sms['type'] == 'debit': early_expenses += amt
            elif mid_point <= dt < window_end:
                if sms['type'] == 'credit': late_incomes += amt
                elif sms['type'] == 'debit': late_expenses += amt
                
        if early_incomes > 0 and late_incomes > 0:
            sr_early = (early_incomes - early_expenses) / early_incomes
            sr_late = (late_incomes - late_expenses) / late_incomes
            TD = sr_late - sr_early
            
    # 5. Shortfall Frequency (SF)
    periods_with_tx = 0
    shortfall_periods = 0
    for w in range(5): # Up to 5 periods of 7 days in a 30-day window
        p_start = window_start + timedelta(days=w*7)
        p_end = min(window_start + timedelta(days=(w+1)*7), window_end)
        if p_start >= p_end: break
        
        p_inc = 0.0
        p_exp = 0.0
        tx_count = 0
        for dt, sms in window_sms:
            if p_start <= dt < p_end:
                tx_count += 1
                if sms['type'] == 'credit': p_inc += abs(sms['amount'])
                elif sms['type'] == 'debit': p_exp += abs(sms['amount'])
        
        if tx_count > 0:
            periods_with_tx += 1
            if p_exp > p_inc:
                shortfall_periods += 1
                
    if periods_with_tx > 0:
        SF = shortfall_periods / periods_with_tx

    return [IRI, ISI, EIR, SR, SF, TD, EC, low_confidence]
