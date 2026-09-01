import json
import random
import uuid
import os
from datetime import datetime, timedelta

NUM_USERS = 150
OUTPUT_DIR = "generated_users"

def generate_user_data(user_id, profile):
    """
    Generates 90 days of mock gig worker data based on a behavior profile.
    Profiles: 'reliable', 'sporadic', 'declining', 'improving'
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    sms_logs = []
    app_usage = []
    
    total_income = 0
    active_days = 0

    # 1. Draw a hidden "true reliability" score for this user based on their profile
    if profile == 'reliable':
        base_reliability = random.uniform(0.75, 0.95)
    elif profile == 'sporadic':
        base_reliability = random.uniform(0.15, 0.45)
    elif profile == 'declining':
        base_reliability = random.uniform(0.3, 0.6) # Ends up lower
    elif profile == 'improving':
        base_reliability = random.uniform(0.5, 0.8) # Ends up higher
        
    # Ground truth score (with slight irreducible noise)
    true_reliability = min(1.0, max(0.0, base_reliability + random.uniform(-0.05, 0.05)))
    reliability_score = round(true_reliability, 4)
    
    # Deriving generation parameters from true reliability
    # Add irreducible noise to the parameter mappings so no single feature perfectly reveals the hidden state
    base_income_prob = 0.3 + (0.5 * true_reliability) + random.uniform(-0.1, 0.1)
    base_expense_prob = 0.5 - (0.3 * true_reliability) + random.uniform(-0.1, 0.1)
    
    # Increase mapping noise for volatility to reduce ISI R^2 from 0.85 to ~0.65
    volatility = 0.6 - (0.5 * true_reliability) + random.uniform(-0.15, 0.15)
    
    # Add mapping noise for expense ratio
    expense_ratio = 0.9 - (0.6 * true_reliability) + random.uniform(-0.15, 0.15)
    
    # Clamp valid ranges
    base_income_prob = min(0.95, max(0.1, base_income_prob))
    base_expense_prob = min(0.95, max(0.1, base_expense_prob))
    volatility = min(0.9, max(0.05, volatility))
    expense_ratio = min(1.2, max(0.3, expense_ratio))
    
    # We still keep some app usage generation for the cold-start EC fallback, but it's no longer the primary driver
    
    for day in range(90):
        current_date = start_date + timedelta(days=day)
        
        # Adjust probabilities dynamically for trend profiles
        day_factor = day / 90.0
        if profile == 'declining':
            income_prob = base_income_prob * (1.0 - 0.7 * day_factor)
            expense_prob = base_expense_prob * (1.0 + 0.5 * day_factor)
            active_prob = 0.9 - 0.8 * day_factor
            hours_base = 9 - 7 * day_factor
        elif profile == 'improving':
            income_prob = base_income_prob * (0.3 + 0.7 * day_factor)
            expense_prob = base_expense_prob * (1.0 - 0.5 * day_factor)
            active_prob = 0.2 + 0.7 * day_factor
            hours_base = 3 + 6 * day_factor
        else:
            income_prob = base_income_prob
            expense_prob = base_expense_prob
            if profile == 'reliable':
                active_prob = 0.9
                hours_base = 8
            else: # sporadic
                active_prob = 0.4
                hours_base = 5
                
        is_active = random.random() < active_prob
        
        if is_active:
            active_days += 1
            sessions = random.randint(3, 12)
            hours = hours_base + random.uniform(-1.5, 1.5)
            hours = max(1, min(14, hours))
            
            app_usage.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "sessions": sessions,
                "hours_active": round(hours, 2)
            })
            
        # Income generation (tied to income_prob, not just being active)
        if random.random() < income_prob:
            amount = 80.0 * (1.0 + random.uniform(-volatility, volatility))
            amount = max(10.0, amount)
            total_income += amount
            sms_logs.append({
                "timestamp": (current_date + timedelta(hours=random.randint(12, 22))).strftime("%Y-%m-%dT%H:%M:%S"),
                "sender": "MobileMoney",
                "text": f"Received ${amount:.2f} from GigPlatform.",
                "amount": round(amount, 2),
                "type": "credit"
            })
            
        # Expense generation
        if random.random() < expense_prob:
            # Expected expense is a fraction of expected income
            expected_expense = 80.0 * expense_ratio
            expense = expected_expense * (1.0 + random.uniform(-volatility, volatility))
            expense = max(1.0, expense)
            sms_logs.append({
                "timestamp": (current_date + timedelta(hours=random.randint(8, 20))).strftime("%Y-%m-%dT%H:%M:%S"),
                "sender": "MobileMoney",
                "text": f"Payment of ${expense:.2f} at Local Store.",
                "amount": -round(expense, 2),
                "type": "debit"
            })

    return {
        "user_id": user_id,
        "profile": profile,
        "reliability_score": reliability_score,
        "metrics_summary": {
            "total_income_30d": round(total_income, 2),
            "active_days_30d": active_days
        },
        "app_usage": app_usage,
        "sms_logs": sorted(sms_logs, key=lambda x: x["timestamp"]) # Sort SMS by time
    }

def main():
    import shutil
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    profiles = ['reliable', 'sporadic', 'declining', 'improving']
    
    for i in range(NUM_USERS):
        user_id = str(uuid.uuid4())
        # Distribute profiles evenly across the generated set
        profile = profiles[i % len(profiles)]
        
        user_data = generate_user_data(user_id, profile)
        
        file_path = os.path.join(OUTPUT_DIR, f"user_{user_id}.json")
        with open(file_path, 'w') as f:
            json.dump(user_data, f, indent=2)
            
    print(f"Generated {NUM_USERS} mock user profiles in '{OUTPUT_DIR}' directory.")
    print("Profile distribution: 12-13 per category (reliable, sporadic, declining, improving)")

if __name__ == "__main__":
    main()
