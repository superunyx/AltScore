import json
import random
import uuid
import os
from datetime import datetime, timedelta

NUM_USERS = 50
OUTPUT_DIR = "generated_users"

def generate_user_data(user_id, profile):
    """
    Generates 30 days of mock gig worker data based on a behavior profile.
    Profiles: 'reliable', 'sporadic', 'declining', 'improving'
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    sms_logs = []
    app_usage = []
    
    total_income = 0
    active_days = 0
    
    for day in range(90):
        current_date = start_date + timedelta(days=day)
        
        # Determine daily activity based on profile
        if profile == 'reliable':
            active_prob = 0.9
            hours_base = 8
        elif profile == 'sporadic':
            active_prob = 0.4
            hours_base = 5
        elif profile == 'declining':
            # Starts high (0.9), drops to low (0.1) over 90 days
            active_prob = 0.9 - (day / 90) * 0.8
            hours_base = 9 - (day / 90) * 7
        elif profile == 'improving':
            # Starts low (0.2), increases to high (0.9) over 90 days
            active_prob = 0.2 + (day / 90) * 0.7
            hours_base = 3 + (day / 90) * 6
            
        is_active = random.random() < active_prob
        
        if is_active:
            active_days += 1
            sessions = random.randint(3, 12)
            hours = hours_base + random.uniform(-1.5, 1.5)
            hours = max(1, min(14, hours)) # Bound between 1 and 14
            
            app_usage.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "sessions": sessions,
                "hours_active": round(hours, 2)
            })
            
            # SMS transactions for income on active days
            if random.random() < 0.85: # 85% chance of getting paid on an active day
                amount = random.uniform(30, 120)
                total_income += amount
                sms_logs.append({
                    "timestamp": (current_date + timedelta(hours=random.randint(12, 22))).strftime("%Y-%m-%dT%H:%M:%S"),
                    "sender": "MobileMoney",
                    "text": f"Received ${amount:.2f} from GigPlatform.",
                    "amount": round(amount, 2),
                    "type": "credit"
                })
        
        # Add some random daily expenses via SMS
        if random.random() < 0.35:
            expense = random.uniform(5, 45)
            sms_logs.append({
                "timestamp": (current_date + timedelta(hours=random.randint(8, 20))).strftime("%Y-%m-%dT%H:%M:%S"),
                "sender": "MobileMoney",
                "text": f"Payment of ${expense:.2f} at Local Store.",
                "amount": -round(expense, 2),
                "type": "debit"
            })
            
    # Calculate Reliability Score (0.0 to 1.0)
    # Metric 1: Consistency (percentage of days active)
    consistency = active_days / 90.0
    
    # Metric 2: Income regularity (normalized to expected max of ~$6000 over 90 days)
    income_factor = min(1.0, total_income / 6000.0) 
    
    # Metric 3: Trajectory (last 30 days vs first 30 days activity)
    first_30_active = sum(1 for log in app_usage if log['date'] <= (start_date + timedelta(days=30)).strftime("%Y-%m-%d"))
    last_30_active = sum(1 for log in app_usage if log['date'] >= (start_date + timedelta(days=60)).strftime("%Y-%m-%d"))
    
    trend = (last_30_active - first_30_active) / 30.0 # Range roughly -1.0 to 1.0
    trend_factor = (trend + 1) / 2 # Normalize to 0.0 - 1.0
    
    # Reliability Formula: 45% consistency, 35% income, 20% trajectory
    reliability_score = (0.45 * consistency) + (0.35 * income_factor) + (0.20 * trend_factor)
    reliability_score = round(max(0.0, min(1.0, reliability_score)), 4)
    
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
