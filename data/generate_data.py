import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating sample e-commerce data...")
np.random.seed(42)

# Generate Q3 2024 data (July-September)
dates = pd.date_range('2024-07-01', '2024-09-30', freq='D')
categories = ['Electronics', 'Clothing', 'Food & Beverage', 'Home & Garden']

data = []
for date in dates:
    day_num = (date - dates[0]).days
    
    for category in categories:
        # Base revenue by category
        base_revenue = {'Electronics': 45000, 'Clothing': 32000, 
                       'Food & Beverage': 28000, 'Home & Garden': 35000}[category]
        
        # Add trends
        if category == 'Electronics':
            trend = -day_num * 250  # Declining
        elif category == 'Clothing':
            trend = day_num * 150   # Growing
        else:
            trend = np.random.uniform(-50, 50) * day_num
        
        # September anomaly for Electronics
        anomaly = np.random.uniform(-8000, -12000) if (category == 'Electronics' and date.month == 9) else 0
        
        # Calculate metrics
        revenue = max(base_revenue + trend + anomaly + np.random.normal(0, 3000), 5000)
        orders = int(revenue / np.random.uniform(85, 125))
        conversion_rate = np.random.uniform(2.5, 4.5)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'category': category,
            'revenue': round(revenue, 2),
            'orders': orders,
            'conversion_rate': round(conversion_rate, 2),
            'avg_order_value': round(revenue/orders, 2),
            'website_traffic': int(orders / (conversion_rate/100)),
            'marketing_spend': round(revenue * np.random.uniform(0.08, 0.15), 2)
        })

df = pd.DataFrame(data)

# Add data quality issues (realistic)
for col in ['revenue', 'orders']:
    mask = np.random.random(len(df)) < 0.05
    df.loc[mask, col] = np.nan

# Add duplicates
df = pd.concat([df, df.sample(15)], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Save
df.to_csv('data/ecommerce_q3_2024.csv', index=False)
print(f"✅ Generated {len(df)} rows of sample data")
print(f"   Saved to: data/ecommerce_q3_2024.csv")