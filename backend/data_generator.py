import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_marketing_data(output_dir="data", num_users=50000, total_touchpoints=500000, seed=42):
    """
    Generates realistic marketing touchpoint and conversion datasets.
    Stores them as CSVs in the output directory.
    """
    np.random.seed(seed)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Generating data for {num_users} users with approx {total_touchpoints} touchpoints...")
    
    channels = ['Google Ads', 'Meta Ads', 'TikTok Ads', 'Email', 'Organic Search']
    campaigns = {
        'Google Ads': ['Google_Brand_Search', 'Google_NonBrand_Search', 'Google_Shopping'],
        'Meta Ads': ['Meta_Prospecting_Lookalike', 'Meta_Retargeting_Catalog', 'Meta_Brand_Awareness'],
        'TikTok Ads': ['TikTok_Influencer_Push', 'TikTok_Spark_Ads', 'TikTok_Trends'],
        'Email': ['Email_Newsletter_Weekly', 'Email_Abandoned_Cart', 'Email_Promo_Weekend'],
        'Organic Search': ['Organic_SEO_Homepage', 'Organic_Blog_Tutorial', 'Organic_Product_Review']
    }
    
    # CPC configuration
    channel_cpc = {
        'Google Ads': 1.50,
        'Meta Ads': 1.25,
        'TikTok Ads': 0.85,
        'Email': 0.05,
        'Organic Search': 0.00
    }
    
    # Base conversion rates for different channels (for single-touch attribution reference)
    channel_conv_weights = {
        'Google Ads': 0.15,
        'Meta Ads': 0.12,
        'TikTok Ads': 0.08,
        'Email': 0.22,
        'Organic Search': 0.10
    }

    start_date = datetime(2026, 6, 1)
    
    touchpoints_data = []
    conversions_data = []
    
    # User lists
    user_ids = [f"USR_{i:06d}" for i in range(num_users)]
    
    # Generate touchpoints per user
    touchpoints_per_user = np.random.negative_binomial(n=2, p=0.3, size=num_users) + 1 # At least 1 touchpoint
    # Scale to match total_touchpoints roughly
    scale_factor = total_touchpoints / sum(touchpoints_per_user)
    touchpoints_per_user = np.maximum(1, np.round(touchpoints_per_user * scale_factor).astype(int))
    
    print("Simulating user journeys...")
    
    for i, user_id in enumerate(user_ids):
        n_tp = touchpoints_per_user[i]
        
        # Determine if this user will convert
        # Conversion probability increases with more touchpoints and specific channels
        user_channels = np.random.choice(channels, size=n_tp, p=[0.3, 0.25, 0.2, 0.1, 0.15])
        
        # Build path times
        user_times = sorted([
            start_date + timedelta(
                days=np.random.uniform(0, 30),
                hours=np.random.uniform(0, 24),
                minutes=np.random.uniform(0, 60)
            )
            for _ in range(n_tp)
        ])
        
        # Create touchpoints
        journey_touchpoints = []
        for j in range(n_tp):
            channel = user_channels[j]
            campaign = np.random.choice(campaigns[channel])
            cost = max(0.0, channel_cpc[channel] + np.random.normal(0, 0.1))
            timestamp = user_times[j]
            
            journey_touchpoints.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'channel': channel,
                'campaign': campaign,
                'cost': round(cost, 2)
            })
            touchpoints_data.append(journey_touchpoints[-1])
            
        # Determine conversion
        # Multi-touch conversion logic: Retargeting/Email + search yields highest conversion
        has_email = 'Email' in user_channels
        has_meta = 'Meta Ads' in user_channels
        has_google = 'Google Ads' in user_channels
        
        conv_prob = 0.02 # Base probability
        if has_meta and has_google:
            conv_prob += 0.12 # Synergy!
        if has_email:
            conv_prob += 0.08
        if n_tp > 3:
            conv_prob += 0.05
            
        conv_prob = min(0.35, conv_prob) # Cap at 35%
        
        if np.random.rand() < conv_prob:
            # Conversion happens after the last touchpoint
            last_tp_time = user_times[-1]
            conv_time = last_tp_time + timedelta(
                minutes=np.random.uniform(5, 120)
            )
            conv_value = round(float(np.random.exponential(scale=75.0) + 15.0), 2)
            
            conversions_data.append({
                'user_id': user_id,
                'timestamp': conv_time,
                'conversion_value': conv_value
            })
            
    print("Converting to DataFrames...")
    df_touchpoints = pd.DataFrame(touchpoints_data)
    df_conversions = pd.DataFrame(conversions_data)
    
    # Sort by timestamp
    df_touchpoints = df_touchpoints.sort_values('timestamp').reset_index(drop=True)
    df_conversions = df_conversions.sort_values('timestamp').reset_index(drop=True)
    
    # Save to CSV
    touchpoints_path = os.path.join(output_dir, "touchpoints.csv")
    conversions_path = os.path.join(output_dir, "conversions.csv")
    
    print(f"Writing datasets to {output_dir}...")
    df_touchpoints.to_csv(touchpoints_path, index=False)
    df_conversions.to_csv(conversions_path, index=False)
    
    print(f"Successfully generated:")
    print(f" - Touchpoints: {len(df_touchpoints)} rows in {touchpoints_path}")
    print(f" - Conversions: {len(df_conversions)} rows in {conversions_path}")
    
    return df_touchpoints, df_conversions

if __name__ == "__main__":
    generate_marketing_data(num_users=10000, total_touchpoints=100000)
