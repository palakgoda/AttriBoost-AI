import pytest
import pandas as pd
import numpy as np
from backend.attribution_engine import compute_all_models
from backend.main import validate_params
from fastapi import HTTPException

# Create mock dataframes representing standardized marketing touchpoints and conversions
@pytest.fixture
def sample_marketing_data():
    touchpoints = pd.DataFrame([
        # User 1: 2 touchpoints, converts for $100 (Google -> Meta)
        {'user_id': 'USR_1', 'timestamp': '2026-06-01 10:00:00', 'channel': 'Google Ads', 'campaign': 'G_Brand', 'cost': 1.50},
        {'user_id': 'USR_1', 'timestamp': '2026-06-02 12:00:00', 'channel': 'Meta Ads', 'campaign': 'M_Retargeting', 'cost': 1.00},
        
        # User 2: 3 touchpoints, converts for $100 (Google -> Meta -> TikTok)
        {'user_id': 'USR_2', 'timestamp': '2026-06-01 10:00:00', 'channel': 'Google Ads', 'campaign': 'G_Brand', 'cost': 1.50},
        {'user_id': 'USR_2', 'timestamp': '2026-06-02 12:00:00', 'channel': 'Meta Ads', 'campaign': 'M_Prosp', 'cost': 1.00},
        {'user_id': 'USR_2', 'timestamp': '2026-06-03 14:00:00', 'channel': 'TikTok Ads', 'campaign': 'T_Influencer', 'cost': 2.00},
        
        # User 3: 1 touchpoint, converts for $50 (Organic Search)
        {'user_id': 'USR_3', 'timestamp': '2026-06-04 10:00:00', 'channel': 'Organic Search', 'campaign': 'SEO', 'cost': 0.00},
        
        # User 4: Clicks but does NOT convert
        {'user_id': 'USR_4', 'timestamp': '2026-06-05 10:00:00', 'channel': 'Google Ads', 'campaign': 'G_Brand', 'cost': 1.50}
    ])
    
    conversions = pd.DataFrame([
        {'user_id': 'USR_1', 'timestamp': '2026-06-02 15:00:00', 'conversion_value': 100.00},
        {'user_id': 'USR_2', 'timestamp': '2026-06-03 16:00:00', 'conversion_value': 100.00},
        {'user_id': 'USR_3', 'timestamp': '2026-06-04 12:00:00', 'conversion_value': 50.00}
    ])
    
    # Ensure datetime formats
    touchpoints['timestamp'] = pd.to_datetime(touchpoints['timestamp'])
    conversions['timestamp'] = pd.to_datetime(conversions['timestamp'])
    
    return touchpoints, conversions

def test_first_touch_attribution(sample_marketing_data):
    df_tp, df_conv = sample_marketing_data
    results = compute_all_models(df_tp, df_conv, half_life=7.0, lookback_days=14.0)
    
    first_touch = results['First Touch']
    
    # Extract attributed revenues
    meta_rev = next(x['attributed_revenue'] for x in first_touch if x['channel'] == 'Meta Ads')
    google_rev = next(x['attributed_revenue'] for x in first_touch if x['channel'] == 'Google Ads')
    organic_rev = next(x['attributed_revenue'] for x in first_touch if x['channel'] == 'Organic Search')
    tiktok_rev = next(x['attributed_revenue'] for x in first_touch if x['channel'] == 'TikTok Ads')
    
    # User 1: First is Google ($100)
    # User 2: First is Google ($100)
    # User 3: First is Organic ($50)
    # User 4: Did not convert ($0)
    # Expected: Google=$200, Organic=$50, Meta=$0, TikTok=$0
    assert google_rev == 200.00
    assert organic_rev == 50.00
    assert meta_rev == 0.00
    assert tiktok_rev == 0.00

def test_last_touch_attribution(sample_marketing_data):
    df_tp, df_conv = sample_marketing_data
    results = compute_all_models(df_tp, df_conv, half_life=7.0, lookback_days=14.0)
    
    last_touch = results['Last Touch']
    
    meta_rev = next(x['attributed_revenue'] for x in last_touch if x['channel'] == 'Meta Ads')
    google_rev = next(x['attributed_revenue'] for x in last_touch if x['channel'] == 'Google Ads')
    organic_rev = next(x['attributed_revenue'] for x in last_touch if x['channel'] == 'Organic Search')
    tiktok_rev = next(x['attributed_revenue'] for x in last_touch if x['channel'] == 'TikTok Ads')
    
    # User 1: Last is Meta ($100)
    # User 2: Last is TikTok ($100)
    # User 3: Last is Organic ($50)
    # Expected: Meta=$100, TikTok=$100, Organic=$50, Google=$0
    assert meta_rev == 100.00
    assert tiktok_rev == 100.00
    assert organic_rev == 50.00
    assert google_rev == 0.00

def test_linear_attribution(sample_marketing_data):
    df_tp, df_conv = sample_marketing_data
    results = compute_all_models(df_tp, df_conv, half_life=7.0, lookback_days=14.0)
    
    linear = results['Linear']
    
    google_rev = next(x['attributed_revenue'] for x in linear if x['channel'] == 'Google Ads')
    meta_rev = next(x['attributed_revenue'] for x in linear if x['channel'] == 'Meta Ads')
    tiktok_rev = next(x['attributed_revenue'] for x in linear if x['channel'] == 'TikTok Ads')
    organic_rev = next(x['attributed_revenue'] for x in linear if x['channel'] == 'Organic Search')
    
    # User 1: Google + Meta (50% each of $100 -> Google=$50, Meta=$50)
    # User 2: Google + Meta + TikTok (33.33% each of $100 -> Google=$33.33, Meta=$33.33, TikTok=$33.33)
    # User 3: Organic (100% of $50 -> Organic=$50)
    # Total Expected: Google=83.33, Meta=83.33, TikTok=33.33, Organic=50.00
    assert round(google_rev, 2) == 83.33
    assert round(meta_rev, 2) == 83.33
    assert round(tiktok_rev, 2) == 33.33
    assert round(organic_rev, 2) == 50.00

def test_ushaped_attribution(sample_marketing_data):
    df_tp, df_conv = sample_marketing_data
    results = compute_all_models(df_tp, df_conv, half_life=7.0, lookback_days=14.0)
    
    ushaped = results['U-Shaped']
    
    google_rev = next(x['attributed_revenue'] for x in ushaped if x['channel'] == 'Google Ads')
    meta_rev = next(x['attributed_revenue'] for x in ushaped if x['channel'] == 'Meta Ads')
    tiktok_rev = next(x['attributed_revenue'] for x in ushaped if x['channel'] == 'TikTok Ads')
    organic_rev = next(x['attributed_revenue'] for x in ushaped if x['channel'] == 'Organic Search')
    
    # User 1 (2 clicks): Google (First) + Meta (Last). Credit splits 50/50 -> Google=$50, Meta=$50.
    # User 2 (3 clicks): Google (First 40%), TikTok (Last 40%), Meta (Middle 20%) -> Google=$40, TikTok=$40, Meta=$20.
    # User 3 (1 click): Organic (100%) -> Organic=$50.
    # Total Expected: Google=$90, Meta=$70, TikTok=$40, Organic=$50.
    assert round(google_rev, 2) == 90.00
    assert round(meta_rev, 2) == 70.00
    assert round(tiktok_rev, 2) == 40.00
    assert round(organic_rev, 2) == 50.00

def test_attribution_revenue_checksum(sample_marketing_data):
    df_tp, df_conv = sample_marketing_data
    results = compute_all_models(df_tp, df_conv, half_life=7.0, lookback_days=14.0)
    
    # Total converted revenue in mock conversions is $250.00
    # Every single model must sum up to exactly $250.00 total attributed revenue!
    for model_name, channels in results.items():
        total_attributed = sum(ch['attributed_revenue'] for ch in channels)
        assert total_attributed == pytest.approx(250.00, abs=0.1)

def test_lookback_window_filter():
    # Touchpoint is 20 days before conversion.
    # With a 14-day lookback, it should get 0 credit (i.e. organic search conversion gets full credit, or 0 overall credit)
    touchpoints = pd.DataFrame([
        {'user_id': 'USR_Z', 'timestamp': '2026-06-01 10:00:00', 'channel': 'Google Ads', 'campaign': 'G_Brand', 'cost': 1.00}
    ])
    conversions = pd.DataFrame([
        {'user_id': 'USR_Z', 'timestamp': '2026-06-21 10:00:00', 'conversion_value': 100.00}
    ])
    
    touchpoints['timestamp'] = pd.to_datetime(touchpoints['timestamp'])
    conversions['timestamp'] = pd.to_datetime(conversions['timestamp'])
    
    # 1. 14-day lookback -> Google should get 0 credit since it happened 20 days ago
    results_14 = compute_all_models(touchpoints, conversions, half_life=7.0, lookback_days=14.0)
    google_rev_14 = next(x['attributed_revenue'] for x in results_14['Linear'] if x['channel'] == 'Google Ads')
    assert google_rev_14 == 0.0

    # 2. 30-day lookback -> Google should get the full $100 credit since 20 days is within 30 days
    results_30 = compute_all_models(touchpoints, conversions, half_life=7.0, lookback_days=30.0)
    google_rev_30 = next(x['attributed_revenue'] for x in results_30['Linear'] if x['channel'] == 'Google Ads')
    assert google_rev_30 == 100.00

def test_main_parameter_validation_bounds():
    # Valid parameters should pass silently
    validate_params(14.0, 7.0)
    validate_params(1.0, 1.0)
    validate_params(30.0, 14.0)
    
    # Invalid lookback (too small)
    with pytest.raises(HTTPException) as excinfo:
        validate_params(0.5, 7.0)
    assert excinfo.value.status_code == 400
    assert "Lookback window" in excinfo.value.detail
    
    # Invalid lookback (too large)
    with pytest.raises(HTTPException) as excinfo:
        validate_params(31.0, 7.0)
    assert excinfo.value.status_code == 400
    
    # Invalid half life (too large)
    with pytest.raises(HTTPException) as excinfo:
        validate_params(14.0, 15.0)
    assert excinfo.value.status_code == 400
    assert "Half-life" in excinfo.value.detail
