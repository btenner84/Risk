from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import os

router = APIRouter()

# Load the step-by-step analysis data
try:
    results_df = pd.read_csv('step_by_step_acceleration_results_20250703_130114.csv')
    lineage_df = pd.read_csv('step_by_step_lineage_data_20250703_130114.csv')
    print(f"Loaded lineage data: {len(results_df):,} lineages, {len(lineage_df):,} lineage-year records")
except Exception as e:
    print(f"Error loading lineage data: {e}")
    results_df = pd.DataFrame()
    lineage_df = pd.DataFrame()

@router.get("/search")
async def search_lineages(query: str = ""):
    """Search for lineages by organization name, contract number, or lineage ID"""
    
    if results_df.empty:
        raise HTTPException(status_code=500, detail="Lineage data not loaded")
    
    if not query or len(query) < 2:
        # Return top accelerators as default
        top_results = results_df.nlargest(20, 'acceleration_vs_competitors_pct_per_year')
    else:
        # Search in multiple fields
        query_lower = query.lower()
        mask = (
            results_df['parent_organization_name'].str.lower().str.contains(query_lower, na=False) |
            results_df['persistent_entity_id'].str.lower().str.contains(query_lower, na=False)
        )
        
        # Also search in lineage data for contract numbers
        if query_lower.startswith('h'):
            contract_mask = lineage_df['contract_number'].str.lower().str.contains(query_lower, na=False)
            matching_lineages = lineage_df[contract_mask]['persistent_entity_id'].unique()
            lineage_mask = results_df['persistent_entity_id'].isin(matching_lineages)
            mask = mask | lineage_mask
        
        top_results = results_df[mask].nlargest(50, 'acceleration_vs_competitors_pct_per_year')
    
    # Format results for frontend
    search_results = []
    for _, row in top_results.iterrows():
        search_results.append({
            'lineage_id': row['persistent_entity_id'],
            'parent_organization': row['parent_organization_name'],
            'plan_type': row['plan_type'],
            'acceleration_pct_per_year': round(row['acceleration_vs_competitors_pct_per_year'], 2),
            'r_squared': round(row['r_squared_competitors'], 3),
            'years_tracked': int(row['years_tracked']),
            'latest_enrollment': int(row['latest_enrollment']),
            'first_year': int(row['first_year']),
            'last_year': int(row['last_year'])
        })
    
    return {
        'results': search_results,
        'total_found': len(search_results)
    }

@router.get("/lineage/{lineage_id}")
async def get_lineage_details(lineage_id: str):
    """Get complete lineage details for a specific lineage ID"""
    
    if results_df.empty or lineage_df.empty:
        raise HTTPException(status_code=500, detail="Lineage data not loaded")
    
    # Get summary data
    summary_data = results_df[results_df['persistent_entity_id'] == lineage_id]
    if summary_data.empty:
        raise HTTPException(status_code=404, detail="Lineage not found")
    
    summary_row = summary_data.iloc[0]
    
    # Get year-by-year detailed data
    yearly_data = lineage_df[lineage_df['persistent_entity_id'] == lineage_id].sort_values('year')
    if yearly_data.empty:
        raise HTTPException(status_code=404, detail="No yearly data found for lineage")
    
    # Format summary metrics
    summary = {
        'lineage_id': lineage_id,
        'parent_organization': summary_row['parent_organization_name'],
        'plan_type': summary_row['plan_type'],
        'is_dual_eligible': bool(summary_row['is_dual_eligible']),
        'years_tracked': int(summary_row['years_tracked']),
        'first_year': int(summary_row['first_year']),
        'last_year': int(summary_row['last_year']),
        'latest_enrollment': int(summary_row['latest_enrollment']),
        'latest_risk_score': round(summary_row['latest_risk_score'], 4),
        'acceleration_vs_competitors_pct_per_year': round(summary_row['acceleration_vs_competitors_pct_per_year'], 3),
        'acceleration_vs_market_pct_per_year': round(summary_row['acceleration_vs_market_pct_per_year'], 3),
        'r_squared_competitors': round(summary_row['r_squared_competitors'], 4),
        'r_squared_market': round(summary_row['r_squared_market'], 4),
        'p_value_competitors': round(summary_row['p_value_competitors'], 6),
        'p_value_market': round(summary_row['p_value_market'], 6),
        'current_performance_vs_competitors_pct': round(summary_row['current_performance_vs_competitors_pct'], 2),
        'current_performance_vs_market_pct': round(summary_row['current_performance_vs_market_pct'], 2)
    }
    
    # Format yearly data
    yearly_details = []
    for _, row in yearly_data.iterrows():
        yearly_details.append({
            'year': int(row['year']),
            'contract_number': row['contract_number'],
            'plan_id': row['plan_id'],
            'parent_organization_name': row['parent_organization_name'],
            'plan_type': row['plan_type'],
            'is_dual_eligible': bool(row['is_dual_eligible']),
            'is_chronic_snp': bool(row['is_chronic_snp']),
            'is_institutional_snp': bool(row['is_institutional_snp']),
            'total_enrollment': int(row['total_enrollment']) if pd.notna(row['total_enrollment']) else 0,
            'actual_risk_score': round(row['actual_risk_score'], 4) if pd.notna(row['actual_risk_score']) else None,
            'num_counties': int(row['num_counties']) if pd.notna(row['num_counties']) else 0,
            'county_weighted_benchmark_including': round(row['county_weighted_benchmark_including'], 4) if pd.notna(row['county_weighted_benchmark_including']) else None,
            'county_weighted_benchmark_excluding': round(row['county_weighted_benchmark_excluding'], 4) if pd.notna(row['county_weighted_benchmark_excluding']) else None,
            'counties_in_footprint': row['counties_in_footprint'] if pd.notna(row['counties_in_footprint']) else '',
            'total_footprint_enrollment': int(row['total_footprint_enrollment']) if pd.notna(row['total_footprint_enrollment']) else 0,
            'performance_vs_market_including_pct': round(row['performance_vs_market_including_pct'], 3) if pd.notna(row['performance_vs_market_including_pct']) else None,
            'performance_vs_market_excluding_pct': round(row['performance_vs_market_excluding_pct'], 3) if pd.notna(row['performance_vs_market_excluding_pct']) else None
        })
    
    # Create graph data for time series
    graph_data = {
        'years': [d['year'] for d in yearly_details],
        'actual_risk_scores': [d['actual_risk_score'] for d in yearly_details],
        'benchmark_scores': [d['county_weighted_benchmark_excluding'] for d in yearly_details],
        'enrollment_sizes': [d['total_enrollment'] for d in yearly_details],
        'performance_vs_competitors': [d['performance_vs_market_excluding_pct'] for d in yearly_details]
    }
    
    # Calculate trend lines
    years_array = np.array(graph_data['years'])
    actual_scores = np.array([s for s in graph_data['actual_risk_scores'] if s is not None])
    benchmark_scores = np.array([s for s in graph_data['benchmark_scores'] if s is not None])
    
    trend_data = {}
    if len(actual_scores) >= 2:
        # Calculate trend line for actual scores
        valid_years = years_array[:len(actual_scores)]
        actual_trend = np.polyfit(valid_years, actual_scores, 1)
        trend_data['actual_trend'] = {
            'slope': float(actual_trend[0]),
            'intercept': float(actual_trend[1]),
            'values': [actual_trend[0] * year + actual_trend[1] for year in valid_years]
        }
        
        # Calculate trend line for benchmark
        if len(benchmark_scores) >= 2:
            benchmark_trend = np.polyfit(valid_years[:len(benchmark_scores)], benchmark_scores, 1)
            trend_data['benchmark_trend'] = {
                'slope': float(benchmark_trend[0]),
                'intercept': float(benchmark_trend[1]),
                'values': [benchmark_trend[0] * year + benchmark_trend[1] for year in valid_years[:len(benchmark_scores)]]
            }
    
    return {
        'summary': summary,
        'yearly_data': yearly_details,
        'graph_data': graph_data,
        'trend_data': trend_data
    } 