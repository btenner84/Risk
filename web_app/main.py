from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os # Import os for path joining if needed, though db_utils handles its own paths
from pathlib import Path
from typing import Optional, List
import pandas as pd # Added for enrollment data loading and aggregation

# Import utility functions and schemas
from . import db_utils # Import the module itself
from .routers import lineage_analyzer
from .schemas import (
    FilterOptionsResponse, 
    Message, 
    AnalysisFiltersRequest, 
    AnalysisDataResponse, 
    ParentOrganizationDetailsResponseRevised as ParentOrganizationDetailsResponse, # Alias to minimize changes below
    # OrgTimeDataPoint, # No longer directly used by this endpoint's response model
    # OrgRiskScoreTimeDataPoint, # No longer directly used by this endpoint's response model
    EnrollmentMetricRow, # For type hinting if needed, though response model handles it
    RiskScoreMetricRow,   # For type hinting if needed
    PlanListResponse,      # New: For Contract Analyzer - Plan IDs
    PlanDetailsResponse,   # New: For Contract Analyzer - Plan Details
    MarketAnalysisResponse, # Added for Market Analyzer
    CountyAnalysisResponse, # Added for County Analyzer
    StateAnalysisResponse, # Added for State Analyzer
    PerformanceHeatMapResponse
)
from .db_utils import (
    get_filter_options_from_db, get_analysis_data_from_db, 
    get_org_details_from_db, get_plans_for_contract_from_db, get_plan_details_from_db,
    get_market_analysis_data_from_db,
    get_county_analysis_from_db, # Added for County Analyzer
    get_county_name_suggestions_from_db, # Added for County Name Suggestions
    get_state_analysis_from_db, # Added for State Analyzer
    get_performance_heatmap_data_from_db,
    # UNH Analysis functions
    get_unh_provider_data_from_db, get_unh_timeline_data_from_db,
    get_unh_acquisition_summary_from_db, get_unh_provider_detail_from_db,
    get_unh_risk_acceleration_results_from_db, get_acquisition_detail_data_from_db,
    get_unh_detailed_analysis_data_from_db, get_unh_acquisition_timeline_data_from_db,
    get_unh_ground_truth_acquisitions_from_db, get_provider_timeline_data_from_db,
    get_provider_comparison_data_from_db, get_master_unh_analysis_from_db
)

# Initialize FastAPI app
app = FastAPI()

# Configure templates and static files
BASE_PATH = Path(__file__).resolve().parent
TEMPLATES_PATH = BASE_PATH / "templates"
STATIC_PATH = BASE_PATH / "static"

app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
templates = Jinja2Templates(directory=TEMPLATES_PATH)

# Include routers
app.include_router(lineage_analyzer.router, prefix="/api/lineage", tags=["lineage-analyzer"])

@app.on_event("startup")
async def startup_event():
    # Initialize the database connection and UDF when the application starts
    db_utils.get_db_connection()
    print("FastAPI application startup: DuckDB connection initialized.")

@app.on_event("shutdown")
async def shutdown_event():
    if db_utils.con: # Access the connection through the db_utils module as 'con'
        try:
            db_utils.con.close()
            print("FastAPI application shutdown: DuckDB connection closed.")
        except Exception as e:
            print(f"Error closing DuckDB connection during shutdown: {e}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request, "title": "Medicare Advantage Risk Score Analyzer"})
    except Exception as e:
        print(f"Error rendering template: {e}")
        raise HTTPException(status_code=500, detail="Internal server error rendering page.")

@app.get("/api/filter-options", response_model=FilterOptionsResponse)
async def get_filter_options_api():
    try:
        options_data = db_utils.get_filter_options_from_db()
        # errors = options_data.pop('load_errors', None) # No, load_errors is part of the dict from db_utils
        
        response_payload = {
            'unique_cleaned_parent_org_names': options_data.get('unique_cleaned_parent_org_names', []),
            'parent_org_name_tuples': options_data.get('parent_org_name_tuples', []),
            'unique_plan_types': options_data.get('unique_plan_types', []),
            'available_snp_flags': options_data.get('available_snp_flags', {}),
            'errors': options_data.get('load_errors') # Pass errors if present
        }
        return FilterOptionsResponse(**response_payload)
    except Exception as e:
        print(f"Error in /api/filter-options endpoint: {str(e)}")
        # Optionally, re-raise as HTTPException or return a custom error model
        # For now, let Pydantic validation handle response structure if an error occurs before this point
        # Or, return an error structure consistent with FilterOptionsResponse
        return FilterOptionsResponse(
            unique_cleaned_parent_org_names=[],
            parent_org_name_tuples=[],
            unique_plan_types=[],
            available_snp_flags={},
            errors=[f"Server error fetching filter options: {str(e)}"]
        )

@app.get("/parent-analyzer", response_class=HTMLResponse)
async def parent_analyzer_page(request: Request):
    return templates.TemplateResponse("parent_analyzer.html", {"request": request, "title": "Parent Organization Analyzer"})

@app.get("/contract-analyzer", response_class=HTMLResponse)
async def contract_analyzer_page(request: Request):
    return templates.TemplateResponse("contract_analyzer.html", {"request": request, "title": "Contract Analyzer"})

@app.get("/api/parent-organization-details", response_model=ParentOrganizationDetailsResponse)
async def get_parent_organization_details(
    org_name_raw: str = Query(..., description="The raw parent organization name as selected by the user"),
    target_year: Optional[int] = Query(2023, description="The target year for contract and plan enrollment data") # New query parameter
):
    print(f"API called for parent organization details: {org_name_raw}, Target Year: {target_year}")
    if not org_name_raw:
        # Should be caught by FastAPI validation if ... is used, but good practice
        raise HTTPException(status_code=400, detail="Organization name is required.")
    if target_year not in db_utils.ALL_AVAILABLE_YEARS:
        raise HTTPException(status_code=400, detail=f"Invalid target year. Must be between {db_utils.ALL_AVAILABLE_YEARS[0]} and {db_utils.ALL_AVAILABLE_YEARS[-1]}.")

    try:
        # Pass the target_year to the db_utils function
        details = db_utils.get_org_details_from_db(raw_org_name=org_name_raw, target_contract_year=target_year) 
        
        # The Pydantic model now expects selected_contract_year, which should be set by get_org_details_from_db
        # or we can set it here explicitly if get_org_details_from_db doesn't directly put it in the dict.
        # For now, assuming get_org_details_from_db returns a dict that can be unpacked into the Pydantic model.
        # Let's ensure the response includes the target_year as selected_contract_year.
        response_data = details.model_dump() # Use .model_dump() for Pydantic v2
        response_data['selected_contract_year'] = target_year # Ensure it's part of the response dict
        
        # print(f"API response for {org_name_raw}, Target Year {target_year}: {response_data}")
        return ParentOrganizationDetailsResponse(**response_data)
    except ValueError as ve:
        print(f"ValueError in API for parent organization details: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Unhandled exception in API for parent organization details: {e}")
        # Log the full error for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/api/plans-for-contract", response_model=PlanListResponse)
async def get_plans_for_contract_api(contract_id: str):
    print(f"API called for plans for contract: {contract_id}")
    if not contract_id:
        raise HTTPException(status_code=400, detail="Contract ID is required.")
    
    plan_list_response = db_utils.get_plans_for_contract_from_db(contract_id)
    # The PlanListResponse model handles the structure including errors.
    if plan_list_response.errors and not plan_list_response.plan_ids:
        # Consider if a specific HTTP status is better here, e.g., 404 if no plans and error indicates not found
        # For now, letting the response model carry the error details with a 200, or client can check errors.
        pass 
    return plan_list_response

@app.get("/api/plan-details", response_model=PlanDetailsResponse)
async def get_plan_details_api(contract_id: str, plan_id: str):
    print(f"[MAIN.PY LOG] API /api/plan-details called with Contract ID: {contract_id}, Plan ID: {plan_id}")
    if not contract_id or not plan_id:
        raise HTTPException(status_code=400, detail="Contract ID and Plan ID are required.")
    
    plan_details_response = db_utils.get_plan_details_from_db(contract_id, plan_id)
    # Similar to above, the PlanDetailsResponse model handles error reporting within its structure.
    if plan_details_response.errors and not plan_details_response.enrollment_summary: # Example check
        pass # Client should check for errors in the response body
    return plan_details_response

@app.post("/api/analysis-data", response_model=AnalysisDataResponse)
async def get_analysis_data_api(filters: AnalysisFiltersRequest):
    try:
        print(f"Received analysis request with filters: {filters.model_dump_json(indent=2)}")
        data = db_utils.get_analysis_data_from_db(filters) 
        return AnalysisDataResponse(**data)
    except Exception as e:
        print(f"Error in /api/analysis-data endpoint: {str(e)}")
        return AnalysisDataResponse(
            parent_org_metrics_data=[],
            parent_org_metrics_columns=['Year'],
            industry_summary_data=[],
            chart_data={'labels': [str(y) for y in db_utils.ALL_AVAILABLE_YEARS], 'datasets': []},
            load_errors=[f"Server error fetching analysis data: {str(e)}"]
        )

@app.get("/ping", response_class=HTMLResponse) # Or PlainTextResponse
async def ping_test():
    return "pong"

@app.get("/hello", response_model=Message)
async def hello_world():
    return Message(message="Hello World from the API - FastAPI is running!")

# --- New Page: Market Analyzer ---
@app.get("/market-analyzer", response_class=HTMLResponse)
async def market_analyzer_page(request: Request):
    return templates.TemplateResponse("market_analyzer.html", {"request": request})

# --- New API: Market Analysis Data ---
@app.get("/api/market-analysis", response_model=MarketAnalysisResponse)
async def market_analysis_data(year: int = Query(default=2023, ge=2015, le=2023)):
    try:
        data = get_market_analysis_data_from_db(year)
        return data
    except Exception as e:
        # Log the exception details for server-side review
        print(f"Error in /api/market-analysis endpoint: {e}") 
        # Return a Pydantic model that includes the error, matching the response_model
        error_response = MarketAnalysisResponse(year=year, market_data=[], load_errors=[str(e)])
        # To send an HTTP error status, you might raise HTTPException directly:
        # raise HTTPException(status_code=500, detail=str(e))
        # However, returning the error within the response model is also a valid approach chosen here.
        return error_response

# --- County Analyzer Page Route ---
@app.get("/county-analyzer", response_class=HTMLResponse)
async def county_analyzer_page(request: Request):
    return templates.TemplateResponse("county_analyzer.html", {"request": request, "title": "County Analyzer"})

@app.get("/state-analyzer", response_class=HTMLResponse)
async def state_analyzer_page(request: Request):
    return templates.TemplateResponse("state_analyzer.html", {"request": request, "title": "State Analyzer"})

@app.get("/provider-analyzer", response_class=HTMLResponse)
async def provider_analyzer_page(request: Request):
    return templates.TemplateResponse("provider_analyzer.html", {"request": request, "title": "Provider Acquisition Analyzer"})

@app.get("/lineage-analyzer", response_class=HTMLResponse)
async def lineage_analyzer_page(request: Request):
    return templates.TemplateResponse("lineage_analyzer.html", {"request": request, "title": "Medicare Advantage Lineage Analyzer"})

@app.get("/contract-tracker", response_class=HTMLResponse)
async def contract_tracker_page(request: Request):
    return templates.TemplateResponse("contract_tracker.html", {"request": request, "title": "Contract Tracker"})

@app.get("/api/contract-tracker/{contract_id}")
async def contract_tracker_data(contract_id: str):
    """Get lineage history for a specific contract ID"""
    try:
        from db_utils import get_contract_lineage_from_db
        data = get_contract_lineage_from_db(contract_id)
        return data
    except Exception as e:
        return {
            'success': False,
            'lineage': [],
            'message': f'Error loading contract lineage: {str(e)}'
        }

@app.get("/api/state-analysis", response_model=StateAnalysisResponse)
async def state_analysis_data(
    state: str = Query(..., min_length=1),
    plan_type_filter: str = Query(default="all", description="Plan type filter: 'all', 'traditional', 'dual_eligible', 'chronic', 'institutional'")
):
    try:
        # Basic validation
        if not state or state.strip() == "":
            return StateAnalysisResponse(
                state_name=state, 
                state_metrics=[], 
                parent_org_market_share=[], 
                parent_org_enrollment=[], 
                parent_org_risk_scores=[], 
                available_years=[], 
                load_errors=["State name cannot be empty."]
            )
        
        print(f"[MAIN.PY LOG] State analysis request: state={state}, plan_type={plan_type_filter}")
        data = get_state_analysis_from_db(state, plan_type_filter)
        return data
    except Exception as e:
        print(f"Error in /api/state-analysis endpoint for state '{state}': {e}")
        return StateAnalysisResponse(
            state_name=state, 
            state_metrics=[], 
            parent_org_market_share=[], 
            parent_org_enrollment=[], 
            parent_org_risk_scores=[], 
            available_years=[], 
            load_errors=[f"An unexpected error occurred: {str(e)}"]
        )

@app.get("/api/provider-analysis")
async def provider_analysis_data(acquisition_code: str = Query(..., description="UNH acquisition code from ground truth data")):
    print(f"Provider analysis API called for acquisition: {acquisition_code}")
    try:
        # Call the provider analysis function from db_utils
        provider_data = db_utils.get_provider_analysis_from_db(acquisition_code)
        return provider_data
    except Exception as e:
        print(f"Error in provider analysis endpoint: {str(e)}")
        return {
            'provider_metrics_data': [],
            'provider_metrics_columns': [],
            'chart_years': [],
            'chart_provider_count': [],
            'chart_avg_risk_score': [],
            'load_errors': [f"Internal server error: {str(e)}"]
        }

@app.get("/api/county-analysis", response_model=CountyAnalysisResponse)
async def county_analysis_data(county_name: str = Query(..., min_length=1)):
    try:
        # Basic validation, though more can be added
        if not county_name or county_name.strip() == "":
            return CountyAnalysisResponse(county_name=county_name, metrics_by_year=[], available_years=[], load_errors=["County name cannot be empty."])
        
        data = get_county_analysis_from_db(county_name)
        return data
    except Exception as e:
        print(f"Error in /api/county-analysis endpoint for county '{county_name}': {e}")
        return CountyAnalysisResponse(
            county_name=county_name, 
            metrics_by_year=[], 
            available_years=[], 
            load_errors=[f"An unexpected error occurred: {str(e)}"]
        )

# --- New API Endpoint for County Name Suggestions ---
@app.get("/api/county-name-suggestions", response_model=List[str])
async def county_name_suggestions_api(query: str = Query("", min_length=0, max_length=100)):
    """Provides a list of county name suggestions based on the query string."""
    if not query.strip(): # If query is empty or only whitespace, return empty list immediately
        return []
    try:
        suggestions = get_county_name_suggestions_from_db(query_str=query)
        return suggestions
    except Exception as e:
        print(f"Error in /api/county-name-suggestions endpoint for query '{query}': {e}")
        # In case of an error, return an empty list. Client-side can handle this gracefully.
        # For more detailed error reporting, you could define a Pydantic model for the response that includes an error field.
        return []

# --- Performance Heat Map Page Route ---
@app.get("/performance-heatmap", response_class=HTMLResponse)
async def performance_heatmap_page(request: Request):
    return templates.TemplateResponse("performance_heatmap.html", {"request": request, "title": "Performance Heat Map"})

# --- Performance Heat Map API Endpoint ---
@app.get("/api/performance-heatmap", response_model=PerformanceHeatMapResponse)
async def performance_heatmap_data(
    org_name: str = Query(..., description="The cleaned parent organization name"),
    year: int = Query(default=2023, ge=2015, le=2023, description="The year for analysis"),
    metric: str = Query(default="risk_delta", regex="^(risk_delta|market_share|enrollment_growth|total_enrollment)$", description="The performance metric to analyze"),
    county_limit: str = Query(default="100", description="Maximum number of counties to return (or 'all')"),
    min_enrollment: int = Query(default=100, ge=0, description="Minimum enrollment threshold per county"),
    plan_type_filter: str = Query(default="all", description="Plan type filter: 'all', 'traditional', 'dual_eligible', 'chronic', 'institutional'")
):
    """
    Get performance heat map data for a specific organization with performance optimization controls.
    
    Performance optimizations:
    - county_limit: Limits results to top N counties by total enrollment (or 'all' for no limit)
    - min_enrollment: Only includes counties with at least this many enrollees
    - plan_type_filter: Filter by plan type (all, traditional, dual_eligible, chronic, institutional)
    """
    print(f"[MAIN.PY LOG] Performance heat map API called: org={org_name}, year={year}, metric={metric}, limit={county_limit}, min_enrollment={min_enrollment}, plan_type={plan_type_filter}")
    
    try:
        result = get_performance_heatmap_data_from_db(
            org_name, 
            year, 
            metric, 
            county_limit=county_limit,
            min_enrollment=min_enrollment,
            plan_type_filter=plan_type_filter
        )
        return result
    except Exception as e:
        print(f"[MAIN.PY ERROR] Performance heat map error: {e}")
        return PerformanceHeatMapResponse(
            organization_name=org_name,
            year=year,
            metric_type=metric,
            county_performance=[],
            errors=[f"Internal server error: {str(e)}"]
        )

# --- UNH Acquisition Analyzer ---
@app.get("/unh-analyzer", response_class=HTMLResponse)
async def unh_analyzer_page(request: Request):
    import time
    timestamp = int(time.time())
    return templates.TemplateResponse("unh_analyzer.html", {"request": request, "title": "UNH Acquisition Analyzer", "timestamp": timestamp})

@app.get("/api/unh-dashboard")
async def unh_dashboard_data():
    """Get UNH acquisition dashboard summary data"""
    try:
        data = get_unh_acquisition_summary_from_db()
        return data
    except Exception as e:
        return {
            'dashboard_stats': {},
            'acquisition_timeline': [],
            'geographic_distribution': [],
            'errors': [f'Error loading UNH dashboard data: {str(e)}']
        }

@app.get("/api/unh-providers")
async def unh_providers_data(
    category: Optional[str] = Query(None, description="Filter by UNH acquisition category"),
    state: Optional[str] = Query(None, description="Filter by practice state"),
    acquisition_year_start: Optional[int] = Query(None, description="Filter acquisition year range start"),
    acquisition_year_end: Optional[int] = Query(None, description="Filter acquisition year range end")
):
    """Get UNH provider data with optional filters"""
    try:
        filters = {}
        
        if category:
            filters['unh_category'] = [category]
        
        if state:
            filters['practice_state'] = [state]
            
        if acquisition_year_start and acquisition_year_end:
            filters['acquisition_year_range'] = [acquisition_year_start, acquisition_year_end]
        
        data = get_unh_provider_data_from_db(filters)
        return data
    except Exception as e:
        return {
            'provider_data': [],
            'provider_columns': [],
            'summary_stats': {},
            'errors': [f'Error loading UNH provider data: {str(e)}']
        }

@app.get("/api/unh-timeline")
async def unh_timeline_data(
    npi: Optional[str] = Query(None, description="Filter by specific provider NPI"),
    category: Optional[str] = Query(None, description="Filter by UNH acquisition category")
):
    """Get UNH provider timeline data for temporal analysis"""
    try:
        data = get_unh_timeline_data_from_db(npi=npi, category=category)
        return data
    except Exception as e:
        return {
            'timeline_data': [],
            'chart_data': {},
            'errors': [f'Error loading UNH timeline data: {str(e)}']
        }

@app.get("/api/unh-provider-detail/{npi}")
async def unh_provider_detail_data(npi: str):
    """Get detailed information for a specific UNH provider"""
    try:
        data = get_unh_provider_detail_from_db(npi)
        return data
    except Exception as e:
        return {
            'provider_info': {},
            'timeline_data': [],
            'errors': [f'Error loading UNH provider detail: {str(e)}']
        }

@app.get("/api/unh-risk-acceleration")
async def unh_risk_acceleration_data():
    """Get UNH risk acceleration analysis results"""
    try:
        data = get_unh_risk_acceleration_results_from_db()
        return data
    except Exception as e:
        return {
            'results_available': False,
            'statistical_tests': {},
            'temporal_summary': [],
            'category_analysis': [],
            'errors': [f'Error loading risk acceleration results: {str(e)}']
        }

@app.get("/api/unh-detailed-analysis")
async def unh_detailed_analysis_data():
    """Get comprehensive UNH risk acceleration analysis with detailed insights"""
    try:
        data = get_unh_detailed_analysis_data_from_db()
        return data
    except Exception as e:
        return {
            'summary_stats': {},
            'statistical_results': {},
            'period_stats': {},
            'acceleration_stats': {},
            'category_analysis': {},
            'temporal_trends': {},
            'errors': [f'Error loading detailed analysis: {str(e)}']
        }

@app.get("/api/unh-ground-truth-acquisitions")
async def unh_ground_truth_acquisitions_data():
    """Get list of known UNH acquisitions from ground truth data"""
    try:
        data = get_unh_ground_truth_acquisitions_from_db()
        return data
    except Exception as e:
        return {
            'acquisitions': [],
            'errors': [f'Error loading ground truth acquisitions: {str(e)}']
        }

@app.get("/api/unh-acquisition-timeline")
async def unh_acquisition_timeline_data(category: str):
    """Get timeline data for a specific UNH acquisition category or ground truth acquisition name"""
    try:
        data = get_unh_acquisition_timeline_data_from_db(category)
        return data
    except Exception as e:
        return {
            'timeline_data': [],
            'acquisition_info': {},
            'errors': [f'Error loading acquisition timeline: {str(e)}']
        }

@app.get("/api/unh-provider-timeline")
async def unh_provider_timeline_data(category: str):
    """Get detailed provider-level timeline data for a specific UNH acquisition"""
    try:
        data = get_provider_timeline_data_from_db(category)
        return data
    except Exception as e:
        return {
            'providers': [],
            'years': [],
            'specialties': [],
            'errors': [f'Error loading provider timeline for {category}: {str(e)}']
        }

@app.get("/api/unh-acquisition-detail")
async def get_unh_acquisition_detail(name: str, year: int, state: str = None):
    """Get detailed information for a specific UNH acquisition"""
    try:
        data = get_acquisition_detail_data_from_db(name, year, state)
        return data
    except Exception as e:
        return {
            'acquisition_info': {},
            'providers': [],
            'geographic_summary': {},
            'risk_timeline': [],
            'errors': [f'Error loading acquisition detail for {name}: {str(e)}']
        }

@app.get("/api/unh-provider-comparison")
async def unh_provider_comparison_data(npi: str):
    """Get provider comparison data: UNH provider vs peers in same specialty/geography"""
    try:
        data = get_provider_comparison_data_from_db(npi)
        return data
    except Exception as e:
        return {
            'provider_info': {},
            'provider_timeline': [],
            'peer_timeline': [],
            'comparison_stats': {},
            'errors': [f'Error loading provider comparison for {npi}: {str(e)}']
        }

@app.get("/api/unh-master-analysis")
async def unh_master_analysis_data():
    """
    Get comprehensive Master Analysis of UNH acquisition effects on provider risk score growth.
    
    This endpoint provides the definitive analysis answering: 
    Do UNH-acquired providers accelerate or decelerate risk score changes after acquisition?
    
    Compares providers in two ways:
    1. Before vs After: Each provider's trend before acquisition vs after acquisition
    2. UNH vs Peers: Acquired providers vs similar providers in same county/specialty
    
    Returns detailed results across multiple dimensions:
    - Executive summary with primary conclusions
    - Analysis by individual acquisition
    - Analysis by specialty
    - Analysis by geography
    - Individual provider results
    - Statistical methodology and data quality metrics
    """
    try:
        data = get_master_unh_analysis_from_db()
        return data
    except Exception as e:
        return {
            'executive_summary': {'error': f'Error loading master analysis: {str(e)}'},
            'acquisition_analysis': [],
            'specialty_analysis': [],
            'geographic_analysis': [],
            'provider_results': [],
            'statistical_summary': {},
            'methodology': {},
            'data_quality': {},
            'errors': [f'Error loading UNH master analysis: {str(e)}']
        }

# For running directly with uvicorn, if needed (though poetry script is preferred)
if __name__ == "__main__":
    # Determine the correct relative path to the 'web_app' directory if this script is run directly.
    # This is tricky because uvicorn might be run from project root or from web_app directory.
    # Assuming it's run from project root: `python -m web_app.main` is not how this file is structured for direct run.
    # If running `python web_app/main.py` directly from project root:
    # uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir="web_app") is not quite right.
    # The standard way is `uvicorn web_app.main:app --reload` from the project root.
    # This __main__ block is more for conceptual testing or if you restructure.
    print("Attempting to run uvicorn directly from main.py - ensure you are in the project root.")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 