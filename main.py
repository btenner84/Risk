"""
Medicare Risk Analysis - Client Portal
Full-featured platform with Contract and Parent Organization analysis
Uses your complete dataset with 7,215+ contracts
"""

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from pathlib import Path
from typing import Optional, List
import pandas as pd

# Import your existing functionality - adjust paths for client portal
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import your existing schemas and utilities
from web_app.schemas import (
    FilterOptionsResponse, 
    Message, 
    AnalysisFiltersRequest, 
    AnalysisDataResponse, 
    ParentOrganizationDetailsResponse,
    PlanListResponse,
    PlanDetailsResponse
)

# Import your existing database utilities (unchanged)
from web_app.db_utils import (
    get_filter_options_from_db, 
    get_analysis_data_from_db, 
    get_org_details_from_db, 
    get_plans_for_contract_from_db, 
    get_plan_details_from_db
)

# Initialize FastAPI app
app = FastAPI(title="Medicare Risk Analysis - Client Portal")

# Mount static files - use absolute path to your existing static files
static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'web_app', 'static'))
app.mount("/static", StaticFiles(directory=static_path), name="static")

# Templates - use your existing templates
templates_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'web_app', 'templates'))
templates = Jinja2Templates(directory=templates_path)

# ============================================================================
# CORE ENDPOINTS - Same as your research environment
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard with Contract and Parent Organization analyzers only"""
    return templates.TemplateResponse("dashboard_client.html", {"request": request})

@app.get("/contract-analyzer", response_class=HTMLResponse)
async def contract_analyzer_page(request: Request):
    """Contract analyzer page"""
    return templates.TemplateResponse("contract_analyzer_client.html", {"request": request})

@app.get("/parent-analyzer", response_class=HTMLResponse)
async def parent_analyzer_page(request: Request):
    """Parent organization analyzer page"""
    return templates.TemplateResponse("parent_analyzer_client.html", {"request": request})

# ============================================================================
# API ENDPOINTS - Exact copies from your main app
# ============================================================================

@app.get("/api/filter-options", response_model=FilterOptionsResponse)
async def get_filter_options():
    """Get filter options using your existing database"""
    try:
        return get_filter_options_from_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching filter options: {str(e)}")

@app.get("/api/analysis-data", response_model=AnalysisDataResponse)
async def get_analysis_data(
    contract_ids: List[str] = Query(...),
    years: List[int] = Query(...),
    states: Optional[List[str]] = Query(None),
    plan_types: Optional[List[str]] = Query(None),
    parent_organizations: Optional[List[str]] = Query(None)
):
    """Get analysis data using your existing database"""
    try:
        filters = AnalysisFiltersRequest(
            contract_ids=contract_ids,
            years=years,
            states=states or [],
            plan_types=plan_types or [],
            parent_organizations=parent_organizations or []
        )
        return get_analysis_data_from_db(filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching analysis data: {str(e)}")

@app.get("/api/parent-organization-details", response_model=ParentOrganizationDetailsResponse)
async def get_parent_organization_details(
    org_name_raw: str = Query(...),
    target_year: int = Query(2023)
):
    """Get parent organization details using your existing database"""
    try:
        return get_org_details_from_db(org_name_raw, target_year)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching organization details: {str(e)}")

@app.get("/api/plans/{contract_id}", response_model=PlanListResponse)
async def get_plans_for_contract(contract_id: str):
    """Get plans for contract using your existing database"""
    try:
        return get_plans_for_contract_from_db(contract_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching plans for contract {contract_id}: {str(e)}")

@app.get("/api/plan-details/{contract_id}/{plan_id}", response_model=PlanDetailsResponse)
async def get_plan_details(contract_id: str, plan_id: str):
    """Get plan details using your existing database"""
    try:
        return get_plan_details_from_db(contract_id, plan_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching plan details for {contract_id}-{plan_id}: {str(e)}")

# ============================================================================
# HEALTH CHECK FOR RAILWAY
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway deployment"""
    return {"status": "healthy", "service": "Medicare Risk Analysis Client Portal"}

# ============================================================================
# STARTUP - Initialize your existing database connection
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    print("🚀 Starting Medicare Risk Analysis Client Portal...")
    print("📊 Loading lineage data and initializing DuckDB connection...")
    
    # Your existing startup logic will run when db_utils is imported
    # This ensures the same data initialization as your research environment

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    print("🛑 Shutting down Medicare Risk Analysis Client Portal...")

# ============================================================================
# MAIN - For running locally or on Railway
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    print(f"🌐 Starting Client Portal on port {port}")
    print("📋 Available analyzers: Contract Analysis, Parent Organization Analysis")
    print("🔗 Full Medicare fraud detection capabilities enabled")
    uvicorn.run(app, host="0.0.0.0", port=port)
