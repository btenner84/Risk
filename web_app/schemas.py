from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional

class FilterOptionsResponse(BaseModel):
    unique_cleaned_parent_org_names: List[str]
    parent_org_name_tuples: List[Tuple[str, str]] # List of (raw_name, cleaned_name)
    unique_plan_types: List[str]
    available_snp_flags: Dict[str, bool]
    errors: Optional[List[str]] = None # Optional field for reporting errors

class Message(BaseModel):
    message: str

# We will add more schemas here as we define more API responses, 
# for example, for the main data tables. 

# --- New Models for Analysis Data Endpoint ---

class AnalysisFiltersRequest(BaseModel):
    parent_organizations_raw: Optional[List[str]] = None
    plan_types: Optional[List[str]] = None
    snp_types_ui: Optional[List[str]] = None # Matches the keys used in Streamlit app's filter construction

class ParentOrgMetricValues(BaseModel):
    risk_score: Optional[float] = None
    raf_yy: Optional[float] = None
    enrollment: Optional[int] = None

class ParentOrgMetricsRow(BaseModel):
    year: int
    # Dynamically structured in the response, e.g., "ORG_NAME_MetricName": value
    # For schema, we can represent it as Dict[str, Any] and handle specific parsing in JS or backend if needed
    # Or, more explicitly:
    # metrics: Dict[str, ParentOrgMetricValues] # Keyed by Org Name
    # For sending to JS, a flat structure is often easier:
    # Example: { "year": 2023, "OrgA_Risk_Score": 1.2, "OrgA_Enrollment": 1000, "OrgB_Risk_Score": 1.1 ... }
    # So, we'll allow arbitrary keys for the org metrics in the row.
    # This will be a list of flat dictionaries; the structure is defined by parent_org_metrics_columns.
    pass # This will be a generic Dict[str, Any] effectively, defined by AnalysisDataResponse


class IndustrySummaryRow(BaseModel):
    year: int
    industry_weighted_avg_risk_score: Optional[float] = None
    industry_total_enrollment: Optional[int] = None


class ChartDataset(BaseModel):
    label: str
    data: List[Optional[float]]
    borderColor: str # To be set by backend/JS
    fill: bool = False


class ChartData(BaseModel):
    labels: List[str]  # Years
    datasets: List[ChartDataset]


class AnalysisDataResponse(BaseModel):
    parent_org_metrics_data: List[Dict[str, Any]] # Each dict is a row, keys match columns list
    parent_org_metrics_columns: List[str]      # List of column headers for the parent_org_metrics table
    industry_summary_data: List[IndustrySummaryRow]
    chart_data: ChartData
    # Add a field for raw query for debugging if needed by client (optional)
    # debug_queries: Optional[Dict[str, str]] = None
    load_errors: Optional[List[str]] = None

# --- Schemas for Parent Organization Analyzer Page ---

class OrgTimeDataPoint(BaseModel):
    year: int
    value: Optional[float] = None # For risk score, enrollment etc.

class OrgRiskScoreTimeDataPoint(BaseModel):
    year: int
    consolidated_risk_score: Optional[float] = None
    dual_risk_score: Optional[float] = None         # Weighted average risk score for dual eligible plan beneficiaries
    non_dual_risk_score: Optional[float] = None     # Weighted average risk score for non-dual eligible plan beneficiaries

class ParentOrganizationDetailsResponse(BaseModel):
    organization_name_cleaned: str
    organization_name_raw: str # The specific raw name used for querying
    contracts: List[str]
    states: List[str]
    enrollment_over_time: List[OrgTimeDataPoint]       # [{year: YYYY, value: count}, ...]
    risk_score_over_time_consolidated: List[OrgTimeDataPoint] # [{year: YYYY, value: score}, ...]
    risk_score_over_time_by_dual_status: List[OrgRiskScoreTimeDataPoint] # [{year: YYYY, dual_risk_score: X, non_dual_risk_score: Y}, ...]
    errors: Optional[List[str]] = None 

# --- New Schemas for Parent Organization Analyzer Page (Revised) ---
class EnrollmentMetricRow(BaseModel):
    year: int
    total_enrollment: Optional[int] = None
    total_enrollment_yoy_growth: Optional[float] = None # Percentage
    traditional_enrollment: Optional[int] = None
    traditional_enrollment_yoy_growth: Optional[float] = None # Percentage
    dual_enrollment: Optional[int] = None
    dual_enrollment_yoy_growth: Optional[float] = None # Percentage

class RiskScoreMetricRow(BaseModel):
    year: int
    weighted_avg_risk_score: Optional[float] = None
    weighted_avg_risk_score_yoy_growth: Optional[float] = None # Percentage
    traditional_weighted_avg_risk_score: Optional[float] = None
    traditional_weighted_avg_risk_score_yoy_growth: Optional[float] = None # Percentage
    dual_weighted_avg_risk_score: Optional[float] = None
    dual_weighted_avg_risk_score_yoy_growth: Optional[float] = None # Percentage

class ContractPlanEnrollmentRow(BaseModel):
    contract_id: str
    plan_id: str
    enrollment_2023: int # Data for target_contract_year, field name fixed for consistency
    total_addressable_market_2023: Optional[int] = None # TAM for target_contract_year (hidden)
    market_share_percentage_2023: Optional[float] = None # enrollment/TAM * 100 (displayed)
    risk_score_2023: Optional[float] = None # Data for target_contract_year
    typical_county_wtd_risk_score_2023: Optional[float] = None # Data for target_contract_year
    county_wtd_risk_score_ex_contract_2023: Optional[float] = None # NEW: County risk score excluding current contract
    plan_type: Optional[str] = None  
    snp_status: Optional[str] = None 
    risk_score_delta_vs_typical_county: Optional[float] = None # Data for target_contract_year
    risk_score_delta_vs_ex_contract: Optional[float] = None # NEW: Delta vs ex-contract county risk score

class ParentOrganizationDetailsResponseRevised(BaseModel):
    organization_name_cleaned: str
    organization_name_raw: str
    enrollment_metrics: List[Dict[str, Any]]
    enrollment_metrics_columns: List[str]
    risk_score_metrics: List[Dict[str, Any]]
    risk_score_metrics_columns: List[str]
    
    # NEW: 3 separate performance analysis tables
    overall_performance_metrics: List[Dict[str, Any]]  # Overall + HMO/PPO split
    overall_performance_metrics_columns: List[str]
    traditional_performance_metrics: List[Dict[str, Any]]  # Traditional + Traditional HMO/PPO
    traditional_performance_metrics_columns: List[str]
    dual_performance_metrics: List[Dict[str, Any]]  # Dual + Dual HMO/PPO  
    dual_performance_metrics_columns: List[str]
    
    chart_years: List[str]
    chart_total_enrollment: List[Optional[float]] # Allow float for None
    chart_weighted_avg_risk_score: List[Optional[float]]
    chart_weighted_avg_county_risk_score: List[Optional[float]]  # NEW: Purple line data
    chart_weighted_avg_county_risk_score_ex_contract: List[Optional[float]]  # NEW: Green dashed line data
    
    # NEW: Traditional chart data
    chart_traditional_enrollment: List[Optional[float]]
    chart_traditional_weighted_avg_risk_score: List[Optional[float]]
    chart_traditional_weighted_avg_county_risk_score: List[Optional[float]]
    chart_traditional_weighted_avg_county_risk_score_ex_contract: List[Optional[float]]  # NEW: Green dashed line
    
    # NEW: Dual chart data  
    chart_dual_enrollment: List[Optional[float]]
    chart_dual_weighted_avg_risk_score: List[Optional[float]]
    chart_dual_weighted_avg_county_risk_score: List[Optional[float]]
    chart_dual_weighted_avg_county_risk_score_ex_contract: List[Optional[float]]  # NEW: Green dashed line
    
    # NEW: Traditional HMO chart data
    chart_traditional_hmo_enrollment: List[Optional[float]]
    chart_traditional_hmo_weighted_avg_risk_score: List[Optional[float]]
    chart_traditional_hmo_weighted_avg_county_risk_score: List[Optional[float]]
    chart_traditional_hmo_weighted_avg_county_risk_score_ex_contract: List[Optional[float]]  # NEW: Green dashed line
    
    # NEW: Traditional PPO chart data  
    chart_traditional_ppo_enrollment: List[Optional[float]]
    chart_traditional_ppo_weighted_avg_risk_score: List[Optional[float]]
    chart_traditional_ppo_weighted_avg_county_risk_score: List[Optional[float]]
    chart_traditional_ppo_weighted_avg_county_risk_score_ex_contract: List[Optional[float]]  # NEW: Green dashed line
    
    # NEW: Dual HMO chart data
    chart_dual_hmo_enrollment: List[Optional[float]]
    chart_dual_hmo_weighted_avg_risk_score: List[Optional[float]]
    chart_dual_hmo_weighted_avg_county_risk_score: List[Optional[float]]
    chart_dual_hmo_weighted_avg_county_risk_score_ex_contract: List[Optional[float]]  # NEW: Green dashed line
    
    # NEW: Dual PPO chart data
    chart_dual_ppo_enrollment: List[Optional[float]]
    chart_dual_ppo_weighted_avg_risk_score: List[Optional[float]]
    chart_dual_ppo_weighted_avg_county_risk_score: List[Optional[float]]
    chart_dual_ppo_weighted_avg_county_risk_score_ex_contract: List[Optional[float]]  # NEW: Green dashed line
    
    # This list will contain data for the selected_contract_year
    contract_plan_enrollment_2023: List[ContractPlanEnrollmentRow] 
    selected_contract_year: int # New field to indicate the year for the contract_plan_enrollment data
    
    errors: Optional[List[str]] = None

# --- Schemas for Contract/Plan Analyzer Page ---

class PlanListResponse(BaseModel):
    contract_id: str
    plan_ids: List[str]
    errors: Optional[List[str]] = None

class PlanEnrollmentRiskSummaryRow(BaseModel):
    year: int
    total_enrollment: Optional[int] = None
    risk_score: Optional[float] = None
    risk_score_yoy_growth: Optional[float] = None # Percentage

class PlanCountyEnrollmentRow(BaseModel):
    county: str
    enrollment: Optional[int] = None

class PlanCountyEnrollmentForYear(BaseModel):
    year: int
    county_data: List[PlanCountyEnrollmentRow]

class PlanDetailsResponse(BaseModel):
    contract_id_cleaned: str
    plan_id_cleaned: str
    enrollment_summary: List[PlanEnrollmentRiskSummaryRow]
    pivoted_county_enrollment_data: List[Dict[str, Any]] 
    pivoted_county_enrollment_columns: List[str]      
    pivoted_county_risk_data: List[Dict[str, Any]] 
    pivoted_county_risk_columns: List[str]      
    weighted_avg_market_risk_row: Optional[Dict[str, Any]] = None 
    weighted_avg_market_risk_excl_contract_row: Optional[Dict[str, Any]] = None  # New field
    master_comparison_data: List[Dict[str, Any]] = [] # New field for master table data
    master_comparison_columns: List[str] = []      # New field for master table columns
    total_addressable_market_overall: Optional[int] = None 
    errors: Optional[List[str]] = None 

# --- Schemas for Market Analyzer Page (New) ---
class MarketAnalysisRow(BaseModel):
    contract_id: Optional[str] = None
    plan_id: Optional[str] = None
    organization_name: Optional[str] = None
    enrollment: Optional[int] = None
    plan_actual_risk_score: Optional[float] = None
    county_weighted_risk_score: Optional[float] = None
    delta_risk_score: Optional[float] = None
    # New fields for Market Analyzer
    plan_type: Optional[str] = None 
    snp_category: Optional[str] = None 

class MarketAnalysisResponse(BaseModel):
    year: int
    market_data: List[MarketAnalysisRow]
    load_errors: Optional[List[str]] = None

# --- Schemas for County Analyzer ---
# class CountyMetricValues(BaseModel): # No longer directly used in the final response in this shape
#     enrollment: Optional[int] = None
#     weighted_avg_risk_score: Optional[float] = None

# class CountyMetricsRow(BaseModel): # No longer directly used in the final response in this shape
#     year: int
#     overall: CountyMetricValues
#     traditional: CountyMetricValues
#     dual_eligible: CountyMetricValues

class CountyAnalysisResponse(BaseModel):
    county_name: str
    # metrics_by_year: List[CountyMetricsRow] # Old structure
    pivoted_metrics_data: List[Dict[str, Any]] # New: List of {"Metric": "Name", "YYYY": value, ...}
    pivoted_metrics_columns: List[str]        # New: e.g., ["Metric", "2015", "2016", ...]
    available_years: List[int] # Still useful to know which years had any data at all
    
    # New fields for the first chart
    chart_years: Optional[List[str]] = None
    chart_overall_enrollment: Optional[List[Optional[int]]] = None
    chart_overall_risk_score: Optional[List[Optional[float]]] = None
    
    # New fields for Parent Organization enrollment in the county
    county_parent_org_enrollment_data: Optional[List[Dict[str, Any]]] = None
    county_parent_org_enrollment_columns: Optional[List[str]] = None

    # New fields for Top 10 Parent Orgs Market Share Chart
    top_orgs_market_share_chart_years: Optional[List[str]] = None
    top_orgs_market_share_chart_datasets: Optional[List[Dict[str, Any]]] = None # e.g., { label: 'Org', data: [], borderColor: 'color' }
    
    load_errors: Optional[List[str]] = None

# --- Schemas for State Analyzer ---
class StateMetricRow(BaseModel):
    year: int
    is_dual_eligible: Optional[bool] = None
    total_enrollment: Optional[int] = None
    weighted_avg_risk_score: Optional[float] = None

class StateParentOrgMarketShare(BaseModel):
    parent_organization_name: str
    total_enrollment: int
    market_share_percentage: float

class StateParentOrgEnrollment(BaseModel):
    parent_organization_name: str
    year: int
    total_enrollment: int

class StateParentOrgRiskScore(BaseModel):
    parent_organization_name: str
    year: int
    weighted_avg_risk_score: float

class StateAnalysisResponse(BaseModel):
    state_name: str
    state_metrics: List[StateMetricRow]  # Enrollment and risk score by year and dual status
    parent_org_market_share: List[StateParentOrgMarketShare]  # Market share of top organizations
    parent_org_enrollment: List[StateParentOrgEnrollment]  # Parent org enrollment over time
    parent_org_risk_scores: List[StateParentOrgRiskScore]  # Parent org risk scores over time
    available_years: List[int]
    load_errors: Optional[List[str]] = None

# --- Schemas for Performance Heat Map Page ---
class PerformanceHeatMapCounty(BaseModel):
    county_name: str
    state: Optional[str] = None
    fips_state_county_code: Optional[str] = None
    total_enrollment: Optional[int] = None
    risk_score_delta: Optional[float] = None  # Plan performance vs county benchmark
    market_share_pct: Optional[float] = None
    enrollment_growth_pct: Optional[float] = None  # YoY growth
    total_addressable_market: Optional[int] = None  # Total enrollment in county across all plans

class PerformanceHeatMapResponse(BaseModel):
    organization_name: str
    year: int
    metric_type: str  # 'risk_delta', 'market_share', 'enrollment_growth', 'total_enrollment'
    county_performance: List[PerformanceHeatMapCounty]
    summary_stats: Optional[Dict[str, float]] = None  # Overall stats like avg_risk_delta, total_counties, etc.
    errors: Optional[List[str]] = None 