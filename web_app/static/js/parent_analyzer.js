let currentParentOrgSelection = null;
let parentOrgSuggestions = [];
let uniqueCleanedParentOrgNames = []; // Store all unique cleaned names from API
let parentOrgNameTuples = []; // Store (raw, cleaned) tuples from API
const API_BASE_URL = `${window.location.protocol}//${window.location.host}`;

// Store original data for sorting, and current sort state
let originalTableData = {}; 
let currentSortState = {}; // { tableId: { key: 'col_key', direction: 'asc'/'desc' } }

// Available years for the contract/plan table selector
const CONTRACT_TABLE_AVAILABLE_YEARS = Array.from({length: 2023 - 2015 + 1}, (_, i) => 2015 + i).reverse(); // Generates [2023, 2022, ..., 2015]

document.addEventListener('DOMContentLoaded', () => {
    console.log("Parent Analyzer Page Loaded");
    fetchParentOrgFilterOptions(); // Separate fetcher if needed, or reuse main page's if suitable

    // Placeholder for analyze button listener
    const analyzeButton = document.getElementById('analyze-org-button');
    if (analyzeButton) {
        analyzeButton.addEventListener('click', () => {
            if (parentAnalyzerState.selectedParentOrgRaw) {
                fetchAndDisplayOrgDetails(parentAnalyzerState.selectedParentOrgRaw);
            } else {
                const errorDisplay = document.getElementById('parent-analyzer-error');
                if (errorDisplay) {
                    errorDisplay.textContent = "Please select a parent organization from the suggestions.";
                    errorDisplay.style.display = 'block';
                } else {
                    alert("Please select a parent organization from the suggestions.");
                }
            }
        });
    }

    // Populate year selector for contract/plan table
    const yearSelector = document.getElementById('contract-year-select');
    if (yearSelector) {
        CONTRACT_TABLE_AVAILABLE_YEARS.forEach(year => {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            yearSelector.appendChild(option);
        });
        // Add event listener to refetch data when year changes
        yearSelector.addEventListener('change', () => {
            if (currentParentOrgSelection && currentParentOrgSelection.rawName) {
                fetchAndDisplayOrgDetails(currentParentOrgSelection.rawName); // Refetch with new year
            }
        });
    }
});

// Store for this page's filter state
const parentAnalyzerState = {
    allParentOrgs: [], // Will be {raw: String, cleaned: String}
    selectedParentOrgRaw: null // Store the single selected RAW parent org name
};

async function fetchParentOrgFilterOptions() {
    try {
        const response = await fetch('/api/filter-options'); // Use existing endpoint
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        if (data.errors && data.errors.length > 0) {
            console.error("Errors fetching filter options for parent analyzer:", data.errors);
            // Display error on this page if an error display element exists
            return;
        }

        parentAnalyzerState.allParentOrgs = data.parent_org_name_tuples.map(t => ({ raw: t[0], cleaned: t[1] }));

        setupSimpleParentOrgFilter(
            'parent-org-analyzer-search', 
            'parent-org-analyzer-suggestions', 
            parentAnalyzerState.allParentOrgs.map(org => org.cleaned) // Provide cleaned names for suggestions
        );

    } catch (error) {
        console.error('Failed to fetch filter options for parent analyzer:', error);
    }
}

// Simpler version for single selection (no tags)
function setupSimpleParentOrgFilter(searchInputId, suggestionsContainerId, allCleanedOrgNames) {
    const searchInput = document.getElementById(searchInputId);
    const suggestionsContainer = document.getElementById(suggestionsContainerId);

    searchInput.addEventListener('input', () => {
        const query = searchInput.value.toLowerCase();
        suggestionsContainer.innerHTML = '';
        if (query) {
            const filteredOptions = allCleanedOrgNames.filter(cleanedName => cleanedName.toLowerCase().includes(query));
            filteredOptions.slice(0, 10).forEach(cleanedName => {
                const suggestionItem = document.createElement('div');
                suggestionItem.classList.add('suggestion-item');
                suggestionItem.textContent = cleanedName;
                suggestionItem.onclick = () => {
                    searchInput.value = cleanedName; // Set input to selected name
                    // Find the corresponding raw name and store it
                    const selectedOrgTuple = parentAnalyzerState.allParentOrgs.find(o => o.cleaned === cleanedName);
                    if (selectedOrgTuple) {
                        parentAnalyzerState.selectedParentOrgRaw = selectedOrgTuple.raw;
                        console.log("Selected Parent Org (RAW):", parentAnalyzerState.selectedParentOrgRaw);
                        console.log("Selected Parent Org (Cleaned):", cleanedName);
                    }
                    suggestionsContainer.innerHTML = '';
                };
                suggestionsContainer.appendChild(suggestionItem);
            });
        }
    });

    document.addEventListener('click', (event) => {
        if (!suggestionsContainer.contains(event.target) && event.target !== searchInput) {
            suggestionsContainer.innerHTML = '';
        }
    });
}

function createSortableTableSection(title, dataRows, columnConfig, tableId) {
    const sectionDiv = document.createElement('div');
    const heading = document.createElement('h4');
    heading.textContent = title;
    sectionDiv.appendChild(heading);

    if (dataRows && dataRows.length > 0 && columnConfig && columnConfig.length > 0) {
        const table = document.createElement('table');
        table.className = 'results-table mini-table'; // Use existing styles
        table.id = tableId;
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');
        tbody.id = `${tableId}-tbody`;
        const headerRow = document.createElement('tr');

        // Store original data for sorting
        table.originalData = [...dataRows];
        table.currentSortColumn = null;
        table.currentSortDirection = 'asc';

        columnConfig.forEach((col, index) => {
            const th = document.createElement('th');
            th.textContent = col.header;
            
            // Add sorting functionality for sortable columns
            if (col.sortable) {
                th.style.cursor = 'pointer';
                th.style.userSelect = 'none';
                th.title = `Click to sort by ${col.header}`;
                
                // Add sort indicator
                const sortIndicator = document.createElement('span');
                sortIndicator.className = 'sort-indicator';
                sortIndicator.innerHTML = ' ↕️'; // Default sort icon
                th.appendChild(sortIndicator);
                
                th.addEventListener('click', () => {
                    sortTable(table, col.key, col.format || 'string', tbody);
                });
            }
            
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Initial table population
        populateTableBody(tbody, dataRows, columnConfig);
        table.appendChild(tbody);
        sectionDiv.appendChild(table);
    } else {
        const p = document.createElement('p');
        p.textContent = `No ${title.toLowerCase()} data available.`;
        sectionDiv.appendChild(p);
    }
    
    return sectionDiv;
}

function sortTable(table, sortKey, dataType, tbody) {
    const currentColumn = table.currentSortColumn;
    const currentDirection = table.currentSortDirection;
    
    // Determine new sort direction
    let newDirection = 'asc';
    if (currentColumn === sortKey && currentDirection === 'asc') {
        newDirection = 'desc';
    }
    
    // Update sort state
    table.currentSortColumn = sortKey;
    table.currentSortDirection = newDirection;
    
    // Sort the data
    const sortedData = [...table.originalData].sort((a, b) => {
        let valA = a[sortKey];
        let valB = b[sortKey];
        
        // Handle null/undefined values
        if (valA === null || valA === undefined) valA = '';
        if (valB === null || valB === undefined) valB = '';
        
        // Convert based on data type
        if (dataType === 'float3' || dataType === 'float') {
            valA = parseFloat(valA) || 0;
            valB = parseFloat(valB) || 0;
        } else if (dataType === 'integer') {
            valA = parseInt(valA) || 0;
            valB = parseInt(valB) || 0;
        } else {
            // String comparison
            valA = String(valA).toLowerCase();
            valB = String(valB).toLowerCase();
        }
        
        // Compare values
        if (valA < valB) return newDirection === 'asc' ? -1 : 1;
        if (valA > valB) return newDirection === 'asc' ? 1 : -1;
        return 0;
    });
    
    // Update sort indicators
    const allHeaders = table.querySelectorAll('th .sort-indicator');
    allHeaders.forEach(indicator => {
        indicator.innerHTML = ' ↕️'; // Reset all indicators
    });
    
    // Set active sort indicator
    const activeHeader = Array.from(table.querySelectorAll('th')).find(th => 
        th.textContent.includes(table.querySelector(`th:nth-child(${Array.from(table.querySelectorAll('th')).findIndex(h => h.addEventListener && h.textContent.includes(sortKey.replace(/_/g, ' '))) + 1})`))
    );
    
    // Find the correct header and update indicator
    const headers = table.querySelectorAll('th');
    headers.forEach(th => {
        if (th.addEventListener) {
            const indicator = th.querySelector('.sort-indicator');
            if (indicator) {
                if (th.textContent.includes('Delta vs Typical') && sortKey === 'risk_score_delta_vs_typical_county') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                } else if (th.textContent.includes('Risk Score') && sortKey === 'risk_score_2023') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                } else if (th.textContent.includes('Enrollment') && sortKey === 'enrollment_2023') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                } else if (th.textContent.includes('Market Share') && sortKey === 'market_share_percentage_2023') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                } else if (th.textContent.includes('Contract ID') && sortKey === 'contract_id') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                } else if (th.textContent.includes('Plan ID') && sortKey === 'plan_id') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                } else if (th.textContent.includes('Plan Type') && sortKey === 'plan_type') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                } else if (th.textContent.includes('SNP Category') && sortKey === 'snp_status') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                } else if (th.textContent.includes('Typical County') && sortKey === 'typical_county_wtd_risk_score_2023') {
                    indicator.innerHTML = newDirection === 'asc' ? ' ↑' : ' ↓';
                }
            }
        }
    });
    
    // Re-populate table with sorted data
    const columnConfig = [
        { header: 'Contract ID', key: 'contract_id', sortable: true },
        { header: 'Plan ID', key: 'plan_id', sortable: true },
        { header: 'Plan Type', key: 'plan_type', sortable: true },
        { header: 'SNP Category', key: 'snp_status', sortable: true },
        { header: 'Enrollment', key: 'enrollment_2023', format: 'integer', sortable: true },
                                         { header: 'Market Share %', key: 'market_share_percentage_2023', format: 'percentage', sortable: true },
        { header: 'Risk Score', key: 'risk_score_2023', format: 'float3', sortable: true },
        { header: 'Typical County', key: 'typical_county_wtd_risk_score_2023', format: 'float3', sortable: true },
        { header: 'Delta vs Typical', key: 'risk_score_delta_vs_typical_county', format: 'float3', sortable: true }
    ];
    
    populateTableBody(tbody, sortedData, columnConfig);
}

function populateTableBody(tbody, dataRows, columnConfig) {
    // Clear existing rows
    tbody.innerHTML = '';
    
    dataRows.forEach(rowData => {
        const tr = document.createElement('tr');
        columnConfig.forEach(col => {
            const td = document.createElement('td');
            let value = rowData[col.key];
            
            let currentFormat = col.format;
            let currentPrecision = undefined;
            
            // Extract precision from format string like 'float3' or 'percentage1'
            if (typeof currentFormat === 'string') {
                const match = currentFormat.match(/^(float|percentage)(\d+)$/i);
                if (match) {
                    currentFormat = match[1].toLowerCase();
                    currentPrecision = parseInt(match[2], 10);
                }
            }
            
            if (value === null || typeof value === 'undefined') {
                value = '-';
            }
            
            if (value !== '-') {
                if (currentFormat === 'percentage') {
                    td.textContent = `${parseFloat(value).toFixed(currentPrecision || 1)}%`;
                } else if (currentFormat === 'integer') {
                    td.textContent = Number.isInteger(Number(value)) ? Number(value).toLocaleString() : parseFloat(value).toFixed(0);
                } else if (currentFormat === 'float') {
                    td.textContent = parseFloat(value).toFixed(currentPrecision || 3);
                } else {
                    td.textContent = value;
                }
            } else {
                td.textContent = value;
            }
            
            // Right-align numeric columns for better readability
            if (col.key !== 'contract_id' && col.key !== 'plan_id' && col.key !== 'plan_type' && col.key !== 'snp_status') {
                td.style.textAlign = 'center';
            }
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

async function fetchAndDisplayOrgDetails(rawOrgName) {
    const loadingIndicator = document.getElementById('parent-analyzer-loading');
    const errorDisplay = document.getElementById('parent-analyzer-error');
    const resultsContainer = document.getElementById('parent-analyzer-results');
    const yearSelectorArea = document.getElementById('contract-year-selector-area');
    const yearSelector = document.getElementById('contract-year-select');

    // Clear previous results and errors, show loading
    resultsContainer.innerHTML = '';
    loadingIndicator.style.display = 'block';
    errorDisplay.style.display = 'none';

    if (!rawOrgName) {
        console.warn("rawOrgName is null or empty in fetchAndDisplayOrgDetails. Displaying error.");
        if (errorDisplay) {
            errorDisplay.textContent = 'Please select a Parent Organization first.';
            errorDisplay.style.display = 'block';
            loadingIndicator.style.display = 'none';
        }
        console.log("--- fetchAndDisplayOrgDetails ended (no rawOrgName) ---");
        return;
    }

    const cleanedOrgNameProvisional = parentAnalyzerState.allParentOrgs.find(o => o.raw === rawOrgName)?.cleaned || rawOrgName;
    console.log("Provisional cleaned name for display:", cleanedOrgNameProvisional);
    
    // Get the selected year for contract details
    const selectedContractTableYear = yearSelector ? parseInt(yearSelector.value) : CONTRACT_TABLE_AVAILABLE_YEARS[0];

    try {
        const apiUrl = `${API_BASE_URL}/api/parent-organization-details?org_name_raw=${encodeURIComponent(rawOrgName)}&target_year=${selectedContractTableYear}`;
        console.log("Fetching from API URL:", apiUrl);
        const response = await fetch(apiUrl);
        console.log("API response received. Status:", response.status);

        if (!response.ok) {
            let errorData = { detail: "Unknown error during fetch." };
            try {
                errorData = await response.json();
                console.error("API response not OK. Error data from JSON:", errorData);
            } catch (jsonError) {
                console.error("API response not OK, and failed to parse error JSON:", jsonError);
                errorData.detail = await response.text().catch(() => "Could not get error text.");
                console.error("API response not OK. Error text:", errorData.detail);
            }
            throw new Error(`HTTP error ${response.status}: ${errorData.detail || "Failed to fetch details"}`);
        }
        
        const details = await response.json();
        console.log("API response JSON parsed successfully. Details:", details);

        resultsContainer.innerHTML = ''; // Clear loading message

        if (details.errors && details.errors.length > 0) {
            console.warn("API returned errors in details object:", details.errors);
            if (errorDisplay) { // Check if errorDisplay is available
                errorDisplay.textContent = `Error: ${details.errors.join(', ')}`;
                errorDisplay.style.display = 'block'; // Ensure it is visible
            } else {
                alert(`Error: ${details.errors.join(', ')}`);
            }
        }

        console.log("Rendering details HTML into grid sections...");
        
        // Overall title for the analyzed organization
        const orgTitle = document.createElement('h3');
        orgTitle.textContent = `${details.organization_name_cleaned} (Raw: ${details.organization_name_raw})`;
        orgTitle.style.textAlign = 'center';
        orgTitle.style.gridColumn = "1 / -1"; // Span full width if grid has multiple columns
        resultsContainer.appendChild(orgTitle);

        // Show the year selector once data is being displayed
        if (yearSelectorArea) {
            yearSelectorArea.style.display = 'block';
        }

        // --- Log data before attempting to render chart --- 
        console.log("[DEBUG-PRE-CHART] details.chart_years:", JSON.stringify(details.chart_years));
        console.log("[DEBUG-PRE-CHART] details.chart_total_enrollment:", JSON.stringify(details.chart_total_enrollment));
        console.log("[DEBUG-PRE-CHART] details.chart_weighted_avg_risk_score:", JSON.stringify(details.chart_weighted_avg_risk_score));
        console.log("[DEBUG-PRE-CHART] details.chart_weighted_avg_county_risk_score:", JSON.stringify(details.chart_weighted_avg_county_risk_score));
        console.log("[DEBUG-PRE-CHART] FULL details object:", JSON.stringify(details, null, 2));

        // --- Create and Render Three Separate Charts ---
        if (details.chart_years && details.chart_total_enrollment && details.chart_weighted_avg_risk_score) {
            
            // Create a section title
            const chartsTitle = document.createElement('h4');
            chartsTitle.textContent = 'Risk Score Performance Analysis';
            chartsTitle.style.textAlign = 'center';
            chartsTitle.style.marginTop = '20px';
            chartsTitle.style.marginBottom = '10px';
            chartsTitle.style.gridColumn = "1 / -1";
            resultsContainer.appendChild(chartsTitle);
            
            // Definition of getPaddedMinMax (ensure it's here and correctly scoped)
            function getPaddedMinMax(dataArray, isRiskScoreData) {
                const filteredData = dataArray.filter(val => val !== null && typeof val !== 'undefined');
                if (filteredData.length === 0) return { min: undefined, max: undefined };

                let minVal = Math.min(...filteredData); // Initialized to actual data min
                let maxVal = Math.max(...filteredData); // Initialized to actual data max
                
                const actualRange = maxVal - minVal;
                const midPoint = (minVal + maxVal) / 2;

                if (isRiskScoreData) {
                    // Risk score logic - modifies minVal and maxVal directly
                    if (actualRange === 0) { 
                        minVal = midPoint - 0.01; 
                        maxVal = midPoint + 0.01;
                    } else {
                        const paddingAmount = actualRange * 0.05; 
                        minVal -= paddingAmount; 
                        maxVal += paddingAmount; 
                        
                        let currentVisualSpan = maxVal - minVal;
                        const minimumSensibleSpan = 0.025; 
                        if (currentVisualSpan < minimumSensibleSpan) {
                           const adjustment = (minimumSensibleSpan - currentVisualSpan) / 2;
                           minVal -= adjustment;
                           maxVal += adjustment;
                        }
                        // Ensure a minimum visual span of 0.01 centered around midpoint if too small
                        if ((maxVal - minVal) < 0.01) { 
                            minVal = midPoint - 0.005;
                            maxVal = midPoint + 0.005;
                        }
                    }
                } else { // Enrollment data
                    // minVal is already Math.min(...filteredData), so it remains the actual data minimum.
                    // We only need to adjust maxVal for padding on top.
                    
                    if (actualRange === 0) {
                        if (maxVal === 0) { // All data points are 0 (maxVal is actualDataMax here, minVal is 0)
                            maxVal = minVal + 1000; // Provide a small range, e.g., 0 to 1000
                        } else { // All data points are the same non-zero value
                            maxVal = maxVal + (maxVal * 0.10); // Add 10% padding on top
                        }
                    } else { // Data has a range
                        maxVal = maxVal + (actualRange * 0.20); // Add 20% of the range as padding to the max
                    }
                    // minVal for enrollment remains the direct Math.min(...filteredData)
                }
                return { min: minVal, max: maxVal };
            }

            // Function to create and render a dual-axis chart
            function createDualAxisChart(containerId, canvasId, chartInstanceVar, enrollmentData, riskScoreData, countyRiskData, countyRiskDataExContract, chartTitle, segmentType) {
                let chartContainer = document.getElementById(containerId);
                let canvasElement;

                if (!chartContainer) {
                    chartContainer = document.createElement('div');
                    chartContainer.id = containerId;
                    chartContainer.style.width = '100%';
                    chartContainer.style.marginBottom = '30px';
                    resultsContainer.appendChild(chartContainer);
                }
                chartContainer.innerHTML = '';
                chartContainer.style.height = '400px';

                canvasElement = document.createElement('canvas');
                canvasElement.id = canvasId;
                chartContainer.appendChild(canvasElement);

                // Destroy existing chart instance if it exists
                if (window[chartInstanceVar]) {
                    window[chartInstanceVar].destroy();
                }

                const enrollmentAxisMeta = getPaddedMinMax(enrollmentData, false);
                
                // Combine all risk score datasets to determine proper axis bounds
                let combinedRiskScoreData = [...riskScoreData];
                if (countyRiskData) {
                    combinedRiskScoreData = combinedRiskScoreData.concat(countyRiskData);
                }
                if (countyRiskDataExContract) {
                    combinedRiskScoreData = combinedRiskScoreData.concat(countyRiskDataExContract);
                }
                const riskScoreAxisMeta = getPaddedMinMax(combinedRiskScoreData, true);

                // Build datasets array dynamically
                const datasets = [
                    {
                        label: `${segmentType} Enrollment`,
                        data: enrollmentData,
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        yAxisID: 'y-axis-enrollment', 
                        tension: 0.1,
                        type: 'line'
                    },
                    {
                        label: `${segmentType} Risk Score (Actual)`,
                        data: riskScoreData,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        yAxisID: 'y-axis-risk-score',
                        tension: 0.1,
                        type: 'line'
                    }
                ];

                // Add purple line only if county risk score data exists
                if (countyRiskData && countyRiskData.length > 0) {
                    datasets.push({
                        label: `Expected ${segmentType} County Risk Score`,
                        data: countyRiskData,
                        borderColor: 'rgb(128, 0, 128)', // Purple color
                        backgroundColor: 'rgba(128, 0, 128, 0.3)',
                        yAxisID: 'y-axis-risk-score',
                        tension: 0.1,
                        type: 'line'
                    });
                }

                // Add green dashed line for ex-contract county risk score data
                if (countyRiskDataExContract && countyRiskDataExContract.length > 0) {
                    datasets.push({
                        label: `Expected ${segmentType} County Risk Score (Ex-Contract)`,
                        data: countyRiskDataExContract,
                        borderColor: 'rgb(16, 185, 129)', // Green color
                        backgroundColor: 'rgba(16, 185, 129, 0.3)',
                        yAxisID: 'y-axis-risk-score',
                        tension: 0.1,
                        type: 'line',
                        borderDash: [5, 5] // Dashed line for distinction
                    });
                }

                const chartData = {
                    labels: details.chart_years,
                    datasets: datasets
                };

                const chartOptions = {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: chartTitle,
                            font: { size: 14 }
                        },
                        legend: {
                            position: 'top'
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Year'
                            }
                        },
                        'y-axis-enrollment': { 
                            type: 'linear',
                            display: true,
                            position: 'left',
                            min: enrollmentAxisMeta.min,
                            max: enrollmentAxisMeta.max,
                            title: {
                                display: true,
                                text: 'Enrollment'
                            },
                            ticks: {
                                callback: function(value) {
                                    if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                                    if (value >= 1000) return (value / 1000).toFixed(0) + 'K';
                                    return value.toLocaleString();
                                },
                                autoSkip: false,
                                maxTicksLimit: 8
                            }
                        },
                        'y-axis-risk-score': {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            min: riskScoreAxisMeta.min,
                            max: riskScoreAxisMeta.max,
                            title: {
                                display: true,
                                text: 'Risk Score (Actual vs Expected)'
                            },
                            ticks: {
                                autoSkip: false,
                                maxTicksLimit: 8,
                                precision: 3
                            },
                            grid: {
                                drawOnChartArea: false
                            }
                        }
                    }
                };

                window[chartInstanceVar] = new Chart(canvasElement, {
                    type: 'line',
                    data: chartData,

                    options: chartOptions
                });
            }

            // Create three separate charts
            
            // 1. Overall Performance Chart (All Beneficiaries)
            createDualAxisChart(
                'parent-org-chart-overall',
                'parentOrgOverallChart',
                'parentOrgOverallChartInstance',
                details.chart_total_enrollment,
                details.chart_weighted_avg_risk_score,
                details.chart_weighted_avg_county_risk_score,
                details.chart_weighted_avg_county_risk_score_ex_contract,
                `Overall Performance: ${details.organization_name_cleaned} (All Beneficiaries)`,
                'Total'
            );

            // 2. Traditional Beneficiaries Chart  
            if (details.chart_traditional_enrollment && details.chart_traditional_weighted_avg_risk_score) {
                createDualAxisChart(
                    'parent-org-chart-traditional',
                    'parentOrgTraditionalChart', 
                    'parentOrgTraditionalChartInstance',
                    details.chart_traditional_enrollment,
                    details.chart_traditional_weighted_avg_risk_score,
                    details.chart_traditional_weighted_avg_county_risk_score,
                    details.chart_traditional_weighted_avg_county_risk_score_ex_contract,
                    `Traditional Beneficiaries: ${details.organization_name_cleaned}`,
                    'Traditional'
                );
            }

            // 3. Dual Eligible Beneficiaries Chart
            if (details.chart_dual_enrollment && details.chart_dual_weighted_avg_risk_score) {
                createDualAxisChart(
                    'parent-org-chart-dual',
                    'parentOrgDualChart',
                    'parentOrgDualChartInstance', 
                    details.chart_dual_enrollment,
                    details.chart_dual_weighted_avg_risk_score,
                    details.chart_dual_weighted_avg_county_risk_score,
                    details.chart_dual_weighted_avg_county_risk_score_ex_contract,
                    `Dual Eligible Beneficiaries: ${details.organization_name_cleaned}`,
                    'Dual'
                );
            }
        } else {
            console.log("No chart data available or chart_years is empty.");
        }
        // --- End of Chart Block --- (Ensure this comment is not a block comment ender)

        // Helper function to create a table section more generically
        function createTableSection(title, dataRows, columnConfig, tableId) {
            const sectionDiv = document.createElement('div');
            const heading = document.createElement('h4');
            heading.textContent = title;
            sectionDiv.appendChild(heading);

            if (dataRows && dataRows.length > 0 && columnConfig && columnConfig.length > 0) {
                const table = document.createElement('table');
                table.className = 'results-table mini-table'; // Use existing styles
                const thead = document.createElement('thead');
                const tbody = document.createElement('tbody');
                const headerRow = document.createElement('tr');

                columnConfig.forEach(col => {
                    const th = document.createElement('th');
                    th.textContent = col.header;
                    headerRow.appendChild(th);
                });
                thead.appendChild(headerRow);
                table.appendChild(thead);

                dataRows.forEach(rowData => {
                    const tr = document.createElement('tr');
                    columnConfig.forEach(col => {
                        console.log(`[JS_TABLE_CELL_DEBUG] Processing col.key: ${col.key}, col.header: ${col.header}`); // Log column being processed
                        const td = document.createElement('td');
                        let value = rowData[col.key]; 
                        console.log(`[JS_TABLE_CELL_DEBUG]   Raw value for ${col.key}:`, value); // Log raw value
                        
                        // Handle separator row
                        if (rowData.isSeparator && rowData.Metric === "SECTION_BREAK_RISK_SCORES") {
                            if (col.key === 'Metric') {
                                td.colSpan = columnConfig.length;
                                td.textContent = 'Risk Score Metrics';
                                td.style.textAlign = 'center';
                                td.style.fontWeight = 'bold';
                                td.style.backgroundColor = '#f0f0f0'; // Light grey background for separator
                                td.style.paddingTop = '10px';
                                td.style.paddingBottom = '10px';
                                tr.appendChild(td);
                                return; // Skip other cells for this row
                            } else {
                                return; // Don't create other cells for separator row
                            }
                        }

                        let currentFormat = col.format; 
                        let currentPrecision = undefined; // Initialize precision

                        // Extract precision from format string like 'float3' or 'percentage1'
                        if (typeof currentFormat === 'string') {
                            const match = currentFormat.match(/^(float|percentage)(\d+)$/i);
                            if (match) {
                                currentFormat = match[1].toLowerCase(); // 'float' or 'percentage'
                                currentPrecision = parseInt(match[2], 10);
                            }
                        }

                        // Determine formatting based on Metric name and column key for PIVOTED tables
                        if (col.key !== 'Metric' && dataRows[0] && typeof dataRows[0].Metric === 'string') { // Check if Metric column exists for pivoted tables
                            if (rowData.Metric === 'YoY') {
                                currentFormat = 'percentage';
                                currentPrecision = currentPrecision === undefined ? 1 : currentPrecision; // Default to 1 if not from format string
                            } else if (rowData.Metric.includes('Performance vs Expected (%)')) {
                                // NEW: Format percentage delta rows
                                currentFormat = 'percentage';
                                currentPrecision = currentPrecision === undefined ? 1 : currentPrecision; // Default to 1 decimal place
                            } else if (rowData.Metric.includes('Performance vs Ex-Contract (%)')) {
                                // NEW: Format ex-contract percentage delta rows with same formatting as expected
                                currentFormat = 'percentage';
                                currentPrecision = currentPrecision === undefined ? 1 : currentPrecision; // Default to 1 decimal place
                            } else if (rowData.Metric.includes('Enrollment')) {
                                currentFormat = 'integer';
                            } else if (rowData.Metric.includes('Risk Score')) {
                                currentFormat = 'float';
                                currentPrecision = currentPrecision === undefined ? 3 : currentPrecision; // Default to 3 if not from format string
                            }
                        }

                        if (value === null || typeof value === 'undefined') {
                            value = '-';
                        }

                        if (value !== '-') {
                            if (currentFormat === 'percentage') {
                                td.textContent = `${parseFloat(value).toFixed(currentPrecision || 1)}%`;
                            } else if (currentFormat === 'integer') {
                                td.textContent = Number.isInteger(Number(value)) ? Number(value).toLocaleString() : parseFloat(value).toFixed(0);
                            } else if (currentFormat === 'float') {
                                td.textContent = parseFloat(value).toFixed(currentPrecision || 3);
                            } else {
                                td.textContent = value;
                            }
                        } else {
                            td.textContent = value;
                        }
                        
                        // Right-align numeric and percentage columns for better readability
                        if (col.key !== 'Metric' && !rowData.isSeparator) { // All year columns are effectively numeric/percentage
                            td.style.textAlign = 'center';
                        }
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                });
                table.appendChild(tbody);
                sectionDiv.appendChild(table);
            } else {
                const p = document.createElement('p');
                p.textContent = `No ${title.toLowerCase()} data available.`;
                sectionDiv.appendChild(p);
            }
            return sectionDiv;
        }

        // 1. Enrollment Metrics Table (Pivoted)
        // if (details.enrollment_metrics && details.enrollment_metrics.length > 0 && details.enrollment_metrics_columns && details.enrollment_metrics_columns.length > 0) {
        //     const enrollmentColumnConfigPivoted = details.enrollment_metrics_columns.map(colName => ({
        //         header: colName,
        //         key: colName // Key is the column name itself (e.g., "Metric", "2015", "2016")
        //     }));
        //     resultsContainer.appendChild(
        //         createTableSection('Enrollment Metrics (Pivoted: Years as Columns)', details.enrollment_metrics, enrollmentColumnConfigPivoted)
        //     );
        // }

        // 2. Risk Score Metrics Table (Pivoted)
        // if (details.risk_score_metrics && details.risk_score_metrics.length > 0 && details.risk_score_metrics_columns && details.risk_score_metrics_columns.length > 0) {
        //     const riskScoreColumnConfigPivoted = details.risk_score_metrics_columns.map(colName => ({
        //         header: colName,
        //         key: colName
        //     }));
        //     resultsContainer.appendChild(
        //         createTableSection('Risk Score Metrics (Pivoted: Years as Columns)', details.risk_score_metrics, riskScoreColumnConfigPivoted)
        //     );
        // }

        // --- Combine Enrollment and Risk Score Metrics into a Single Table for Alignment ---
        let combinedMetricsData = [];
        let combinedMetricsColumnsConfig = [];
        let mainTableTitle = "Key Metrics (Pivoted: Years as Columns)";

        if (details.enrollment_metrics && details.enrollment_metrics.length > 0 && details.enrollment_metrics_columns && details.enrollment_metrics_columns.length > 0) {
            combinedMetricsData.push(...details.enrollment_metrics);
            combinedMetricsColumnsConfig = details.enrollment_metrics_columns.map(colName => ({
                header: colName,
                key: colName
            }));

            // Add a visual separator row if risk score data also exists
            if (details.risk_score_metrics && details.risk_score_metrics.length > 0 && details.risk_score_metrics_columns && details.risk_score_metrics_columns.length > 0) {
                const separatorRow = { 
                    Metric: "SECTION_BREAK_RISK_SCORES", // Use a unique key for easy identification
                    isSeparator: true // Add a flag to identify this row type
                };
                // No need to add empty year columns for a separator that will span
                combinedMetricsData.push(separatorRow);
            }
        }

        if (details.risk_score_metrics && details.risk_score_metrics.length > 0 && details.risk_score_metrics_columns && details.risk_score_metrics_columns.length > 0) {
            combinedMetricsData.push(...details.risk_score_metrics);
            // Ensure column config is set if enrollment data was empty
            if (combinedMetricsColumnsConfig.length === 0) {
                combinedMetricsColumnsConfig = details.risk_score_metrics_columns.map(colName => ({
                    header: colName,
                    key: colName
                }));
            }
        }

        if (combinedMetricsData.length > 0 && combinedMetricsColumnsConfig.length > 0) {
            resultsContainer.appendChild(
                createTableSection(mainTableTitle, combinedMetricsData, combinedMetricsColumnsConfig, 'combined-metrics-table')
            );
        }
        // --- End of Combined Table ---

        // --- NEW: 3 Separate Performance Analysis Tables ---
        
        // Table 1: Overall Performance Analysis
        if (details.overall_performance_metrics && details.overall_performance_metrics.length > 0 && details.overall_performance_metrics_columns && details.overall_performance_metrics_columns.length > 0) {
            const overallPerformanceColumnConfig = details.overall_performance_metrics_columns.map(colName => ({
                header: colName,
                key: colName
            }));
            resultsContainer.appendChild(
                createTableSection('Overall Performance Analysis (Actual vs Expected)', details.overall_performance_metrics, overallPerformanceColumnConfig, 'overall-performance-analysis-table')
            );
        }
        
        // Table 2: Traditional Performance Analysis  
        if (details.traditional_performance_metrics && details.traditional_performance_metrics.length > 0 && details.traditional_performance_metrics_columns && details.traditional_performance_metrics_columns.length > 0) {
            const traditionalPerformanceColumnConfig = details.traditional_performance_metrics_columns.map(colName => ({
                header: colName,
                key: colName
            }));
            resultsContainer.appendChild(
                createTableSection('Traditional Performance Analysis (Actual vs Expected)', details.traditional_performance_metrics, traditionalPerformanceColumnConfig, 'traditional-performance-analysis-table')
            );
        }
        
        // Table 3: Dual Performance Analysis
        if (details.dual_performance_metrics && details.dual_performance_metrics.length > 0 && details.dual_performance_metrics_columns && details.dual_performance_metrics_columns.length > 0) {
            const dualPerformanceColumnConfig = details.dual_performance_metrics_columns.map(colName => ({
                header: colName,
                key: colName
            }));
            resultsContainer.appendChild(
                createTableSection('Dual Performance Analysis (Actual vs Expected)', details.dual_performance_metrics, dualPerformanceColumnConfig, 'dual-performance-analysis-table')
            );
        }
        
        // --- End of Performance Analysis Tables ---

        // --- Year Selector Creation (Moved here, to be prepended to the Contract/Plan table section) ---
        const yearSelectorContainer = document.createElement('div');
        yearSelectorContainer.id = 'contract-year-selector-area'; // Keep ID for potential styling
        yearSelectorContainer.style.marginTop = '20px';
        yearSelectorContainer.style.marginBottom = '10px'; // Add some space below it

        const yearSelectorLabel = document.createElement('label');
        yearSelectorLabel.htmlFor = 'contract-year-select';
        yearSelectorLabel.textContent = 'Select Year for Contract/Plan Details: ';
        yearSelectorContainer.appendChild(yearSelectorLabel);

        const yearSelector = document.createElement('select');
        yearSelector.id = 'contract-year-select';
        CONTRACT_TABLE_AVAILABLE_YEARS.forEach(year => {
            const option = document.createElement('option');
            option.value = year;
            option.textContent = year;
            if (year === selectedContractTableYear) { // Set selected based on API call
                option.selected = true;
            }
            yearSelector.appendChild(option);
        });
        yearSelector.addEventListener('change', () => {
            if (currentParentOrgSelection && currentParentOrgSelection.rawName) {
                fetchAndDisplayOrgDetails(currentParentOrgSelection.rawName); // Refetch with new year
            }
        });
        yearSelectorContainer.appendChild(yearSelector);

        // --- Create and Add Output to CSV Button ---
        const outputCsvButton = document.createElement('button');
        outputCsvButton.id = 'output-contract-plan-csv';
        outputCsvButton.textContent = 'Output to Excel';
        outputCsvButton.style.marginLeft = '10px'; // Add some space next to the dropdown
        
        // Define contractPlanColumns here so it's in scope for the listener
        let contractPlanColumns = []; 

        outputCsvButton.addEventListener('click', () => {
            // Data for this table is in 'contractPlanData' variable available in this scope
            // Column config will be populated if data exists
            if (contractPlanData && contractPlanData.length > 0) {
                // Populate contractPlanColumns if not already (it should be if data is present for the table)
                // This re-definition within the click listener ensures it uses the latest column config 
                // tied to the displayed table, especially if column headers change dynamically with the year.
                const actualContractYearForTableFromDetails = details.selected_contract_year || selectedContractTableYear;
                contractPlanColumns = [
                    { header: 'Contract ID', key: 'contract_id', sortable: true },
                    { header: 'Plan ID', key: 'plan_id', sortable: true },
                    { header: 'Plan Type', key: 'plan_type', sortable: true },
                    { header: 'SNP Category', key: 'snp_status', sortable: true },
                    { header: `${actualContractYearForTableFromDetails} Enrollment`, key: 'enrollment_2023', format: 'integer', sortable: true },
                    { header: `Market Share % ${actualContractYearForTableFromDetails}`, key: 'market_share_percentage_2023', format: 'percentage', sortable: true },
                    { header: `${actualContractYearForTableFromDetails} Risk Score`, key: 'risk_score_2023', format: 'float3', sortable: true },
                    { header: `County Risk Score (${actualContractYearForTableFromDetails})`, key: 'typical_county_wtd_risk_score_2023', format: 'float3', sortable: true },
                    { header: `County Risk Score (Ex-Contract)`, key: 'county_wtd_risk_score_ex_contract_2023', format: 'float3', sortable: true },
                    { header: `Delta vs County`, key: 'risk_score_delta_vs_typical_county', format: 'float3', sortable: true },
                    { header: `Delta vs Ex-Contract`, key: 'risk_score_delta_vs_ex_contract', format: 'float3', sortable: true }
                ];

                const cleanedOrgName = details.organization_name_cleaned || "ParentOrg";
                const year = selectedContractTableYear || "data"; // selectedContractTableYear should be correct here
                const fileName = `${cleanedOrgName}_ContractPlanEnrollment_${year}.csv`;
                exportTableDataToCSV(contractPlanData, contractPlanColumns, fileName);
            } else {
                alert("No data in the Contract & Plan Enrollment table to export.");
            }
        });
        yearSelectorContainer.appendChild(outputCsvButton);
        // --- End Output to CSV Button ---

        // --- End Year Selector Creation ---

        // 3. New Table: Contract & Plan Enrollment for 2023
        const contractPlanData = details.contract_plan_enrollment_2023 || [];
        console.log("[JS_DEBUG] Contract Plan Data for Table:", JSON.stringify(contractPlanData.slice(0, 5), null, 2)); // Log first 5 rows

        // Use selected_contract_year from API response for the table title
        const actualContractYearForTable = details.selected_contract_year || selectedContractTableYear;
        const contractPlanTableTitle = `Contract & Plan Enrollment for ${actualContractYearForTable} (Sorted by Enrollment)`;

        if (contractPlanData.length > 0) {
            // Define column configuration for creating the table (this can be separate from CSV export's needs if they diverge)
            const displayColumns = [
                { header: 'Contract ID', key: 'contract_id', sortable: true },
                { header: 'Plan ID', key: 'plan_id', sortable: true },
                { header: 'Plan Type', key: 'plan_type', sortable: true },
                { header: 'SNP Category', key: 'snp_status', sortable: true },
                { header: `${actualContractYearForTable} Enrollment`, key: 'enrollment_2023', format: 'integer', sortable: true },
                { header: `Market Share % ${actualContractYearForTable}`, key: 'market_share_percentage_2023', format: 'percentage', sortable: true },
                { header: `${actualContractYearForTable} Risk Score`, key: 'risk_score_2023', format: 'float3', sortable: true },
                { header: `County Risk Score (${actualContractYearForTable})`, key: 'typical_county_wtd_risk_score_2023', format: 'float3', sortable: true },
                { header: `County Risk Score (Ex-Contract)`, key: 'county_wtd_risk_score_ex_contract_2023', format: 'float3', sortable: true },
                { header: `Delta vs County`, key: 'risk_score_delta_vs_typical_county', format: 'float3', sortable: true },
                { header: `Delta vs Ex-Contract`, key: 'risk_score_delta_vs_ex_contract', format: 'float3', sortable: true }
            ];
            // Store the original data for this table for sorting
            originalTableData = contractPlanData.map(row => ({ ...row }));
            currentSortState = {};

            const contractPlanTableSection = createSortableTableSection(
                contractPlanTableTitle,
                contractPlanData,
                displayColumns, // Use displayColumns for table rendering
                'contract-plan-enrollment-2023-table'
            );
            
            // Prepend the year selector to this specific section
            contractPlanTableSection.insertBefore(yearSelectorContainer, contractPlanTableSection.firstChild);
            resultsContainer.appendChild(contractPlanTableSection);
        } else {
            // Optionally, display a message if no contract/plan data for 2023
            const noContractsPlansMsg = document.createElement('p');
            noContractsPlansMsg.textContent = `No specific contract and plan enrollment data found for ${details.organization_name_cleaned} in ${actualContractYearForTable}.`;
            noContractsPlansMsg.style.gridColumn = "1 / -1";
            noContractsPlansMsg.style.textAlign = "center";
            noContractsPlansMsg.style.marginTop = "20px";
            resultsContainer.appendChild(noContractsPlansMsg);
        }

        console.log("New metric tables rendered into grid.");

    } catch (error) {
        console.error("--- Error in fetchAndDisplayOrgDetails catch block ---");
        console.error("Error object:", error);
        console.error("Error message:", error.message);
        resultsContainer.innerHTML = ''; // Clear any loading message
        if (errorDisplay) { // Check if errorDisplay is available
            errorDisplay.textContent = `Failed to load details: ${error.message}`;
            errorDisplay.style.display = 'block'; // Ensure it is visible
        } else {
            // Fallback if the specific errorDisplay element isn't found (though it should be)
            alert(`Failed to load details: ${error.message}`);
        }
    }
    console.log("--- fetchAndDisplayOrgDetails finished ---");
    loadingIndicator.style.display = 'none';
}

function exportTableDataToCSV(data, columns, fileName) {
    if (!data || data.length === 0) {
        console.warn("No data provided to exportTableDataToCSV");
        return;
    }

    // Extract headers from the column configuration
    const headers = columns.map(col => col.header).join(',');

    // Extract data rows using column keys
    const rows = data.map(rowData => {
        return columns.map(col => {
            let value = rowData[col.key];
            // Basic handling for null/undefined and potential commas (wrap in quotes)
            if (value === null || typeof value === 'undefined') {
                value = '';
            } else {
                value = String(value);
                if (value.includes(',')) {
                    value = `"${value}"`; // Simple CSV quoting
                }
            }
            return value;
        }).join(',');
    });

    const csvContent = [headers, ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");

    if (link.download !== undefined) { // feature detection
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", fileName);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } else {
        // Fallback for older browsers (e.g., IE)
        navigator.msSaveBlob(blob, fileName);
    }
}

// We don't need the global selectedParentOrgAnalyzerRaw variable anymore
// let selectedParentOrgAnalyzerRaw = null;
// Modify setupSimpleParentOrgFilter to update this global or pass a callback.
// For simplicity, let's assume parentAnalyzerState.selectedParentOrgRaw will be used directly by analyze button's listener for now. 