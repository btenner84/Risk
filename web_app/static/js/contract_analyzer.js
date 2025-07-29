let currentContractId = null;
let plansFetchedForContract = null;

document.addEventListener('DOMContentLoaded', () => {
    console.log('Contract Analyzer JS loaded');

    const analyzeButton = document.getElementById('analyze-contract-button');
    const contractIdInput = document.getElementById('contract-id-search');
    const planIdFilterGroup = document.getElementById('plan-id-filter-group');
    const planIdSelect = document.getElementById('plan-id-select');
    const resultsContainer = document.getElementById('contract-analyzer-results');

    if (analyzeButton) {
        analyzeButton.addEventListener('click', async () => {
            const enteredContractId = contractIdInput.value.trim().toUpperCase();

            if (!enteredContractId) {
                resultsContainer.innerHTML = '<p class="error-text" style="display: block;">Please enter a Contract ID.</p>';
                hidePlanSelector();
                return;
            }

            // If contract ID changed, or plans not fetched yet for this contract
            if (enteredContractId !== currentContractId || !plansFetchedForContract) {
                currentContractId = enteredContractId;
                plansFetchedForContract = false; // Reset flag
                hidePlanSelector();
                resultsContainer.innerHTML = ''; // Clear previous results
                
                // --- Step 1: Fetch Plan IDs for the Contract ID ---
                resultsContainer.innerHTML = `<p class="loading-text">Fetching plans for Contract ID: ${currentContractId}...</p>`;
                try {
                    const response = await fetch(`/api/plans-for-contract?contract_id=${encodeURIComponent(currentContractId)}`);
                    if (!response.ok) {
                        let errorDetail = "Unknown error fetching plans.";
                        try {
                            const errorData = await response.json();
                            errorDetail = errorData.detail || `Failed to fetch plans. Status: ${response.status}`;
                        } catch (e) {
                            errorDetail = `Failed to fetch plans. Status: ${response.status}, Response not JSON.`;
                        }
                        throw new Error(errorDetail);
                    }
                    const planData = await response.json(); 
                    const planIds = planData.plan_ids;

                    if (planIds && planIds.length > 0) {
                        populatePlanSelector(planIds);
                        planIdFilterGroup.style.display = 'block';
                        resultsContainer.innerHTML = '<p>Please select a Plan ID to analyze.</p>';
                        analyzeButton.textContent = 'Analyze Selected Plan';
                        plansFetchedForContract = true;
                    } else {
                        resultsContainer.innerHTML = '<p class="error-text" style="display: block;">No plans found for this Contract ID, or contract does not exist.</p>';
                        currentContractId = null; // Reset so user can try new contract ID
                    }
                } catch (error) {
                    resultsContainer.innerHTML = `<p class="error-text" style="display: block;">Error fetching plans: ${error.message}</p>`;
                    console.error('Error fetching plans for contract:', error);
                    currentContractId = null; // Reset
                }
            } else {
                // --- Step 2: Plans already fetched, now fetch details for selected plan ---
                const selectedPlanId = planIdSelect.value;
                if (!selectedPlanId) {
                    resultsContainer.innerHTML = '<p class="error-text" style="display: block;">Please select a Plan ID from the dropdown.</p>';
                    return;
                }
                fetchAndDisplayPlanDetails(currentContractId, selectedPlanId);
            }
        });
    }

    // Optional: If contract ID input changes, reset plan selection
    if (contractIdInput) {
        contractIdInput.addEventListener('input', () => {
            if (contractIdInput.value.trim().toUpperCase() !== currentContractId) {
                hidePlanSelector();
                analyzeButton.textContent = 'Fetch Plans / Analyze';
                plansFetchedForContract = false;
                // resultsContainer.innerHTML = '<p>Enter a Contract ID and click \'Fetch Plans / Analyze\'.</p>';
            }
        });
    }
});

function hidePlanSelector() {
    const planIdFilterGroup = document.getElementById('plan-id-filter-group');
    const planIdSelect = document.getElementById('plan-id-select');
    planIdFilterGroup.style.display = 'none';
    planIdSelect.innerHTML = ''; // Clear options
}

function populatePlanSelector(planIds) {
    const planIdSelect = document.getElementById('plan-id-select');
    planIdSelect.innerHTML = ''; // Clear existing options
    
    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = '-- Select a Plan ID --';
    planIdSelect.appendChild(defaultOption);

    planIds.forEach(planId => {
        const option = document.createElement('option');
        option.value = planId;
        option.textContent = planId;
        planIdSelect.appendChild(option);
    });
}

async function fetchAndDisplayPlanDetails(contractId, planId) { // Renamed from fetchAndDisplayContractDetails
    const resultsContainer = document.getElementById('contract-analyzer-results');
    resultsContainer.innerHTML = `<p class="loading-text">Fetching details for Contract ID: ${contractId}, Plan ID: ${planId}...</p>`;

    try {
        const response = await fetch(`/api/plan-details?contract_id=${encodeURIComponent(contractId)}&plan_id=${encodeURIComponent(planId)}`);
        if (!response.ok) {
            let errorDetail = "Unknown error fetching plan details.";
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || `Failed to fetch plan details. Status: ${response.status}`;
            } catch (e) {
                errorDetail = `Failed to fetch plan details. Status: ${response.status}, Response not JSON.`;
            }
            throw new Error(errorDetail);
        }
        const details = await response.json();

        renderPlanDetails(details); // Renamed from renderContractDetails

    } catch (error) {
        resultsContainer.innerHTML = `<p class="error-text" style="display: block;">Error: ${error.message}</p>`;
        console.error('Error fetching or displaying plan details:', error);
    }
}

function renderPlanDetails(details) { // Renamed from renderContractDetails
    const resultsContainer = document.getElementById('contract-analyzer-results');
    resultsContainer.innerHTML = ''; // Clear previous results/loading

    // --- Create a dedicated container for the chart ---
    const chartContainer = document.createElement('div');
    chartContainer.id = 'contract-plan-risk-chart-container'; // Match the ID in HTML
    // chartContainer.style.height = '400px'; // Set height via CSS later for better maintainability
    // chartContainer.style.width = '100%'; 
    // chartContainer.style.marginBottom = '30px';
    resultsContainer.appendChild(chartContainer);

    const chartCanvas = document.createElement('canvas');
    chartContainer.appendChild(chartCanvas);

    const mainTitleText = `Details for Contract ID: ${details.contract_id_cleaned}, Plan ID: ${details.plan_id_cleaned}`;
    const overallTitle = document.createElement('h3');
    overallTitle.textContent = mainTitleText;
    overallTitle.style.gridColumn = "1 / -1"; 
    overallTitle.style.textAlign = "center";
    resultsContainer.appendChild(overallTitle);

    // --- Master Comparison Table (New - Render this first) ---
    if (details.master_comparison_data && details.master_comparison_data.length > 0 && details.master_comparison_columns && details.master_comparison_columns.length > 0) {
        const masterTableTitleText = `Plan vs. Market Risk Score Comparison (Plan Footprint - C: ${details.contract_id_cleaned}, P: ${details.plan_id_cleaned})`;
        const masterColumnConfig = details.master_comparison_columns.map(colName => {
            let config = { header: colName, key: colName };
            if (colName !== 'Metric') { // Default for year columns (will be overridden by row type)
                config.format = 'float'; // Default to float for risk scores
                config.precision = 3;
            }
            return config;
        });

        // Custom formatting logic within createTableSection or by modifying data before passing it
        // For now, we adjust createTableSection's internal logic slightly or rely on specific row key checks.
        // The createTableSection will need to know how to format based on the 'Metric' name or a new config property.

        // Let's refine the column config based on metric type for more precise formatting in createTableSection
        const refinedMasterData = details.master_comparison_data.map(row => {
            const newRow = {...row};
            if (row.Metric === "Specific Plan's Total Enrollment" || row.Metric === "Total Addressable Market (Contract)") {
                details.master_comparison_columns.forEach(colName => {
                    if (colName !== 'Metric' && newRow[colName] !== null && typeof newRow[colName] !== 'undefined') {
                        // Ensure integer formatting for these rows if values exist
                        // The createTableSection will handle .toLocaleString() if format is 'integer'
                    }
                });
            }
            return newRow;
        });

        const masterTableSection = createTableSection(
            masterTableTitleText,
            refinedMasterData, // Use potentially refined data
            masterColumnConfig // Pass the generic column config, formatting decisions made in createTableSection based on Metric
        );
        resultsContainer.appendChild(masterTableSection);
    } else {
        const pMaster = document.createElement('p');
        pMaster.textContent = 'Master comparison data is not available.';
        pMaster.style.textAlign = 'center';
        resultsContainer.appendChild(pMaster);
    }

    // --- Chart Rendering Logic ---
    try {
        if (details.master_comparison_data && details.master_comparison_data.length > 0 && details.master_comparison_columns) {
            // Extract data for the chart from master_comparison_data
            let years = [];
            if (details.master_comparison_columns && details.master_comparison_columns.length > 0) {
                // Filter out 'Metric' and '2021' for chart labels
                years = details.master_comparison_columns.filter(col => col !== 'Metric' && col !== '2021');
            }

            let planRiskData = [];
            let marketRiskData = [];
            let marketRiskExclData = [];

            if (details.master_comparison_data && details.master_comparison_data.length > 0) {
                console.log("All master comparison rows:", details.master_comparison_data);
                details.master_comparison_data.forEach((row, index) => {
                    console.log(`Row ${index}:`, row.Metric);
                    if (row.Metric === "Specific Plan's Avg. Risk Score") {
                        // Extract data for years excluding '2021'
                        planRiskData = years.map(year => row[year]);
                    }
                    if (row.Metric === "Weighted Avg. Market Risk (Plan Footprint)") {
                        marketRiskData = years.map(year => row[year]);
                    }
                    if (row.Metric === "Weighted Avg. Market Risk (Excl. Current Contract)") {
                        console.log("Found excluded market risk row:", row);
                        marketRiskExclData = years.map(year => row[year]);
                    }
                });
            }

            console.log("Chart Years (excluding 2021):", years);
            console.log("Plan Risk Data (excluding 2021):", planRiskData);
            console.log("Market Risk Data (excluding 2021):", marketRiskData);
            console.log("Market Risk Excl Data (excluding 2021):", marketRiskExclData);

            if (years.length > 0 && planRiskData.length > 0 && marketRiskData.length > 0) {
                // Helper function to get min/max with padding for chart axes
                function getPaddedMinMaxContract(dataArrays, paddingPercentage = 0.1) { // Takes an array of data arrays
                    const allFilteredData = [].concat(...dataArrays).filter(val => val !== null && typeof val !== 'undefined' && !Number.isNaN(val));
                    if (allFilteredData.length === 0) return { min: undefined, max: undefined };

                    let minVal = Math.min(...allFilteredData);
                    let maxVal = Math.max(...allFilteredData);
                    const range = maxVal - minVal;

                    if (range === 0) { 
                        minVal -= Math.abs(minVal * paddingPercentage) || 0.1; 
                        maxVal += Math.abs(maxVal * paddingPercentage) || 0.1;
                    } else {
                        minVal -= range * paddingPercentage;
                        maxVal += range * paddingPercentage;
                    }
                    minVal = Math.max(0, minVal); // Ensure risk scores don't go below 0 on axis
                    return { min: minVal, max: maxVal };
                }

                // Calculate bounds based on all three datasets for a single Y-axis
                const allDatasets = [planRiskData, marketRiskData];
                if (marketRiskExclData.length > 0) {
                    allDatasets.push(marketRiskExclData);
                }
                const combinedAxisBounds = getPaddedMinMaxContract(allDatasets, 0.15);

                const datasets = [
                    {
                        label: 'Specific Plan Avg. Risk Score',
                        data: planRiskData,
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        yAxisID: 'y-axis-risk',
                        tension: 0.1
                    },
                    {
                        label: 'Weighted Avg. Market Risk (Plan Footprint)',
                        data: marketRiskData,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        yAxisID: 'y-axis-risk',
                        tension: 0.1
                    }
                ];

                // Add third dataset if data is available
                console.log("marketRiskExclData length:", marketRiskExclData.length);
                console.log("marketRiskExclData content:", marketRiskExclData);
                if (marketRiskExclData.length > 0) {
                    console.log("Adding third dataset to chart");
                    datasets.push({
                        label: 'Weighted Avg. Market Risk (Excl. Current Contract)',
                        data: marketRiskExclData,
                        borderColor: 'rgb(153, 102, 255)',
                        backgroundColor: 'rgba(153, 102, 255, 0.5)',
                        yAxisID: 'y-axis-risk',
                        tension: 0.1
                    });
                } else {
                    console.log("NOT adding third dataset - no data");
                }

                console.log("Final datasets array:", datasets);
                console.log("Number of datasets:", datasets.length);
                
                const chartData = {
                    labels: years,
                    datasets: datasets
                };

                if (window.contractPlanRiskChartInstance) {
                    window.contractPlanRiskChartInstance.destroy();
                }

                window.contractPlanRiskChartInstance = new Chart(chartCanvas, {
                    type: 'line',
                    data: chartData,
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            mode: 'index',
                            intersect: false,
                        },
                        stacked: false,
                        plugins: {
                            title: {
                                display: true,
                                text: `Plan vs. Market Risk Score Over Time (C: ${details.contract_id_cleaned}, P: ${details.plan_id_cleaned})`,
                                font: { size: 16 }
                            },
                            legend: {
                                position: 'top',
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Year'
                                }
                            },
                            'y-axis-risk': { // Renamed from 'y-axis-plan-risk' to be more generic
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: {
                                    display: true,
                                    text: 'Risk Score' // Generic Y-axis title
                                },
                                min: combinedAxisBounds.min, // Use combined bounds
                                max: combinedAxisBounds.max, // Use combined bounds
                                grid: {
                                    drawOnChartArea: true 
                                }
                            }
                        }
                    }
                });
            } else {
                chartContainer.innerHTML = '<p class="chart-unavailable-text" style="text-align:center;">Risk score trend data for chart is not available.</p>';
            }
        } else {
            chartContainer.innerHTML = '<p class="chart-unavailable-text" style="text-align:center;">Master comparison data for chart is not available.</p>';
        }
    } catch (error) {
        console.error("Error rendering chart:", error);
        chartContainer.innerHTML = '<p class="chart-unavailable-text" style="text-align:center; color:red;">Error rendering chart.</p>';
    }
    // --- End Chart Rendering Logic ---

    const countySectionTitleText = `County Enrollment by Year for Plan ${details.plan_id_cleaned}`;
    const countySectionTitle = document.createElement('h4');
    countySectionTitle.textContent = countySectionTitleText;
    countySectionTitle.style.textAlign = 'center';
    countySectionTitle.style.marginTop = '30px';
    countySectionTitle.style.width = '100%';
    resultsContainer.appendChild(countySectionTitle);

    // --- New logic for pivoted county enrollment table ---
    const countyData = details.pivoted_county_enrollment_data;
    const countyColumns = details.pivoted_county_enrollment_columns;

    if (countyData && countyData.length > 0 && countyColumns && countyColumns.length > 1) {
        // Dynamically create columnConfig for the createTableSection helper
        const countyColumnConfig = countyColumns.map(colName => {
            let config = { header: colName, key: colName };
            if (colName !== 'County') { // Assume year columns should be formatted as integers
                config.format = 'integer';
            }
            // Add other specific formatting for 'County' if needed, otherwise default
            return config;
        });

        const pivotedCountyTableSection = createTableSection(
            null, // Title is handled by countySectionTitle already
            countyData,
            countyColumnConfig
        );
        resultsContainer.appendChild(pivotedCountyTableSection);
    } else {
        const p = document.createElement('p');
        p.textContent = 'No county-level enrollment data available for this plan.';
        p.style.textAlign = 'center';
        resultsContainer.appendChild(p);
    }

    // --- New Section: Pivoted County-Wide Average Risk Scores ---
    const countyRiskSectionTitleText = `Market Average Risk Score (for Counties with Plan Presence - C: ${details.contract_id_cleaned}, P: ${details.plan_id_cleaned})`;
    const countyRiskSectionTitle = document.createElement('h4');
    countyRiskSectionTitle.textContent = countyRiskSectionTitleText;
    countyRiskSectionTitle.style.textAlign = 'center';
    countyRiskSectionTitle.style.marginTop = '30px';
    countyRiskSectionTitle.style.width = '100%';
    resultsContainer.appendChild(countyRiskSectionTitle);

    const countyRiskData = details.pivoted_county_risk_data;
    const countyRiskColumns = details.pivoted_county_risk_columns;

    if (countyRiskData && countyRiskData.length > 0 && countyRiskColumns && countyRiskColumns.length > 1) {
        const countyRiskColumnConfig = countyRiskColumns.map(colName => {
            let config = { header: colName, key: colName };
            if (colName !== 'County') { // Assume year columns are risk scores
                config.format = 'float';
                config.precision = 3;
            }
            return config;
        });

        const pivotedCountyRiskTableSection = createTableSection(
            null, // Title handled by countyRiskSectionTitle
            countyRiskData,
            countyRiskColumnConfig
        );
        resultsContainer.appendChild(pivotedCountyRiskTableSection);

        // Add the weighted average market risk row if data exists
        if (details.weighted_avg_market_risk_row && Object.keys(details.weighted_avg_market_risk_row).length > 0) {
            const tableElement = pivotedCountyRiskTableSection.querySelector('table.results-table');
            if (tableElement) {
                const tbody = tableElement.tBodies[0] || tableElement.createTBody(); // Get existing tbody or create if none
                const summaryRow = tbody.insertRow(); // Insert at the end of tbody
                summaryRow.className = 'summary-row'; // For styling

                // The first column in countyRiskColumnConfig is 'County'
                // The subsequent columns are years, matching keys in weighted_avg_market_risk_row
                countyRiskColumnConfig.forEach(colConfig => {
                    const cell = summaryRow.insertCell();
                    let value = details.weighted_avg_market_risk_row[colConfig.key]; // Key will be 'County' or a year string

                    if (value === null || typeof value === 'undefined') {
                        value = '-';
                    }
                    if (value !== '-') {
                        if (colConfig.key === 'County') {
                            // It's the label cell
                            cell.style.fontWeight = 'bold';
                        } else if (colConfig.format === 'float') { // Year columns for risk scores
                            value = parseFloat(value).toFixed(colConfig.precision || 3); // Use 3 for risk scores
                        }
                        // No specific formatting for integer or percentage here as this row is for risk scores
                    }
                    cell.textContent = value;
                    if (colConfig.key !== 'County') {
                        cell.style.textAlign = 'right';
                    }
                });
            }
        }

        // Add the weighted average market risk EXCLUDING current contract row if data exists
        if (details.weighted_avg_market_risk_excl_contract_row && Object.keys(details.weighted_avg_market_risk_excl_contract_row).length > 0) {
            const tableElement = pivotedCountyRiskTableSection.querySelector('table.results-table');
            if (tableElement) {
                const tbody = tableElement.tBodies[0] || tableElement.createTBody(); // Get existing tbody or create if none
                const summaryRowExcl = tbody.insertRow(); // Insert at the end of tbody
                summaryRowExcl.className = 'summary-row'; // For styling

                // The first column in countyRiskColumnConfig is 'County'
                // The subsequent columns are years, matching keys in weighted_avg_market_risk_excl_contract_row
                countyRiskColumnConfig.forEach(colConfig => {
                    const cell = summaryRowExcl.insertCell();
                    let value = details.weighted_avg_market_risk_excl_contract_row[colConfig.key]; // Key will be 'County' or a year string

                    if (value === null || typeof value === 'undefined') {
                        value = '-';
                    }
                    if (value !== '-') {
                        if (colConfig.key === 'County') {
                            // It's the label cell
                            cell.style.fontWeight = 'bold';
                        } else if (colConfig.format === 'float') { // Year columns for risk scores
                            value = parseFloat(value).toFixed(colConfig.precision || 3); // Use 3 for risk scores
                        }
                        // No specific formatting for integer or percentage here as this row is for risk scores
                    }
                    cell.textContent = value;
                    if (colConfig.key !== 'County') {
                        cell.style.textAlign = 'right';
                    }
                });
            }
        }

    } else {
        const pRisk = document.createElement('p');
        pRisk.textContent = 'No county-wide average risk score data available for the counties relevant to this plan.';
        pRisk.style.textAlign = 'center';
        resultsContainer.appendChild(pRisk);
    }
    
    const tamSectionTitleText = `Total Addressable Market (TAM) for Contract ${details.contract_id_cleaned}`;
    const tamSection = document.createElement('div');
    tamSection.className = 'data-section-simple'; 
    const tamTitle = document.createElement('h4'); // This could be an H3
    tamTitle.textContent = tamSectionTitleText;
    tamTitle.style.textAlign = 'center';
    tamTitle.style.marginTop = '30px';
    tamSection.appendChild(tamTitle);

    const tamValue = document.createElement('p');
    tamValue.textContent = details.total_addressable_market_overall ? details.total_addressable_market_overall.toLocaleString() : 'N/A';
    tamValue.style.fontSize = '1.2em';
    tamValue.style.fontWeight = 'bold';
    tamValue.style.textAlign = 'center';
    tamSection.appendChild(tamValue);
    resultsContainer.appendChild(tamSection);
}

// Helper function to create a table section (can be enhanced from parent_analyzer.js or made shared)
function createTableSection(title, dataRows, columnConfig) {
    const sectionDiv = document.createElement('div'); 
    
    const heading = document.createElement('h4'); 
    heading.textContent = title;
    sectionDiv.appendChild(heading);

    if (dataRows && dataRows.length > 0) {
        const table = document.createElement('table');
        table.className = 'results-table mini-table'; 

        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        columnConfig.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col.header;
            headerRow.appendChild(th);
        });

        const tbody = table.createTBody();
        dataRows.forEach(rowData => {
            const row = tbody.insertRow();
            columnConfig.forEach(col => {
                const cell = row.insertCell();
                let value = rowData[col.key];
                let currentFormat = col.format;
                let currentPrecision = col.precision;

                if (value === null || typeof value === 'undefined') {
                    value = '-';
                }

                if (value !== '-') {
                    if (rowData.Metric === "Specific Plan's Total Enrollment" && col.key !== 'Metric') {
                        currentFormat = 'integer';
                    } else if (rowData.Metric === "Total Addressable Market (Contract)" && col.key !== 'Metric') {
                        currentFormat = 'integer'; 
                    } else if (rowData.Metric === "Contract's % of TAM (Plan Footprint)" && col.key !== 'Metric') {
                        currentFormat = 'percentage';
                        currentPrecision = 2; // e.g., 12.34%
                    } else if (rowData.Metric === "YoY Growth" && col.key !== 'Metric') { // New condition for YoY Growth
                        currentFormat = 'percentage';
                        currentPrecision = 1; // e.g., 5.1%
                    } else if (col.key !== 'Metric' && (rowData.Metric === "Specific Plan's Avg. Risk Score" || rowData.Metric === "Weighted Avg. Market Risk (Plan Footprint)" || rowData.Metric === "Weighted Avg. Market Risk (Excl. Current Contract)")) {
                        currentFormat = 'float';
                        currentPrecision = 3;
                    }

                    if (col.header === 'Year') { // Should not happen for master table, but good fallback
                        value = parseInt(value);
                    } else if (currentFormat === 'percentage') {
                        value = parseFloat(value).toFixed(currentPrecision || 1) + '%';
                    } else if (currentFormat === 'float') {
                        value = parseFloat(value).toFixed(currentPrecision || 3); // Ensure 3 decimals for risk scores
                    } else if (currentFormat === 'integer') {
                        // Ensure it's an integer, then add thousand separators. Avoid .000 for whole numbers.
                        const intValue = parseInt(value);
                        if (!isNaN(intValue)) {
                            value = intValue.toLocaleString();
                        } else {
                            value = '-'; // Fallback if parsing fails
                        }
                    }
                }
                cell.textContent = value;
                if (value !== '-' && col.key !== 'Metric' && ['percentage', 'float', 'integer'].includes(currentFormat)) {
                    cell.style.textAlign = 'right';
                }
            });
        });
        sectionDiv.appendChild(table);
    } else {
        const p = document.createElement('p');
        p.textContent = 'No data available for this section.';
        sectionDiv.appendChild(p);
    }
    return sectionDiv;
} 