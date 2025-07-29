// Global store for filter options and selections
const filterState = {
    allParentOrgs: [],        // Array of {raw: "Raw Name", cleaned: "Cleaned Name"}
    allPlanTypes: [],         // Array of strings
    allSnpPlanTypes: [],      // Array of strings like "Dual-Eligible", "Chronic SNP", etc.
    
    selectedParentOrgsRaw: [], // Array of RAW parent org names selected by user
    selectedPlanTypes: [],     // Array of plan type strings selected
    selectedSnpPlanTypes: [],  // Array of SNP type strings selected

    availableSnpFlags: {}    // From API: {'plan_is_dual_eligible': true, ...}
};

// Default selections (cleaned names)
const defaultSelectedCleanedOrgs = [
    "UNITEDHEALTH GROUP", "HUMANA", "CVS HEALTH", "ELEVANCE HEALTH", "CENTENE"
];

let comparisonChartInstance = null; // To hold the Chart.js instance

document.addEventListener('DOMContentLoaded', () => {
    fetchFilterOptionsAndSetup();
    const applyFiltersButton = document.getElementById('apply-filters-button');
    if (applyFiltersButton) {
        applyFiltersButton.addEventListener('click', applyFilters);
    } else {
        console.warn("'apply-filters-button' not found on this page. Event listener not attached.");
    }

    const toggleButton = document.getElementById('toggle-sidebar-button');
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');

    if (toggleButton && sidebar && mainContent) {
        // Function to set initial state and button visibility
        function setInitialSidebarState() {
            if (window.innerWidth < 769) { // Small screens
                sidebar.classList.add('collapsed');
                toggleButton.innerHTML = '&#9776;'; // Hamburger
                toggleButton.title = "Show Filters";
                mainContent.classList.add('sidebar-is-collapsed');
            } else { // Larger screens
                sidebar.classList.remove('collapsed');
                toggleButton.innerHTML = '&times;'; // Times icon
                toggleButton.title = "Hide Filters";
                mainContent.classList.remove('sidebar-is-collapsed');
            }
            toggleButton.style.display = 'block'; // Always show the button now
        }

        setInitialSidebarState(); // Set state on load

        // Optional: Adjust on window resize
        window.addEventListener('resize', setInitialSidebarState);

        toggleButton.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            mainContent.classList.toggle('sidebar-is-collapsed');
            
            if (sidebar.classList.contains('collapsed')) {
                toggleButton.innerHTML = '&#9776;'; // Hamburger icon (Menu)
                toggleButton.title = "Show Filters";
            } else {
                toggleButton.innerHTML = '&times;'; // Times icon (Close)
                toggleButton.title = "Hide Filters";
            }
        });
    } else {
        console.warn("Sidebar toggle button, sidebar, or main-content element not found.");
    }
});

async function fetchFilterOptionsAndSetup() {
    try {
        const response = await fetch('/api/filter-options');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        if (data.errors && data.errors.length > 0) {
            console.error("Errors fetching filter options:", data.errors);
            const errorContainer = document.getElementById('filter-errors');
            if(errorContainer) errorContainer.textContent = "Error loading filter options: " + data.errors.join(', ');
            return;
        }

        filterState.allParentOrgs = data.parent_org_name_tuples.map(t => ({ raw: t[0], cleaned: t[1] }));
        filterState.allPlanTypes = data.unique_plan_types;
        filterState.availableSnpFlags = data.available_snp_flags;
        
        filterState.allSnpPlanTypes = [];
        if (data.available_snp_flags.plan_is_dual_eligible) filterState.allSnpPlanTypes.push("Dual-Eligible");
        if (data.available_snp_flags.plan_is_chronic_snp) filterState.allSnpPlanTypes.push("Chronic/Disabling Condition SNP");
        if (data.available_snp_flags.plan_is_institutional_snp) filterState.allSnpPlanTypes.push("Institutional SNP");
        filterState.allSnpPlanTypes.push("Traditional (Non-SNP)");

        const defaultRawOrgs = [];
        filterState.allParentOrgs.forEach(orgTuple => {
            if (defaultSelectedCleanedOrgs.includes(orgTuple.cleaned)) {
                if (!defaultRawOrgs.includes(orgTuple.raw)) {
                    defaultRawOrgs.push(orgTuple.raw);
                }
            }
        });
        filterState.selectedParentOrgsRaw = defaultRawOrgs; 

        let filtersWereActuallySetup = false; // Variable to track if any filter UI was initialized

        // Setup for Parent Organization Filter
        if (document.getElementById('parent-org-search')) {
            setupAdvancedFilter(
                'parent-org-search', 
                'parent-org-suggestions', 
                'parent-org-selected-tags', 
                filterState.allParentOrgs.map(org => org.cleaned), 
                filterState.selectedParentOrgsRaw, 
                true 
            );
            filtersWereActuallySetup = true;
        } else {
            // console.log("Parent org filter elements (e.g., 'parent-org-search') not found. Skipping setup.");
        }

        // Setup for Plan Type Filter
        if (document.getElementById('plan-type-search')) {
            setupAdvancedFilter(
                'plan-type-search', 
                'plan-type-suggestions', 
                'plan-type-selected-tags', 
                filterState.allPlanTypes,
                filterState.selectedPlanTypes
            );
            filtersWereActuallySetup = true;
        } else {
            // console.log("Plan type filter elements (e.g., 'plan-type-search') not found. Skipping setup.");
        }

        // Setup for SNP Type Filter
        if (document.getElementById('snp-type-search')) {
            setupAdvancedFilter(
                'snp-type-search', 
                'snp-type-suggestions', 
                'snp-type-selected-tags', 
                filterState.allSnpPlanTypes,
                filterState.selectedSnpPlanTypes
            );
            filtersWereActuallySetup = true;
        } else {
            // console.log("SNP type filter elements (e.g., 'snp-type-search') not found. Skipping setup.");
        }
        
        // Apply filters only if at least one filter group was actually set up
        if (filtersWereActuallySetup) {
            applyFilters(); 
        } else {
            console.log("No main filter components (e.g., parent-org-search, plan-type-search, snp-type-search) found on this page. Skipping initial applyFilters call.");
        }

    } catch (error) {
        console.error('Failed to fetch filter options:', error);
        const errorContainer = document.getElementById('filter-errors');
        if(errorContainer) errorContainer.textContent = "Failed to load filter options. Please try refreshing.";
    }
}

function setupAdvancedFilter(searchInputId, suggestionsContainerId, tagsContainerId, allOptions, selectedItemsArray, isParentOrgFilter = false) {
    const searchInput = document.getElementById(searchInputId);
    const suggestionsContainer = document.getElementById(suggestionsContainerId);
    const tagsContainer = document.getElementById(tagsContainerId);

    // If any of these critical elements don't exist, don't proceed with setup for this filter
    if (!searchInput || !suggestionsContainer || !tagsContainer) {
        console.warn(`Skipping setup for filter group: ${searchInputId}. One or more required elements not found.`);
        return;
    }

    function renderInitialTags() {
        tagsContainer.innerHTML = '';
        if (isParentOrgFilter) {
            const selectedCleanedForTags = [];
            filterState.allParentOrgs.forEach(orgTuple => {
                if (selectedItemsArray.includes(orgTuple.raw) && !selectedCleanedForTags.includes(orgTuple.cleaned)) {
                    selectedCleanedForTags.push(orgTuple.cleaned);
                }
            });
            selectedCleanedForTags.forEach(cleanedName => createTag(cleanedName));
        } else {
            selectedItemsArray.forEach(item => createTag(item));
        }
    }

    function createTag(itemText) {
        const tag = document.createElement('div');
        tag.classList.add('tag');
        tag.textContent = itemText;
        const removeBtn = document.createElement('span');
        removeBtn.classList.add('remove-tag');
        removeBtn.innerHTML = '&times;';
        removeBtn.onclick = () => {
            tag.remove();
            if (isParentOrgFilter) {
                const cleanedToRemove = itemText;
                const rawNamesToRemove = filterState.allParentOrgs
                    .filter(org => org.cleaned === cleanedToRemove)
                    .map(org => org.raw);
                
                rawNamesToRemove.forEach(rawName => {
                    const index = selectedItemsArray.indexOf(rawName);
                    if (index > -1) {
                        selectedItemsArray.splice(index, 1);
                    }
                });
            } else {
                const index = selectedItemsArray.indexOf(itemText);
                if (index > -1) {
                    selectedItemsArray.splice(index, 1);
                }
            }
        };
        tag.appendChild(removeBtn);
        tagsContainer.appendChild(tag);
    }

    searchInput.addEventListener('input', () => {
        const query = searchInput.value.toLowerCase();
        suggestionsContainer.innerHTML = '';
        if (query) {
            const filteredOptions = allOptions.filter(opt => opt.toLowerCase().includes(query));
            filteredOptions.slice(0, 10).forEach(opt => {
                const suggestionItem = document.createElement('div');
                suggestionItem.classList.add('suggestion-item');
                suggestionItem.textContent = opt;
                suggestionItem.onclick = () => {
                    if (isParentOrgFilter) {
                        const cleanedToAdd = opt;
                        const rawNamesToAdd = filterState.allParentOrgs
                            .filter(org => org.cleaned === cleanedToAdd)
                            .map(org => org.raw);

                        const existingTags = Array.from(tagsContainer.children).map(t => t.textContent.slice(0, -1).trim());
                        if(!existingTags.includes(cleanedToAdd)) {
                           createTag(cleanedToAdd);
                        }
                        rawNamesToAdd.forEach(rawName => {
                            if(!selectedItemsArray.includes(rawName)) {
                                selectedItemsArray.push(rawName);
                            }
                        });
                    } else {
                        if (!selectedItemsArray.includes(opt)) {
                            selectedItemsArray.push(opt);
                            createTag(opt);
                        }
                    }
                    searchInput.value = '';
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

    renderInitialTags();
}

async function applyFilters() {
    console.log("Applying filters...");
    console.log("Selected Parent Orgs (Raw):", filterState.selectedParentOrgsRaw);
    console.log("Selected Plan Types:", filterState.selectedPlanTypes);
    console.log("Selected SNP Types:", filterState.selectedSnpPlanTypes);

    // Check if the target elements for displaying data exist before trying to update them
    const parentOrgMetricsContainer = document.getElementById('parent-org-metrics-table-container');
    const industrySummaryContainer = document.getElementById('industry-summary-table-container');
    const comparisonChartContainer = document.getElementById('comparison-chart-container');

    if (parentOrgMetricsContainer) {
        parentOrgMetricsContainer.innerHTML = '<p class="loading-text">Loading Parent Organization Metrics...</p>';
    }
    if (industrySummaryContainer) {
        industrySummaryContainer.innerHTML = '<p class="loading-text">Loading Industry Summary...</p>';
    }
    if (comparisonChartContainer) {
        comparisonChartContainer.innerHTML = '<p class="loading-text">Loading Comparison Chart...</p>';
    }
    
    const errorDisplay = document.getElementById('api-error-display');
    if (errorDisplay) errorDisplay.textContent = '';

    const requestBody = {
        parent_organizations_raw: filterState.selectedParentOrgsRaw,
        plan_types: filterState.selectedPlanTypes,
        snp_types_ui: filterState.selectedSnpPlanTypes
    };

    try {
        const response = await fetch('/api/analysis-data', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Unknown error during analysis data fetch." }));
            throw new Error(`HTTP error ${response.status}: ${errorData.detail || "Failed to fetch analysis data"}`);
        }

        const results = await response.json();

        if (results.load_errors && results.load_errors.length > 0) {
            console.error("Errors from analysis API:", results.load_errors);
            if(errorDisplay) errorDisplay.textContent = "Error loading data: " + results.load_errors.join(', ');
        }

        renderParentOrgMetricsTable(results.parent_org_metrics_data, results.parent_org_metrics_columns);
        renderIndustrySummaryTable(results.industry_summary_data);
        renderComparisonChart(results.chart_data);

    } catch (error) {
        console.error('Error applying filters and fetching data:', error);
        if(errorDisplay) errorDisplay.textContent = `Failed to load analysis data: ${error.message}`;
        if (parentOrgMetricsContainer) {
            parentOrgMetricsContainer.innerHTML = '<p class="error-text">Error loading data.</p>';
        }
        if (industrySummaryContainer) {
            industrySummaryContainer.innerHTML = '<p class="error-text">Error loading data.</p>';
        }
        if (comparisonChartContainer) {
            comparisonChartContainer.innerHTML = '<p class="error-text">Error loading data.</p>';
        }
    }
}

function formatCellValue(value, metricName) {
    if (value === null || value === undefined || value === 'N/A' || Number.isNaN(value)) {
        return '-';
    }
    if (metricName) { // Check if metricName is provided
        if (metricName.includes('RAF YY')) {
            return parseFloat(value).toFixed(1) + '%';
        } else if (metricName.includes('Risk Score')) {
            return parseFloat(value).toFixed(3);
        } else if (metricName.includes('Enrollment')) {
            return parseInt(value).toLocaleString();
        }
    }
    // Fallback for generic numbers if metricName isn't specific enough or not provided
    if (typeof value === 'number') {
        if (value % 1 !== 0) { // Check if it's a float
            return parseFloat(value).toFixed(2);
        }
        return parseInt(value).toLocaleString();
    }
    return value;
}

function getUniqueParentOrgsFromColumns(columns) {
    const orgs = new Set();
    // Assuming columns are like 'ORG_Metric' or 'Year'
    // We want to extract 'ORG' part. Exclude 'Year'.
    columns.forEach(col => {
        if (col.toLowerCase() !== 'year') {
            const parts = col.split('_');
            if (parts.length > 1) { // Ensure it's an ORG_Metric format
                orgs.add(parts.slice(0, -1).join('_')); // Handle org names with underscores
            }
        }
    });
    return Array.from(orgs);
}

function renderParentOrgMetricsTable(data, columns) {
    const container = document.getElementById('parent-org-metrics-table-container');
    if (!container) return;

    if (!data || data.length === 0) {
        container.innerHTML = '<p class="loading-text">No data available for Parent Organization Metrics with the current filters.</p>';
        return;
    }

    const uniqueParentOrgs = getUniqueParentOrgsFromColumns(columns);
    // Define the desired metric order for sub-headers
    const metricsOrder = ["Risk Score", "RAF YY", "Enrollment"];

    let tableHtml = '<table class="results-table">';
    
    // Header Row 1: Year and Parent Organization Names (spanning 3 cols each)
    tableHtml += '<thead><tr>';
    tableHtml += '<th rowspan="2">Year</th>'; // Year spans two rows
    uniqueParentOrgs.forEach(orgName => {
        tableHtml += `<th colspan="3">${orgName}</th>`;
    });
    tableHtml += '</tr>';

    // Header Row 2: Metric Names (Risk Score, RAF YY, Enrollment) for each org
    tableHtml += '<tr>';
    uniqueParentOrgs.forEach(() => {
        metricsOrder.forEach(metric => {
            tableHtml += `<th>${metric}</th>`;
        });
    });
    tableHtml += '</tr></thead>';

    // Table Body
    tableHtml += '<tbody>';
    data.forEach(row => {
        tableHtml += '<tr>';
        tableHtml += `<td>${row['Year'] || '-'}</td>`;
        uniqueParentOrgs.forEach(orgName => {
            metricsOrder.forEach(metric => {
                const columnKey = `${orgName}_${metric}`;
                tableHtml += `<td>${formatCellValue(row[columnKey], metric)}</td>`;
            });
        });
        tableHtml += '</tr>';
    });
    tableHtml += '</tbody></table>';

    container.innerHTML = tableHtml;
}

function renderIndustrySummaryTable(data) {
    const container = document.getElementById('industry-summary-table-container');
    container.innerHTML = '';

    if (!data || data.length === 0) {
        container.innerHTML = '<p class="loading-text">No Industry Summary data available for the selected filters.</p>';
        return;
    }

    const table = document.createElement('table');
    table.classList.add('results-table');
    const thead = table.createTHead();
    const headerRow = thead.insertRow();
    const headers = ['Year', 'Industry Weighted Avg Risk Score', 'Industry YoY Risk Score Growth (%)', 'Industry Total Enrollment'];
    headers.forEach(text => {
        const th = document.createElement('th');
        th.textContent = text;
        headerRow.appendChild(th);
    });

    const tbody = table.createTBody();
    data.forEach(item => {
        const row = tbody.insertRow();
        row.insertCell().textContent = formatCellValue(item.year);
        row.insertCell().textContent = formatCellValue(item.industry_weighted_avg_risk_score, 'Risk Score');
        row.insertCell().textContent = formatCellValue(item['Industry YoY Risk Score Growth (%)'], 'RAF YY');
        row.insertCell().textContent = formatCellValue(item.industry_total_enrollment, 'Enrollment');
    });
    container.appendChild(table);
}

function renderComparisonChart(chartData) {
    const container = document.getElementById('comparison-chart-container');
    container.innerHTML = ''; 
    const canvas = document.createElement('canvas');
    canvas.id = 'comparisonLineChart';
    container.appendChild(canvas);

    const ctx = canvas.getContext('2d');

    if (!chartData || !chartData.labels || chartData.labels.length === 0 || !chartData.datasets ) {
        container.innerHTML = '<p>No data available for the comparison chart.</p>';
        if (comparisonChartInstance) {
            comparisonChartInstance.destroy();
            comparisonChartInstance = null;
        }
        return;
    }
     if (chartData.datasets.length === 0 && comparisonChartInstance){
        comparisonChartInstance.destroy();
        comparisonChartInstance = null;
        container.innerHTML = '<p>No data available for the comparison chart (no datasets).</p>';
        return;
    }
    
    const colors = [
        'rgba(54, 162, 235, 1)', 'rgba(255, 99, 132, 1)', 'rgba(75, 192, 192, 1)', 
        'rgba(255, 206, 86, 1)', 'rgba(153, 102, 255, 1)', 'rgba(255, 159, 64, 1)',
        'rgba(199, 199, 199, 1)', 'rgba(83, 102, 83, 1)' 
    ];
    chartData.datasets.forEach((dataset, index) => {
        if (dataset.label.toLowerCase().includes('industry')) {
            dataset.borderColor = 'rgba(0, 0, 0, 0.7)'; 
            dataset.borderDash = [5, 5]; 
        } else {
            dataset.borderColor = colors[index % colors.length];
        }
        dataset.data = dataset.data.map(val => val === null || typeof val === 'undefined' ? NaN : parseFloat(val.toFixed(3))); 
        dataset.tension = 0.1;
        dataset.fill = false;
    });

    if (comparisonChartInstance) {
        comparisonChartInstance.data = chartData;
        comparisonChartInstance.update();
    } else {
        comparisonChartInstance = new Chart(ctx, {
            type: 'line',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false, 
                        title: {
                            display: true,
                            text: 'Weighted Average Risk Score'
                        }
                    },
                    x: {
                         title: {
                            display: true,
                            text: 'Year'
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null && typeof context.parsed.y !== 'undefined') {
                                    label += context.parsed.y.toFixed(3);
                                } else {
                                    label += 'N/A';
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }
} 