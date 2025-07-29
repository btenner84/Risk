document.addEventListener('DOMContentLoaded', () => {
    console.log("State Analyzer Page Loaded");

    // State analysis variables
    let stateData = null;
    let stateDualAxisChart = null;
    let stateMarketShareChart = null;
    
    // Pre-defined state list for auto-complete
    const US_STATES = [
        'Alabama', 'AL', 'Alaska', 'AK', 'Arizona', 'AZ', 'Arkansas', 'AR', 'California', 'CA',
        'Colorado', 'CO', 'Connecticut', 'CT', 'Delaware', 'DE', 'Florida', 'FL', 'Georgia', 'GA',
        'Hawaii', 'HI', 'Idaho', 'ID', 'Illinois', 'IL', 'Indiana', 'IN', 'Iowa', 'IA',
        'Kansas', 'KS', 'Kentucky', 'KY', 'Louisiana', 'LA', 'Maine', 'ME', 'Maryland', 'MD',
        'Massachusetts', 'MA', 'Michigan', 'MI', 'Minnesota', 'MN', 'Mississippi', 'MS',
        'Missouri', 'MO', 'Montana', 'MT', 'Nebraska', 'NE', 'Nevada', 'NV', 'New Hampshire', 'NH',
        'New Jersey', 'NJ', 'New Mexico', 'NM', 'New York', 'NY', 'North Carolina', 'NC',
        'North Dakota', 'ND', 'Ohio', 'OH', 'Oklahoma', 'OK', 'Oregon', 'OR', 'Pennsylvania', 'PA',
        'Rhode Island', 'RI', 'South Carolina', 'SC', 'South Dakota', 'SD', 'Tennessee', 'TN',
        'Texas', 'TX', 'Utah', 'UT', 'Vermont', 'VT', 'Virginia', 'VA', 'Washington', 'WA',
        'West Virginia', 'WV', 'Wisconsin', 'WI', 'Wyoming', 'WY'
    ];

    // Elements
    const stateNameInput = document.getElementById('state-name-input');
    const stateNameSuggestions = document.getElementById('state-name-suggestions');
    const fetchStateDataButton = document.getElementById('fetch-state-data-button');
    const planTypeFilterSelect = document.getElementById('plan-type-filter-select');
    const loadingSection = document.getElementById('state-analyzer-loading');
    const errorSection = document.getElementById('state-analyzer-error');
    const resultsSection = document.getElementById('state-analyzer-results');

    // Setup state autocomplete
    setupStateAutocomplete();

    // Event listeners
    if (fetchStateDataButton) {
        fetchStateDataButton.addEventListener('click', handleStateAnalysis);
    }

    // Auto-refresh when plan type filter changes (if we already have data)
    if (planTypeFilterSelect) {
        planTypeFilterSelect.addEventListener('change', () => {
            if (stateData && stateNameInput.value.trim()) {
                handleStateAnalysis(); // Re-fetch data with new filter
            }
        });
    }

    function setupStateAutocomplete() {
        if (!stateNameInput || !stateNameSuggestions) return;

        stateNameInput.addEventListener('input', () => {
            const query = stateNameInput.value.toLowerCase().trim();
            if (!query) {
                stateNameSuggestions.style.display = 'none';
                return;
            }

            const filteredStates = US_STATES.filter(state => 
                state.toLowerCase().includes(query)
            );

            renderStateSuggestions(filteredStates);
        });

        // Hide suggestions when clicking outside
        document.addEventListener('click', (event) => {
            if (!stateNameInput.contains(event.target) && !stateNameSuggestions.contains(event.target)) {
                stateNameSuggestions.style.display = 'none';
            }
        });
    }

    function renderStateSuggestions(suggestions) {
        stateNameSuggestions.innerHTML = '';
        if (suggestions.length === 0) {
            stateNameSuggestions.style.display = 'none';
            return;
        }

        const ul = document.createElement('ul');
        suggestions.forEach(stateName => {
            const li = document.createElement('li');
            li.textContent = stateName;
            li.addEventListener('click', () => {
                stateNameInput.value = stateName;
                stateNameSuggestions.style.display = 'none';
            });
            ul.appendChild(li);
        });
        
        stateNameSuggestions.appendChild(ul);
        stateNameSuggestions.style.display = 'block';
    }

    async function handleStateAnalysis() {
        const stateName = stateNameInput.value.trim();
        if (!stateName) {
            showError('Please enter a state name');
            return;
        }

        const planTypeFilter = document.getElementById('plan-type-filter-select')?.value || 'all';

        showLoading();
        hideError();
        hideResults();

        try {
            console.log(`Fetching state analysis data for: ${stateName}, plan_type: ${planTypeFilter}`);
            
            // Call the backend API with plan type filter
            const response = await fetch(`/api/state-analysis?state=${encodeURIComponent(stateName)}&plan_type_filter=${encodeURIComponent(planTypeFilter)}`);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            console.log('State analysis data received:', data);

            if (!data.state_metrics || data.state_metrics.length === 0) {
                throw new Error('No data found for the specified state and plan type');
            }

            stateData = data;
            renderStateAnalysis(data, planTypeFilter);
            
        } catch (error) {
            console.error('Error fetching state analysis:', error);
            showError(`Error loading state data: ${error.message}`);
        } finally {
            hideLoading();
        }
    }

    function renderStateAnalysis(data, planTypeFilter = 'all') {
        const stateName = data.state_name || stateNameInput.value;
        
        // Update results title with plan type filter info
        const resultsTitle = document.getElementById('state-results-title');
        if (resultsTitle) {
            const planTypeNames = {
                'all': 'All Plan Types',
                'traditional': 'Traditional (Non-SNP)',
                'dual_eligible': 'Dual Eligible SNP',
                'chronic': 'Chronic Condition SNP',
                'institutional': 'Institutional SNP'
            };
            const filterLabel = planTypeNames[planTypeFilter] || planTypeFilter;
            resultsTitle.textContent = `State Analysis: ${stateName} - ${filterLabel}`;
        }

        // 1. Render State Metrics Table
        renderStateMetricsTable(data.state_metrics);
        
        // 2. Render Dual-Axis Chart
        renderStateDualAxisChart(data.state_metrics);
        
        // 3. Render Market Share Chart
        renderStateMarketShareChart(data.parent_org_market_share, stateName);
        
        // 4. Render Parent Organization Enrollment Table
        renderParentOrgEnrollmentTable(data.parent_org_enrollment);
        
        // 5. Render Parent Organization Risk Score Table
        renderParentOrgRiskScoreTable(data.parent_org_risk_scores);

        showResults();
    }

    function renderStateMetricsTable(stateMetrics) {
        const container = document.getElementById('state-metrics-table-container');
        if (!container || !stateMetrics) return;

        // Process data - new simplified structure (no dual/traditional segmentation)
        const years = [...new Set(stateMetrics.map(row => row.year))].sort();
        
        const yearlyData = {};

        stateMetrics.forEach(row => {
            const year = row.year;
            yearlyData[year] = {
                enrollment: row.total_enrollment,
                risk_score: row.weighted_avg_risk_score
            };
        });

        // Clear container and add title
        container.innerHTML = '';
        
        const title = document.createElement('h4');
        title.className = 'text-center mb-3';
        title.textContent = 'State Market Metrics Over Time';
        container.appendChild(title);

        const table = document.createElement('table');
        table.className = 'results-table table table-striped table-bordered table-hover table-sm';

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        // Create header cells
        const metricHeader = document.createElement('th');
        metricHeader.scope = 'col';
        metricHeader.textContent = 'Metric';
        headerRow.appendChild(metricHeader);
        
        years.forEach(year => {
            const th = document.createElement('th');
            th.scope = 'col';
            th.style.textAlign = 'right';
            th.textContent = year;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        
        // State metrics (single category now)
        addMetricRowsToTable(tbody, 'State', yearlyData, years);

        table.appendChild(tbody);
        container.appendChild(table);
    }

    function addMetricRowsToTable(tbody, category, data, years) {
        // Helper function to format numbers like county analyzer
        function formatNumber(value, formatType, metricName = '') {
            if (value === null || typeof value === 'undefined') {
                return '-';
            }
            if (metricName.toLowerCase().includes('yoy growth')) {
                return Number(value).toFixed(1) + '%';
            }
            if (formatType === 'integer') {
                return Number(value).toLocaleString();
            }
            if (formatType === 'float3') {
                return Number(value).toFixed(3);
            }
            return value;
        }

        // Enrollment row
        const enrollmentRow = document.createElement('tr');
        const enrollmentMetricCell = document.createElement('td');
        enrollmentMetricCell.innerHTML = `<strong>${category} Enrollment</strong>`;
        enrollmentRow.appendChild(enrollmentMetricCell);
        
        years.forEach(year => {
            const cell = document.createElement('td');
            cell.style.textAlign = 'right';
            const value = data[year] ? data[year].enrollment : null;
            cell.textContent = formatNumber(value, 'integer');
            enrollmentRow.appendChild(cell);
        });
        tbody.appendChild(enrollmentRow);

        // Risk Score row
        const riskRow = document.createElement('tr');
        const riskMetricCell = document.createElement('td');
        riskMetricCell.innerHTML = `<strong>${category} Risk Score</strong>`;
        riskRow.appendChild(riskMetricCell);
        
        years.forEach(year => {
            const cell = document.createElement('td');
            cell.style.textAlign = 'right';
            const value = data[year] ? data[year].risk_score : null;
            cell.textContent = formatNumber(value, 'float3');
            riskRow.appendChild(cell);
        });
        tbody.appendChild(riskRow);

        // YoY Growth row
        const yoyRow = document.createElement('tr');
        const yoyMetricCell = document.createElement('td');
        yoyMetricCell.innerHTML = `<strong>YoY Growth (%)</strong>`;
        yoyRow.appendChild(yoyMetricCell);
        
        years.forEach((year, index) => {
            const cell = document.createElement('td');
            cell.style.textAlign = 'right';
            
            if (index === 0) {
                cell.textContent = '-';
            } else {
                const currentYear = data[year];
                const previousYear = data[years[index - 1]];
                if (currentYear && previousYear && previousYear.risk_score) {
                    const growth = ((currentYear.risk_score - previousYear.risk_score) / previousYear.risk_score) * 100;
                    cell.textContent = `${growth >= 0 ? '+' : ''}${growth.toFixed(1)}%`;
                } else {
                    cell.textContent = '-';
                }
            }
            yoyRow.appendChild(cell);
        });
        tbody.appendChild(yoyRow);
    }

    function renderStateDualAxisChart(stateMetrics) {
        const chartContainer = document.getElementById('state-dual-axis-chart-container');
        const canvas = document.getElementById('stateDualAxisChart');
        
        if (!chartContainer || !canvas || !stateMetrics) return;

        // Destroy existing chart
        if (stateDualAxisChart) {
            stateDualAxisChart.destroy();
            stateDualAxisChart = null;
        }

        // Prepare data - simplified since data is already aggregated
        const years = [...new Set(stateMetrics.map(row => row.year))].sort();
        const overallEnrollment = [];
        const overallRiskScore = [];

        years.forEach(year => {
            const yearData = stateMetrics.find(row => row.year === year);
            if (yearData) {
                overallEnrollment.push(yearData.total_enrollment);
                overallRiskScore.push(yearData.weighted_avg_risk_score);
            } else {
                overallEnrollment.push(0);
                overallRiskScore.push(0);
            }
        });

        const ctx = canvas.getContext('2d');
        stateDualAxisChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: years,
                datasets: [
                    {
                        label: 'Total Enrollment',
                        data: overallEnrollment,
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Weighted Avg Risk Score',
                        data: overallRiskScore,
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Enrollment'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Risk Score'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'State Enrollment vs Risk Score Trends'
                    }
                }
            }
        });

        chartContainer.style.display = 'block';
    }

    function renderStateMarketShareChart(marketShareData, stateName) {
        const chartContainer = document.getElementById('state-market-share-chart-container');
        const canvas = document.getElementById('stateMarketShareChart');
        const titleElement = document.getElementById('state-market-share-title');
        
        if (!chartContainer || !canvas || !marketShareData) return;

        // Update title
        if (titleElement) {
            titleElement.textContent = `Parent Organization Market Share in ${stateName}`;
        }

        // Destroy existing chart
        if (stateMarketShareChart) {
            stateMarketShareChart.destroy();
            stateMarketShareChart = null;
        }

        // Prepare data - get top 10 organizations by total enrollment
        const sortedOrgs = marketShareData
            .sort((a, b) => b.total_enrollment - a.total_enrollment)
            .slice(0, 10);

        const labels = sortedOrgs.map(org => org.parent_organization_name);
        const enrollmentData = sortedOrgs.map(org => org.total_enrollment);
        const percentageData = sortedOrgs.map(org => org.market_share_percentage);

        const colors = [
            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
            '#FF9F40', '#FF6384', '#C9CBCF', '#4BC0C0', '#36A2EB'
        ];

        const ctx = canvas.getContext('2d');
        stateMarketShareChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: enrollmentData,
                    backgroundColor: colors,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Market Share by Enrollment'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const orgName = context.label;
                                const enrollment = context.parsed.toLocaleString();
                                const percentage = percentageData[context.dataIndex].toFixed(1);
                                return `${orgName}: ${enrollment} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });

        chartContainer.style.display = 'block';
    }

    function renderParentOrgEnrollmentTable(enrollmentData) {
        const container = document.getElementById('state-parent-org-enrollment-table-container');
        if (!container || !enrollmentData) return;

        // Clear container
        container.innerHTML = '';

        // Group data by organization and year
        const orgData = {};
        enrollmentData.forEach(row => {
            if (!orgData[row.parent_organization_name]) {
                orgData[row.parent_organization_name] = {};
            }
            orgData[row.parent_organization_name][row.year] = row.total_enrollment;
        });

        const years = [...new Set(enrollmentData.map(row => row.year))].sort();
        const organizations = Object.keys(orgData).sort();

        // Create title
        const title = document.createElement('h4');
        title.className = 'text-center mb-3';
        title.textContent = 'Parent Organization Enrollment Over Time';
        container.appendChild(title);

        // Create table
        const table = document.createElement('table');
        table.className = 'results-table table table-striped table-bordered table-hover table-sm';

        // Create header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        const orgHeader = document.createElement('th');
        orgHeader.scope = 'col';
        orgHeader.textContent = 'Parent Organization';
        headerRow.appendChild(orgHeader);
        
        years.forEach(year => {
            const th = document.createElement('th');
            th.scope = 'col';
            th.style.textAlign = 'right';
            th.textContent = year;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create body
        const tbody = document.createElement('tbody');
        
        organizations.forEach(org => {
            const row = document.createElement('tr');
            
            const nameCell = document.createElement('td');
            nameCell.innerHTML = `<strong>${org}</strong>`;
            row.appendChild(nameCell);
            
            years.forEach(year => {
                const cell = document.createElement('td');
                cell.style.textAlign = 'right';
                const enrollment = orgData[org][year];
                cell.textContent = enrollment ? enrollment.toLocaleString() : '-';
                row.appendChild(cell);
            });
            
            tbody.appendChild(row);
        });

        table.appendChild(tbody);
        container.appendChild(table);
    }

    function renderParentOrgRiskScoreTable(riskScoreData) {
        const container = document.getElementById('state-parent-org-risk-score-table-container');
        if (!container || !riskScoreData) return;

        // Clear container
        container.innerHTML = '';

        // Group data by organization and year
        const orgData = {};
        riskScoreData.forEach(row => {
            if (!orgData[row.parent_organization_name]) {
                orgData[row.parent_organization_name] = {};
            }
            orgData[row.parent_organization_name][row.year] = row.weighted_avg_risk_score;
        });

        const years = [...new Set(riskScoreData.map(row => row.year))].sort();
        const organizations = Object.keys(orgData).sort();

        // Create title
        const title = document.createElement('h4');
        title.className = 'text-center mb-3';
        title.textContent = 'Parent Organization Risk Scores Over Time';
        container.appendChild(title);

        // Create table
        const table = document.createElement('table');
        table.className = 'results-table table table-striped table-bordered table-hover table-sm';

        // Create header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        
        const orgHeader = document.createElement('th');
        orgHeader.scope = 'col';
        orgHeader.textContent = 'Parent Organization';
        headerRow.appendChild(orgHeader);
        
        years.forEach(year => {
            const th = document.createElement('th');
            th.scope = 'col';
            th.style.textAlign = 'right';
            th.textContent = year;
            headerRow.appendChild(th);
        });
        
        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create body
        const tbody = document.createElement('tbody');
        
        organizations.forEach(org => {
            const row = document.createElement('tr');
            
            const nameCell = document.createElement('td');
            nameCell.innerHTML = `<strong>${org}</strong>`;
            row.appendChild(nameCell);
            
            years.forEach(year => {
                const cell = document.createElement('td');
                cell.style.textAlign = 'right';
                const riskScore = orgData[org][year];
                cell.textContent = riskScore ? riskScore.toFixed(3) : '-';
                row.appendChild(cell);
            });
            
            tbody.appendChild(row);
        });

        table.appendChild(tbody);
        container.appendChild(table);
    }

    function showLoading() {
        if (loadingSection) loadingSection.style.display = 'block';
    }

    function hideLoading() {
        if (loadingSection) loadingSection.style.display = 'none';
    }

    function showError(message) {
        if (errorSection) {
            errorSection.querySelector('p').textContent = message;
            errorSection.style.display = 'block';
        }
    }

    function hideError() {
        if (errorSection) errorSection.style.display = 'none';
    }

    function showResults() {
        if (resultsSection) resultsSection.style.display = 'block';
    }

    function hideResults() {
        if (resultsSection) resultsSection.style.display = 'none';
    }
}); 