document.addEventListener('DOMContentLoaded', () => {
    console.log("County Analyzer Page Loaded");

    const countyNameInput = document.getElementById('county-name-input');
    const fetchDataButton = document.getElementById('fetch-county-data-button');
    const suggestionsContainer = document.getElementById('county-name-suggestions');
    
    const loadingIndicator = document.getElementById('county-analyzer-loading');
    const errorDisplay = document.getElementById('county-analyzer-error');
    const resultsSection = document.getElementById('county-analyzer-results');
    const tableContainer = document.getElementById('county-metrics-table-container');
    const resultsTitle = document.getElementById('county-results-title');
    const chartContainer = document.getElementById('county-dual-axis-chart-container');
    let countyChart = null;

    fetchDataButton.addEventListener('click', () => {
        const countyName = countyNameInput.value.trim();
        if (!countyName) {
            errorDisplay.textContent = "Please enter a county name.";
            errorDisplay.style.display = 'block';
            resultsSection.style.display = 'none';
            return;
        }
        fetchAndDisplayCountyData(countyName);
    });

    countyNameInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault(); // Prevent form submission if it's in a form
            fetchDataButton.click();
        }
    });

    let suggestionTimeout;
    countyNameInput.addEventListener('input', () => {
        clearTimeout(suggestionTimeout); // Clear previous timeout
        const query = countyNameInput.value.trim();

        if (query.length < 2) { // Only fetch suggestions if query is at least 2 characters
            suggestionsContainer.innerHTML = '';
            suggestionsContainer.style.display = 'none';
            return;
        }

        // Debounce the API call
        suggestionTimeout = setTimeout(async () => {
            try {
                const response = await fetch(`/api/county-name-suggestions?query=${encodeURIComponent(query)}`);
                if (!response.ok) {
                    // Handle error silently for suggestions, or log to console
                    console.error('Failed to fetch county suggestions:', response.status);
                    suggestionsContainer.innerHTML = '';
                    suggestionsContainer.style.display = 'none';
                    return;
                }
                const suggestions = await response.json();
                renderSuggestions(suggestions);
            } catch (error) {
                console.error('Error fetching county suggestions:', error);
                suggestionsContainer.innerHTML = '';
                suggestionsContainer.style.display = 'none';
            }
        }, 300); // Adjust delay as needed (e.g., 300ms)
    });

    function renderSuggestions(suggestions) {
        suggestionsContainer.innerHTML = ''; // Clear previous suggestions
        if (suggestions.length === 0) {
            suggestionsContainer.style.display = 'none';
            return;
        }

        const ul = document.createElement('ul');
        ul.classList.add('suggestion-list'); // Add a class for styling if needed

        suggestions.forEach(suggestion => {
            const li = document.createElement('li');
            li.textContent = suggestion;
            li.addEventListener('click', () => {
                countyNameInput.value = suggestion; // Populate input with selected suggestion
                suggestionsContainer.innerHTML = ''; // Clear suggestions
                suggestionsContainer.style.display = 'none';
                fetchDataButton.click(); // Optionally, trigger analysis directly
            });
            ul.appendChild(li);
        });

        suggestionsContainer.appendChild(ul);
        suggestionsContainer.style.display = 'block';
    }

    document.addEventListener('click', (event) => {
        if (!countyNameInput.contains(event.target) && !suggestionsContainer.contains(event.target)) {
            suggestionsContainer.innerHTML = '';
            suggestionsContainer.style.display = 'none';
        }
    });

    async function fetchAndDisplayCountyData(countyName) {
        loadingIndicator.style.display = 'block';
        errorDisplay.style.display = 'none';
        resultsSection.style.display = 'none';
        tableContainer.innerHTML = '';
        chartContainer.style.display = 'none';
        if (countyChart) {
            countyChart.destroy();
            countyChart = null;
        }

        try {
            const response = await fetch(`/api/county-analysis?county_name=${encodeURIComponent(countyName)}`);
            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: "Failed to fetch county data." }));
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            if (data.load_errors && data.load_errors.length > 0) {
                errorDisplay.textContent = `API Error: ${data.load_errors.join(', ')}`;
                errorDisplay.style.display = 'block';
                loadingIndicator.style.display = 'none';
                return;
            }

            resultsTitle.textContent = `Metrics for ${data.county_name}`;
            // marketDataTableData = data.metrics_by_year || []; // Old variable name, and data structure changed

            if (!data.pivoted_metrics_data || data.pivoted_metrics_data.length === 0) {
                tableContainer.innerHTML = '<p>No metrics found for the specified county and criteria.</p>';
            } else {
                renderCountyMetricsTable(data.pivoted_metrics_data, data.pivoted_metrics_columns);
                if (data.chart_years && data.chart_years.length > 0) {
                    renderCountyDualAxisChart(data.chart_years, data.chart_overall_enrollment, data.chart_overall_risk_score);
                    chartContainer.style.display = 'block';
                } else {
                    chartContainer.style.display = 'none';
                }
            }

            // New: Render Parent Organization Enrollment Table
            if (data.county_parent_org_enrollment_data && data.county_parent_org_enrollment_data.length > 0) {
                renderCountyParentOrgEnrollmentTable(data.county_parent_org_enrollment_data, data.county_parent_org_enrollment_columns);
            } else if (!data.load_errors || data.load_errors.length === 0) {
                document.getElementById('county-parent-org-enrollment-table-container').innerHTML = '<p class="text-center text-muted">No parent organization enrollment data available for this county.</p>';
            }

            // New: Render Top Orgs Market Share Chart
            const topOrgsChartContainer = document.getElementById('county-top-orgs-market-share-chart-container');
            const canvasElement = document.getElementById('countyTopOrgsMarketShareChart');
            const titleElement = topOrgsChartContainer ? topOrgsChartContainer.querySelector('h4') : null;

            if (topOrgsChartContainer) {
                const existingNoDataMessage = topOrgsChartContainer.querySelector('p.text-center.text-muted');
                if (existingNoDataMessage) {
                    existingNoDataMessage.remove();
                }

                if (data.top_orgs_market_share_chart_years && data.top_orgs_market_share_chart_datasets && data.top_orgs_market_share_chart_datasets.length > 0) {
                    if (titleElement) titleElement.style.display = 'block';
                    if (canvasElement) canvasElement.style.display = 'block';
                    
                    renderTopOrgsMarketShareChart(data.top_orgs_market_share_chart_years, data.top_orgs_market_share_chart_datasets);
                    topOrgsChartContainer.style.display = 'block';
                } else {
                    if (titleElement) titleElement.style.display = 'none';
                    if (canvasElement) canvasElement.style.display = 'none';

                    if (typeof countyTopOrgsMarketShareChartInstance !== 'undefined' && countyTopOrgsMarketShareChartInstance) {
                        countyTopOrgsMarketShareChartInstance.destroy();
                        countyTopOrgsMarketShareChartInstance = null; 
                    }

                    const noDataP = document.createElement('p');
                    noDataP.className = 'text-center text-muted';
                    noDataP.textContent = 'Top organizations market share data not available for charting.';
                    topOrgsChartContainer.appendChild(noDataP);
                    topOrgsChartContainer.style.display = 'block';
                }
            }

            resultsSection.style.display = 'block';

        } catch (error) {
            console.error('Failed to fetch or display county data:', error);
            errorDisplay.textContent = `Error: ${error.message}`;
            errorDisplay.style.display = 'block';
        } finally {
            loadingIndicator.style.display = 'none';
        }
    }

    function renderCountyMetricsTable(metricsData, columnHeaders) {
        tableContainer.innerHTML = ''; // Clear previous table
        const table = document.createElement('table');
        table.classList.add('results-table');

        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        // const headers = [ // Old static headers
        //     'Year',
        //     'Overall Enrollment', 'Overall Risk Score',
        //     'Traditional Enrollment', 'Traditional Risk Score', 
        //     'Dual Eligible Enrollment', 'Dual Eligible Risk Score'
        // ];
        columnHeaders.forEach(text => {
            const th = document.createElement('th');
            th.textContent = text;
            headerRow.appendChild(th);
        });

        const tbody = table.createTBody();
        metricsData.forEach(metricRow => {
            const row = tbody.insertRow();
            columnHeaders.forEach(header => {
                const cellValue = metricRow[header];
                let formattedValue = '-'; // Default for undefined or null
                const metricNameForRow = metricRow["Metric"] || ""; // Get the metric name for this row

                if (cellValue !== null && typeof cellValue !== 'undefined') {
                    if (header === 'Metric') {
                        formattedValue = cellValue;
                    } else if (header.includes('Enrollment') || (typeof cellValue === 'number' && !header.includes('Score') && !metricNameForRow.toLowerCase().includes('yoy growth'))) { 
                        formattedValue = formatNumber(cellValue, 'integer', metricNameForRow);
                    } else if (header.includes('Score') || metricNameForRow.toLowerCase().includes('yoy growth')) { 
                        // YoY growth or Score columns are treated as float (YoY formatting handled by metricName in formatNumber)
                        formattedValue = formatNumber(cellValue, 'float3', metricNameForRow);
                    } else {
                        formattedValue = cellValue; // Fallback for unknown data types
                    }
                }
                row.insertCell().textContent = formattedValue;
            });
        });

        tableContainer.appendChild(table);
    }

    // Helper function for formatting numbers (can be expanded)
    function formatNumber(value, formatType, metricName = '') {
        if (value === null || typeof value === 'undefined') {
            return '-';
        }
        if (metricName.toLowerCase().includes('yoy growth (%)')) { // Check if it's a YoY row, match exact name
            return Number(value).toFixed(1) + '%';
        }
        if (formatType === 'integer') {
            return Number(value).toLocaleString();
        }
        if (formatType === 'float3') {
            return Number(value).toFixed(3);
        }
        return value; // Fallback
    }

    // --- New: Function to render Dual Axis Chart for County Analyzer ---
    function renderCountyDualAxisChart(years, enrollmentData, riskScoreData) {
        if (countyChart) {
            countyChart.destroy();
        }
        const ctx = document.getElementById('countyDualAxisChart').getContext('2d');
        countyChart = new Chart(ctx, {
            type: 'bar', // Default type, will have mixed types
            data: {
                labels: years,
                datasets: [
                    {
                        label: 'Overall Enrollment',
                        type: 'line',
                        data: enrollmentData,
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        fill: false,
                        yAxisID: 'y-axis-enrollment',
                        tension: 0.1
                    },
                    {
                        label: 'Overall Risk Score',
                        type: 'line',
                        data: riskScoreData,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)', // Optional fill for line
                        fill: false,
                        yAxisID: 'y-axis-risk-score',
                        tension: 0.1,
                        order: 1 // Ensure line is in front
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
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
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Overall Enrollment'
                        },
                        grid: {
                            drawOnChartArea: false, // Only draw grid for one axis or neither
                        },
                        ticks: {
                            callback: function(value) {
                                if (value >= 1000000) return (value / 1000000).toFixed(1) + 'M';
                                if (value >= 1000) return (value / 1000).toFixed(0) + 'K';
                                return value;
                            }
                        },
                        beginAtZero: true // Often good for enrollment data
                    },
                    'y-axis-risk-score': {
                        type: 'linear',
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Overall Risk Score'
                        },
                        ticks: {
                            // SuggestedMin and Max can be added if needed, or precision
                            // precision: 3 // Example if needed
                            callback: function(value) {
                                return Number(value).toFixed(3);
                            }
                        },
                        beginAtZero: false // Allow risk score axis to not start at zero if values are far from it
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    if (context.dataset.yAxisID === 'y-axis-risk-score') {
                                        label += Number(context.parsed.y).toFixed(3);
                                    } else {
                                        label += Number(context.parsed.y).toLocaleString();
                                    }
                                }
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }

    function renderCountyParentOrgEnrollmentTable(data, columns) {
        const container = document.getElementById('county-parent-org-enrollment-table-container');
        container.innerHTML = ''; // Clear previous results

        if (!data || data.length === 0 || !columns || columns.length === 0) {
            container.innerHTML = '<p class="text-center text-muted">No parent organization enrollment data to display for this county.</p>';
            return;
        }

        const title = document.createElement('h4');
        title.className = 'text-center mb-3';
        title.textContent = 'Parent Organization Enrollment in County by Year';
        container.appendChild(title);

        const table = document.createElement('table');
        table.className = 'results-table table table-striped table-bordered table-hover table-sm'; // Added table-sm for more compact table

        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        columns.forEach(colName => {
            const th = document.createElement('th');
            th.scope = 'col';
            th.textContent = colName;
            if (colName !== 'Parent Organization') { // Right align year columns
                th.style.textAlign = 'right';
            }
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);
        table.appendChild(thead);

        const tbody = document.createElement('tbody');
        data.forEach(rowData => {
            const tr = document.createElement('tr');
            columns.forEach(colName => {
                const td = document.createElement('td');
                let value = rowData[colName];
                if (value === null || value === undefined) {
                    td.textContent = '-';
                } else if (typeof value === 'number' && colName !== 'Parent Organization') {
                    td.textContent = value.toLocaleString(); // Format numbers with commas
                    td.style.textAlign = 'right';
                } else {
                    td.textContent = value;
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        container.appendChild(table);
    }

    let countyTopOrgsMarketShareChartInstance = null;
    function renderTopOrgsMarketShareChart(years, datasets) {
        if (countyTopOrgsMarketShareChartInstance) {
            countyTopOrgsMarketShareChartInstance.destroy();
        }
        const ctx = document.getElementById('countyTopOrgsMarketShareChart').getContext('2d');
        
        // Datasets already come with borderColor from backend
        // Ensure data points are numbers for Chart.js
        const processedDatasets = datasets.map(ds => ({
            ...ds,
            data: ds.data.map(d => (d === null || d === undefined) ? NaN : Number(d))
        }));

        countyTopOrgsMarketShareChartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: years,
                datasets: processedDatasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null && !Number.isNaN(context.parsed.y)) {
                                    label += context.parsed.y.toFixed(1) + '%'; // Market share as percentage
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Year'
                        }
                    },
                    y: { // Single Y-axis for market share
                        type: 'linear',
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Market Share (%)'
                        },
                        ticks: {
                            callback: function(value) {
                                return value + '%'; // Add % to tick values
                            },
                            // min: 0, // Optionally force min to 0
                            // max: 100 // Optionally force max to 100 if it makes sense
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }

}); 