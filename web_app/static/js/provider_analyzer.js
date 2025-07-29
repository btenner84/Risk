document.addEventListener('DOMContentLoaded', () => {
    console.log("Provider Acquisition Analyzer Page Loaded");

    const providerSelect = document.getElementById('provider-acquisition-select');
    const fetchDataButton = document.getElementById('fetch-provider-data-button');
    
    const loadingIndicator = document.getElementById('provider-analyzer-loading');
    const errorDisplay = document.getElementById('provider-analyzer-error');
    const resultsSection = document.getElementById('provider-analyzer-results');
    const tableContainer = document.getElementById('provider-metrics-table-container');
    const resultsTitle = document.getElementById('provider-results-title');
    const chartContainer = document.getElementById('provider-dual-axis-chart-container');
    let providerChart = null;

    fetchDataButton.addEventListener('click', () => {
        const selectedAcquisition = providerSelect.value;
        if (!selectedAcquisition) {
            errorDisplay.textContent = "Please select a UNH acquisition.";
            errorDisplay.style.display = 'block';
            resultsSection.style.display = 'none';
            return;
        }
        fetchAndDisplayProviderData(selectedAcquisition);
    });

    // Auto-fetch data on page load with default selection
    fetchAndDisplayProviderData(providerSelect.value);

    async function fetchAndDisplayProviderData(acquisitionCode) {
        loadingIndicator.style.display = 'block';
        errorDisplay.style.display = 'none';
        resultsSection.style.display = 'none';
        tableContainer.innerHTML = '';
        chartContainer.style.display = 'none';
        if (providerChart) {
            providerChart.destroy();
            providerChart = null;
        }

        try {
            const response = await fetch(`/api/provider-analysis?acquisition_code=${encodeURIComponent(acquisitionCode)}`);
            if (!response.ok) {
                const errData = await response.json().catch(() => ({ detail: "Failed to fetch provider data." }));
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();

            if (data.load_errors && data.load_errors.length > 0) {
                errorDisplay.textContent = `API Error: ${data.load_errors.join(', ')}`;
                errorDisplay.style.display = 'block';
                loadingIndicator.style.display = 'none';
                return;
            }

            // Update title based on selection
            const acquisitionTitles = {
                'davita_ca_2019': 'DaVita Medical Group (Optum California)',
                'caremont_ny_2022': 'CareMount / Optum NY',
                'polyclinic_wa_2018': 'The Polyclinic (WA)',
                'reliant_ma_2018': 'Reliant Medical Group (MA)', 
                'oregon_medical_2022': 'Oregon Medical Group / Greenfield Health',
                'everett_clinic_wa_2019': 'The Everett Clinic (WA)',
                'sound_physicians_wa_2018': 'Sound Physicians (WA)',
                'all_captured': 'All Captured UNH Acquisitions'
            };
            resultsTitle.textContent = `Risk Score Analysis: ${acquisitionTitles[acquisitionCode] || 'Selected Acquisition'}`;

            if (!data.provider_metrics_data || data.provider_metrics_data.length === 0) {
                tableContainer.innerHTML = '<p>No provider data found for the specified acquisition group.</p>';
            } else {
                renderProviderMetricsTable(data.provider_metrics_data, data.provider_metrics_columns);
                
                // Render chart if data available
                if (data.chart_years && data.chart_years.length > 0) {
                    renderProviderDualAxisChart(
                        data.chart_years, 
                        data.chart_provider_count, 
                        data.chart_avg_risk_score
                    );
                    chartContainer.style.display = 'block';
                } else {
                    chartContainer.style.display = 'none';
                }
            }

            resultsSection.style.display = 'block';

        } catch (error) {
            console.error('Failed to fetch or display provider data:', error);
            errorDisplay.textContent = `Error: ${error.message}`;
            errorDisplay.style.display = 'block';
        } finally {
            loadingIndicator.style.display = 'none';
        }
    }

    function renderProviderMetricsTable(providerData, columnHeaders) {
        tableContainer.innerHTML = ''; // Clear previous table
        const table = document.createElement('table');
        table.classList.add('results-table');

        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        
        // Create headers from the provided column headers
        columnHeaders.forEach(header => {
            const th = document.createElement('th');
            th.textContent = header;
            headerRow.appendChild(th);
        });

        const tbody = table.createTBody();
        providerData.forEach(rowData => {
            const row = tbody.insertRow();
            columnHeaders.forEach(header => {
                const cell = row.insertCell();
                const value = rowData[header];
                
                // Format numeric values
                if (typeof value === 'number') {
                    if (header.toLowerCase().includes('risk') || header.toLowerCase().includes('score')) {
                        cell.textContent = value.toFixed(3);
                    } else {
                        cell.textContent = formatNumber(value);
                    }
                } else {
                    cell.textContent = value !== null && value !== undefined ? value : 'N/A';
                }
            });
        });

        tableContainer.appendChild(table);
    }

    function formatNumber(value, formatType = 'default') {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        
        if (formatType === 'percentage') {
            return `${(value * 100).toFixed(1)}%`;
        } else if (formatType === 'currency') {
            return `$${value.toLocaleString()}`;
        } else {
            return value.toLocaleString();
        }
    }

    function renderProviderDualAxisChart(years, providerCountData, avgRiskScoreData) {
        const ctx = document.getElementById('providerDualAxisChart').getContext('2d');
        
        if (providerChart) {
            providerChart.destroy();
        }

        providerChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: years,
                datasets: [
                    {
                        label: 'Average Risk Score',
                        data: avgRiskScoreData,
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'y',
                        tension: 0.1
                    },
                    {
                        label: 'Provider Count',
                        data: providerCountData,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.1)',
                        borderWidth: 2,
                        yAxisID: 'y1',
                        tension: 0.1
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
                plugins: {
                    title: {
                        display: true,
                        text: 'Provider Performance Over Time'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Year'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Average Risk Score'
                        },
                        beginAtZero: false
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Number of Providers'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                        beginAtZero: true
                    }
                }
            }
        });
    }
}); 