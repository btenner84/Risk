class LineageAnalyzer {
    constructor() {
        this.currentChart = null;
        this.currentLineage = null;
        this.searchTimeout = null;
        
        this.initializeEventListeners();
        this.loadInitialData();
    }
    
    initializeEventListeners() {
        const searchInput = document.getElementById('lineageSearch');
        const searchResults = document.getElementById('searchResults');
        
        // Search functionality
        searchInput.addEventListener('input', (e) => {
            clearTimeout(this.searchTimeout);
            this.searchTimeout = setTimeout(() => {
                this.performSearch(e.target.value);
            }, 300);
        });
        
        // Hide search results when clicking outside
        document.addEventListener('click', (e) => {
            if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
                searchResults.style.display = 'none';
            }
        });
        
        // Show search results when focusing on input
        searchInput.addEventListener('focus', () => {
            if (searchResults.children.length > 0) {
                searchResults.style.display = 'block';
            }
        });
    }
    
    async loadInitialData() {
        // Load top accelerators by default
        await this.performSearch('');
    }
    
    async performSearch(query) {
        try {
            const response = await fetch(`/api/lineage/search?query=${encodeURIComponent(query)}`);
            const data = await response.json();
            
            this.displaySearchResults(data.results || []);
        } catch (error) {
            console.error('Search error:', error);
            this.showError('Error performing search: ' + error.message);
        }
    }
    
    displaySearchResults(results) {
        const searchResults = document.getElementById('searchResults');
        
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
            searchResults.style.display = 'block';
            return;
        }
        
        searchResults.innerHTML = '';
        
        results.forEach(result => {
            const item = document.createElement('div');
            item.className = 'search-result-item';
            item.onclick = () => this.selectLineage(result.lineage_id);
            
            const accelerationClass = this.getAccelerationClass(result.acceleration_pct_per_year);
            
            item.innerHTML = `
                <div class="result-org">${result.parent_organization}</div>
                <div class="result-metrics">
                    ${result.first_year}-${result.last_year} • ${result.years_tracked} years • ${result.latest_enrollment.toLocaleString()} enrollees
                    <span class="result-acceleration ${accelerationClass}">
                        ${result.acceleration_pct_per_year > 0 ? '+' : ''}${result.acceleration_pct_per_year}%/yr
                    </span>
                </div>
            `;
            
            searchResults.appendChild(item);
        });
        
        searchResults.style.display = 'block';
    }
    
    getAccelerationClass(acceleration) {
        if (acceleration >= 5) return 'acceleration-high';
        if (acceleration >= 2) return 'acceleration-medium';
        if (acceleration >= -2) return 'acceleration-low';
        return 'acceleration-negative';
    }
    
    async selectLineage(lineageId) {
        try {
            // Hide search results
            document.getElementById('searchResults').style.display = 'none';
            
            // Show loading state
            this.showLoadingState();
            
            // Fetch detailed lineage data
            const response = await fetch(`/api/lineage/lineage/${lineageId}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            this.currentLineage = data;
            
            // Display the lineage details
            this.displayLineageDetails(data);
            this.createLineageChart(data);
            this.displayYearlyData(data.yearly_data);
            this.showCalculationExplanation();
            
        } catch (error) {
            console.error('Error loading lineage:', error);
            this.showError('Error loading lineage details: ' + error.message);
        }
    }
    
    showLoadingState() {
        document.getElementById('lineageDetails').style.display = 'grid';
        document.getElementById('lineageTitle').textContent = 'Loading...';
        document.getElementById('lineageSubtitle').textContent = 'Fetching lineage details...';
    }
    
    displayLineageDetails(data) {
        const summary = data.summary;
        
        // Update title and subtitle
        document.getElementById('lineageTitle').textContent = summary.parent_organization;
        document.getElementById('lineageSubtitle').textContent = 
            `${summary.plan_type} • ${summary.first_year}-${summary.last_year} • ID: ${summary.lineage_id}`;
        
        // Update acceleration badge
        const accelerationBadge = document.getElementById('accelerationBadge');
        const acceleration = summary.acceleration_vs_competitors_pct_per_year;
        accelerationBadge.textContent = `${acceleration > 0 ? '+' : ''}${acceleration.toFixed(2)}%/year`;
        accelerationBadge.className = `acceleration-badge ${this.getAccelerationClass(acceleration)}`;
        
        // Update metrics
        document.getElementById('yearsTracked').textContent = summary.years_tracked;
        document.getElementById('latestEnrollment').textContent = summary.latest_enrollment.toLocaleString();
        document.getElementById('rSquared').textContent = summary.r_squared_competitors.toFixed(3);
        
        const performance = summary.current_performance_vs_competitors_pct;
        const performanceElement = document.getElementById('currentPerformance');
        performanceElement.textContent = `${performance > 0 ? '+' : ''}${performance.toFixed(1)}%`;
        performanceElement.className = `metric-value ${performance > 0 ? 'performance-positive' : 'performance-negative'}`;
        
        // Show the details
        document.getElementById('lineageDetails').style.display = 'grid';
    }
    
    createLineageChart(data) {
        const ctx = document.getElementById('lineageChart').getContext('2d');
        
        // Destroy existing chart if any
        if (this.currentChart) {
            this.currentChart.destroy();
        }
        
        const graphData = data.graph_data;
        const trendData = data.trend_data;
        
        // Prepare datasets
        const datasets = [
            {
                label: 'Actual Risk Score',
                data: graphData.years.map((year, i) => ({
                    x: year,
                    y: graphData.actual_risk_scores[i]
                })).filter(d => d.y !== null),
                borderColor: '#e74c3c',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
                tension: 0.2
            },
            {
                label: 'Competitor Benchmark',
                data: graphData.years.map((year, i) => ({
                    x: year,
                    y: graphData.benchmark_scores[i]
                })).filter(d => d.y !== null),
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 3,
                pointRadius: 6,
                pointHoverRadius: 8,
                tension: 0.2
            }
        ];
        
        // Add trend lines if available
        if (trendData.actual_trend) {
            datasets.push({
                label: 'Actual Trend',
                data: graphData.years.slice(0, trendData.actual_trend.values.length).map((year, i) => ({
                    x: year,
                    y: trendData.actual_trend.values[i]
                })),
                borderColor: '#c0392b',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                tension: 0
            });
        }
        
        if (trendData.benchmark_trend) {
            datasets.push({
                label: 'Benchmark Trend',
                data: graphData.years.slice(0, trendData.benchmark_trend.values.length).map((year, i) => ({
                    x: year,
                    y: trendData.benchmark_trend.values[i]
                })),
                borderColor: '#2980b9',
                borderWidth: 2,
                borderDash: [5, 5],
                pointRadius: 0,
                tension: 0
            });
        }
        
        this.currentChart = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Risk Score Acceleration Analysis',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        position: 'top'
                    },
                    tooltip: {
                        callbacks: {
                            afterBody: (context) => {
                                const dataIndex = context[0].dataIndex;
                                const enrollment = graphData.enrollment_sizes[dataIndex];
                                const performance = graphData.performance_vs_competitors[dataIndex];
                                
                                let extra = [];
                                if (enrollment) extra.push(`Enrollment: ${enrollment.toLocaleString()}`);
                                if (performance !== null) extra.push(`vs Competitors: ${performance > 0 ? '+' : ''}${performance.toFixed(2)}%`);
                                
                                return extra;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: 'Year'
                        },
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return Math.floor(value).toString();
                            }
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Risk Score'
                        },
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(3);
                            }
                        }
                    }
                }
            }
        });
    }
    
    displayYearlyData(yearlyData) {
        const tableBody = document.querySelector('#yearlyDataTable tbody');
        tableBody.innerHTML = '';
        
        yearlyData.forEach(year => {
            const row = document.createElement('tr');
            
            // Format performance with color coding
            const performance = year.performance_vs_market_excluding_pct;
            const performanceClass = performance > 0 ? 'performance-positive' : 'performance-negative';
            const performanceText = performance !== null ? 
                `${performance > 0 ? '+' : ''}${performance.toFixed(2)}%` : 'N/A';
            
            // Format counties list (truncate if too long)
            let countiesText = year.counties_in_footprint || 'N/A';
            if (countiesText.length > 50) {
                countiesText = countiesText.substring(0, 50) + '...';
            }
            
            row.innerHTML = `
                <td><strong>${year.year}</strong></td>
                <td><span class="plan-evolution">${year.contract_number}_${year.plan_id}</span></td>
                <td>${year.total_enrollment.toLocaleString()}</td>
                <td>${year.actual_risk_score !== null ? year.actual_risk_score.toFixed(4) : 'N/A'}</td>
                <td>${year.county_weighted_benchmark_excluding !== null ? year.county_weighted_benchmark_excluding.toFixed(4) : 'N/A'}</td>
                <td class="performance-cell ${performanceClass}">${performanceText}</td>
                <td title="${year.counties_in_footprint}">${year.num_counties} counties</td>
                <td>${year.total_footprint_enrollment.toLocaleString()}</td>
            `;
            
            tableBody.appendChild(row);
        });
        
        document.getElementById('yearlyDataSection').style.display = 'block';
    }
    
    showCalculationExplanation() {
        document.getElementById('calculationExplanation').style.display = 'block';
    }
    
    showError(message) {
        const container = document.querySelector('.lineage-container');
        const existingError = container.querySelector('.error');
        if (existingError) {
            existingError.remove();
        }
        
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        
        container.insertBefore(errorDiv, container.firstChild);
        
        // Auto-hide after 10 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.remove();
            }
        }, 10000);
    }
    
    // Utility function to format numbers
    formatNumber(num, decimals = 0) {
        if (num === null || num === undefined) return 'N/A';
        return num.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }
    
    // Utility function to format percentages
    formatPercentage(num, decimals = 2) {
        if (num === null || num === undefined) return 'N/A';
        const sign = num > 0 ? '+' : '';
        return `${sign}${num.toFixed(decimals)}%`;
    }
}

// Initialize the analyzer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.lineageAnalyzer = new LineageAnalyzer();
});

// Export for potential external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LineageAnalyzer;
} 