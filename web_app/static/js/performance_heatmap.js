const API_BASE_URL = `${window.location.protocol}//${window.location.host}`;

let performanceData = [];
let currentOrgName = '';
let currentYear = 2023;
let currentMetric = 'risk_delta';
let usMapData = null;

document.addEventListener('DOMContentLoaded', () => {
    console.log("Performance Heat Map Page Loaded");

    const parentOrgSelect = document.getElementById('parent-org-select');
    const yearSelect = document.getElementById('year-select');
    const metricSelect = document.getElementById('metric-select');
    const countyLimitSelect = document.getElementById('county-limit-select');
    const minEnrollmentInput = document.getElementById('min-enrollment-input');
    const planTypeFilterSelect = document.getElementById('plan-type-filter-select');
    const generateButton = document.getElementById('generate-heatmap-button');
    const downloadButton = document.getElementById('download-performance-data');

    // Initialize controls
    loadParentOrganizations();
    loadUSMapData();

    generateButton.addEventListener('click', () => {
        const selectedOrg = parentOrgSelect.value;
        const selectedYear = parseInt(yearSelect.value);
        const selectedMetric = metricSelect.value;
        const countyLimit = countyLimitSelect.value;
        const minEnrollment = parseInt(minEnrollmentInput.value) || 0;
        const planTypeFilter = planTypeFilterSelect.value;

        if (!selectedOrg) {
            showError('Please select a parent organization.');
            return;
        }

        currentOrgName = selectedOrg;
        currentYear = selectedYear;
        currentMetric = selectedMetric;

        fetchAndDisplayPerformanceData(countyLimit, minEnrollment, planTypeFilter);
    });

    downloadButton.addEventListener('click', () => {
        downloadPerformanceData();
    });
});

async function loadParentOrganizations() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/filter-options`);
        const data = await response.json();
        
        const parentOrgSelect = document.getElementById('parent-org-select');
        parentOrgSelect.innerHTML = '<option value="">Select an organization...</option>';
        
        data.unique_cleaned_parent_org_names.forEach(org => {
            const option = document.createElement('option');
            option.value = org;
            option.textContent = org;
            parentOrgSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error loading parent organizations:', error);
        showError('Failed to load parent organizations.');
    }
}

async function loadUSMapData() {
    try {
        // Load US counties TopoJSON data
        const response = await fetch('https://cdn.jsdelivr.net/npm/us-atlas@3/counties-10m.json');
        usMapData = await response.json();
        console.log('US map data loaded successfully');
    } catch (error) {
        console.error('Error loading US map data:', error);
        showError('Failed to load map data. Map visualization may not work properly.');
    }
}

async function fetchAndDisplayPerformanceData(countyLimit, minEnrollment, planTypeFilter) {
    showLoading(true);
    hideError();
    hideResults();

    try {
        const response = await fetch(
            `${API_BASE_URL}/api/performance-heatmap?org_name=${encodeURIComponent(currentOrgName)}&year=${currentYear}&metric=${currentMetric}&county_limit=${countyLimit}&min_enrollment=${minEnrollment}&plan_type_filter=${planTypeFilter}`
        );
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ detail: "Failed to fetch performance data." }));
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.errors && data.errors.length > 0) {
            showError(`API Error: ${data.errors.join(', ')}`);
            return;
        }

        performanceData = data.county_performance || [];
        
        if (performanceData.length === 0) {
            showError('No performance data found for the selected organization and criteria.');
            return;
        }

        renderHeatMap();
        renderPerformanceSummary(data);
        renderPerformanceTable();
        
        showResults();
        document.getElementById('download-performance-data').style.display = 'inline-block';

    } catch (error) {
        console.error('Failed to fetch performance data:', error);
        showError(`Error: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function renderHeatMap() {
    if (!usMapData || !performanceData.length) {
        console.error('Missing map data or performance data');
        return;
    }

    console.log('Performance data:', performanceData);
    console.log('Sample performance data item:', performanceData[0]);

    const svg = d3.select("#us-map");
    svg.selectAll("*").remove(); // Clear previous map

    const width = 960;
    const height = 600;

    // Create projection and path
    const projection = d3.geoAlbersUsa()
        .scale(1300)
        .translate([width / 2, height / 2]);

    const path = d3.geoPath().projection(projection);

    // Create color scale based on current metric
    const colorScale = createColorScale();

    // Create tooltip
    const tooltip = d3.select("body").append("div")
        .attr("class", "map-tooltip")
        .style("opacity", 0)
        .style("position", "absolute")
        .style("background", "rgba(0, 0, 0, 0.8)")
        .style("color", "white")
        .style("padding", "10px")
        .style("border-radius", "5px")
        .style("pointer-events", "none");

    // Convert county performance data to lookup object
    const performanceLookup = {};
    performanceData.forEach(d => {
        const fipsCode = d.fips_state_county_code;
        if (fipsCode) {
            // Ensure FIPS code is a string and properly formatted (5 digits with leading zeros)
            let formattedFips;
            if (typeof fipsCode === 'number' || !isNaN(parseFloat(fipsCode))) {
                // Convert numeric FIPS (like 4013 or 4013.0) to 5-digit string
                formattedFips = Math.floor(parseFloat(fipsCode)).toString().padStart(5, '0');
            } else {
                // For string FIPS codes, clean and format
                formattedFips = String(fipsCode).replace(/[^0-9]/g, '').padStart(5, '0');
            }
            
            // Skip invalid FIPS codes (like "00nan")
            if (formattedFips === '00000' || formattedFips.includes('NaN') || formattedFips === '00nan') {
                console.log(`Skipping invalid FIPS: ${fipsCode} -> ${formattedFips}`);
                return;
            }
            
            performanceLookup[formattedFips] = d;
            console.log(`County: ${d.county_name}, FIPS: ${fipsCode} -> Formatted: ${formattedFips}, Risk Delta: ${d.risk_score_delta}`);
        }
    });

    console.log('Performance lookup object:', performanceLookup);
    console.log('Number of counties with FIPS codes:', Object.keys(performanceLookup).length);

    // Draw counties
    svg.append("g")
        .selectAll("path")
        .data(topojson.feature(usMapData, usMapData.objects.counties).features)
        .enter().append("path")
        .attr("d", path)
        .attr("class", "county")
        .style("fill", d => {
            const fipsCode = String(d.id).padStart(5, '0'); // Ensure TopoJSON FIPS is also formatted consistently
            const countyData = performanceLookup[fipsCode];
            if (countyData) {
                const metricValue = getMetricValue(countyData);
                if (metricValue !== null && !isNaN(metricValue)) {
                    const color = colorScale(metricValue);
                    console.log(`Coloring county FIPS ${fipsCode} (${countyData.county_name}) with value ${metricValue} -> color ${color}`);
                    return color;
                } else {
                    console.log(`County FIPS ${fipsCode} (${countyData.county_name}) has invalid metric value: ${metricValue}`);
                }
            } else {
                // Only log the first few missing counties to avoid spam
                if (Math.random() < 0.01) { // Only log 1% of missing counties
                    console.log(`No data found for county FIPS ${fipsCode}`);
                }
            }
            return "#f0f0f0"; // Default color for counties without data
        })
        .style("stroke", "#fff")
        .style("stroke-width", 0.5)
        .on("mouseover", function(event, d) {
            const fipsCode = String(d.id).padStart(5, '0');
            const countyData = performanceLookup[fipsCode];
            
            if (countyData) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(createTooltipContent(countyData))
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            }
        })
        .on("mouseout", function(d) {
            tooltip.transition()
                .duration(500)
                .style("opacity", 0);
        });

    // Draw state boundaries
    svg.append("path")
        .datum(topojson.mesh(usMapData, usMapData.objects.states, (a, b) => a !== b))
        .attr("class", "state-boundary")
        .attr("d", path)
        .style("fill", "none")
        .style("stroke", "#333")
        .style("stroke-width", 1);

    updateHeatMapTitle();
}

function createColorScale() {
    const values = performanceData.map(d => getMetricValue(d)).filter(v => v !== null && !isNaN(v));
    
    console.log('Values for color scale:', values);
    console.log('Number of valid values:', values.length);
    console.log('Sample metric values from data:', performanceData.slice(0, 3).map(d => ({
        county: d.county_name,
        risk_delta: d.risk_score_delta,
        market_share: d.market_share_pct,
        total_enrollment: d.total_enrollment,
        fips: d.fips_state_county_code
    })));
    
    if (values.length === 0) {
        console.log('No valid values found, using default color');
        return d3.scaleOrdinal().range(["#f0f0f0"]);
    }

    const extent = d3.extent(values);
    const colorScheme = getColorScheme();

    console.log('Value extent:', extent);
    console.log('Color scheme:', colorScheme);

    const scale = d3.scaleQuantile()
        .domain(values)
        .range(colorScheme);
        
    console.log('Color scale created, testing a few values:');
    values.slice(0, 5).forEach(v => {
        console.log(`Value ${v} maps to color ${scale(v)}`);
    });
    
    return scale;
}

function getColorScheme() {
    // Red to Blue color scheme (red = bad performance, blue = good performance)
    return ["#d73027", "#fc8d59", "#fee08b", "#e0f3f8", "#91bfdb", "#4575b4"];
}

function getMetricValue(countyData) {
    switch (currentMetric) {
        case 'risk_delta':
            return countyData.risk_score_delta;
        case 'market_share':
            return countyData.market_share_pct;
        case 'enrollment_growth':
            return countyData.enrollment_growth_pct;
        case 'total_enrollment':
            return countyData.total_enrollment;
        default:
            return null;
    }
}

function createTooltipContent(countyData) {
    const metricValue = getMetricValue(countyData);
    const metricLabel = getMetricLabel();
    
    return `
        <strong>${countyData.county_name}, ${countyData.state}</strong><br>
        ${metricLabel}: ${formatMetricValue(metricValue)}<br>
        Total Enrollment: ${(countyData.total_enrollment || 0).toLocaleString()}<br>
        Risk Score Delta: ${(countyData.risk_score_delta || 0).toFixed(3)}<br>
        Market Share: ${(countyData.market_share_pct || 0).toFixed(1)}%
    `;
}

function getMetricLabel() {
    switch (currentMetric) {
        case 'risk_delta':
            return 'Risk Score Delta';
        case 'market_share':
            return 'Market Share';
        case 'enrollment_growth':
            return 'Enrollment Growth';
        case 'total_enrollment':
            return 'Total Enrollment';
        default:
            return 'Value';
    }
}

function formatMetricValue(value) {
    if (value === null || value === undefined || isNaN(value)) {
        return 'N/A';
    }
    
    switch (currentMetric) {
        case 'risk_delta':
            return value.toFixed(3);
        case 'market_share':
        case 'enrollment_growth':
            return value.toFixed(1) + '%';
        case 'total_enrollment':
            return value.toLocaleString();
        default:
            return value.toString();
    }
}

function renderPerformanceSummary(data) {
    // Top performing counties
    const topCounties = [...performanceData]
        .sort((a, b) => (getMetricValue(b) || -Infinity) - (getMetricValue(a) || -Infinity))
        .slice(0, 5);
    
    renderCountyList('top-counties-list', topCounties, 'top');

    // Improvement opportunities (worst performing)
    const opportunityCounties = [...performanceData]
        .sort((a, b) => (getMetricValue(a) || Infinity) - (getMetricValue(b) || Infinity))
        .slice(0, 5);
    
    renderCountyList('opportunity-counties-list', opportunityCounties, 'opportunity');

    // Market gaps (high TAM, low market share)
    const marketGaps = [...performanceData]
        .filter(d => d.market_share_pct < 10 && d.total_addressable_market > 1000)
        .sort((a, b) => (b.total_addressable_market || 0) - (a.total_addressable_market || 0))
        .slice(0, 5);
    
    renderCountyList('market-gaps-list', marketGaps, 'gap');
}

function renderCountyList(containerId, counties, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    if (counties.length === 0) {
        container.innerHTML = '<p class="text-muted">No data available</p>';
        return;
    }

    const list = document.createElement('ul');
    list.className = 'county-performance-list';

    counties.forEach(county => {
        const li = document.createElement('li');
        li.className = 'county-item';
        
        const metricValue = getMetricValue(county);
        const formattedValue = formatMetricValue(metricValue);
        
        li.innerHTML = `
            <div class="county-name">${county.county_name}, ${county.state}</div>
            <div class="county-metric">${getMetricLabel()}: ${formattedValue}</div>
            <div class="county-enrollment">Enrollment: ${(county.total_enrollment || 0).toLocaleString()}</div>
        `;
        
        list.appendChild(li);
    });

    container.appendChild(list);
}

function renderPerformanceTable() {
    const tableContainer = document.getElementById('performance-table-container');
    
    // Clear existing content except the header
    const existingTable = tableContainer.querySelector('table');
    if (existingTable) {
        existingTable.remove();
    }

    if (performanceData.length === 0) {
        const noDataMessage = document.createElement('p');
        noDataMessage.textContent = 'No performance data to display.';
        noDataMessage.className = 'text-muted';
        tableContainer.appendChild(noDataMessage);
        return;
    }

    const table = document.createElement('table');
    table.className = 'results-table';

    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    
    const headers = [
        'County', 'State', 'Total Enrollment', 'Risk Score Delta',
        'Market Share %', 'YoY Growth %', 'Total Addressable Market'
    ];
    
    headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create body
    const tbody = document.createElement('tbody');
    
    performanceData.forEach(county => {
        const row = document.createElement('tr');
        
        const cells = [
            county.county_name || 'N/A',
            county.state || 'N/A',
            (county.total_enrollment || 0).toLocaleString(),
            (county.risk_score_delta || 0).toFixed(3),
            (county.market_share_pct || 0).toFixed(1) + '%',
            (county.enrollment_growth_pct || 0).toFixed(1) + '%',
            (county.total_addressable_market || 0).toLocaleString()
        ];
        
        cells.forEach((cellValue, index) => {
            const td = document.createElement('td');
            td.textContent = cellValue;
            if (index > 1) { // Right align numeric columns
                td.style.textAlign = 'right';
            }
            row.appendChild(td);
        });
        
        tbody.appendChild(row);
    });
    
    table.appendChild(tbody);
    tableContainer.appendChild(table);
}

function updateHeatMapTitle() {
    const title = document.getElementById('heatmap-title');
    const countyLimit = document.getElementById('county-limit-select').value;
    const minEnrollment = document.getElementById('min-enrollment-input').value;
    const planTypeFilter = document.getElementById('plan-type-filter-select').value;
    
    let titleText = `${currentOrgName} - ${getMetricLabel()} (${currentYear})`;
    
    // Add filter info to title
    let filterInfo = [];
    if (countyLimit !== "all") {
        filterInfo.push(`Top ${countyLimit} Counties`);
    }
    if (minEnrollment && parseInt(minEnrollment) > 0) {
        filterInfo.push(`Min ${minEnrollment} Enrollees`);
    }
    if (planTypeFilter && planTypeFilter !== "all") {
        const planTypeNames = {
            'traditional': 'Traditional Plans',
            'dual_eligible': 'Dual Eligible SNP',
            'chronic': 'Chronic Condition SNP',
            'institutional': 'Institutional SNP'
        };
        filterInfo.push(planTypeNames[planTypeFilter] || planTypeFilter);
    }
    
    if (filterInfo.length > 0) {
        titleText += ` (${filterInfo.join(', ')})`;
    }
    
    title.textContent = titleText;
}

function downloadPerformanceData() {
    if (performanceData.length === 0) {
        alert('No data to download');
        return;
    }

    const headers = [
        'county_name', 'state', 'fips_state_county_code', 'total_enrollment',
        'risk_score_delta', 'market_share_pct', 'enrollment_growth_pct', 'total_addressable_market'
    ];

    const csvContent = [
        headers.join(','),
        ...performanceData.map(row => 
            headers.map(header => {
                let value = row[header];
                if (value === null || value === undefined) {
                    value = '';
                } else if (typeof value === 'string' && value.includes(',')) {
                    value = `"${value.replace(/"/g, '""')}"`;
                }
                return value;
            }).join(',')
        )
    ].join('\n');

    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    
    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', `${currentOrgName}_Performance_${currentYear}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

function showLoading(show) {
    document.getElementById('heatmap-loading').style.display = show ? 'block' : 'none';
}

function showError(message) {
    const errorElement = document.getElementById('heatmap-error');
    errorElement.querySelector('p').textContent = message;
    errorElement.style.display = 'block';
}

function hideError() {
    document.getElementById('heatmap-error').style.display = 'none';
}

function showResults() {
    document.getElementById('heatmap-results').style.display = 'block';
}

function hideResults() {
    document.getElementById('heatmap-results').style.display = 'none';
} 