const API_BASE_URL = `${window.location.protocol}//${window.location.host}`;
const AVAILABLE_YEARS = Array.from({length: 2023 - 2015 + 1}, (_, i) => 2015 + i).reverse(); // [2023, 2022, ..., 2015]

let marketDataTableData = []; // To store data for CSV export and sorting
let currentMarketSortState = {}; // { key: 'col_key', direction: 'asc'/'desc' }

document.addEventListener('DOMContentLoaded', () => {
    console.log("Market Analyzer Page Loaded");

    const yearSelector = document.getElementById('market-year-select');
    const fetchDataButton = document.getElementById('fetch-market-data-button');
    const outputExcelButton = document.getElementById('output-market-data-excel');

    // Populate year selector
    AVAILABLE_YEARS.forEach(year => {
        const option = document.createElement('option');
        option.value = year;
        option.textContent = year;
        if (year === 2023) { // Default to 2023
            option.selected = true;
        }
        yearSelector.appendChild(option);
    });

    fetchDataButton.addEventListener('click', () => {
        const selectedYear = parseInt(yearSelector.value);
        fetchAndDisplayMarketData(selectedYear);
    });

    outputExcelButton.addEventListener('click', () => {
        const selectedYear = parseInt(yearSelector.value);
        const fileName = `MarketAnalysis_Data_${selectedYear}.csv`;
        const columns = getMarketTableColumnConfig(selectedYear); // Get current columns
        exportMarketDataToCSV(marketDataTableData, columns, fileName);
    });

    // Initial data load for default year (2023)
    fetchAndDisplayMarketData(2023);
});

function getMarketTableColumnConfig(year) {
    return [
        { header: 'Contract ID', key: 'contract_id', sortable: true },
        { header: 'Plan ID', key: 'plan_id', sortable: true },
        { header: `Enrollment (${year})`, key: 'enrollment', format: 'integer', sortable: true },
        { header: `Plan Risk Score (${year})`, key: 'plan_actual_risk_score', format: 'float3', sortable: true },
        { header: `County Wtd. Risk Score (${year})`, key: 'county_weighted_risk_score', format: 'float3', sortable: true },
        { header: `Delta vs County Wtd. (${year})`, key: 'delta_risk_score', format: 'float3', sortable: true }
    ];
}

async function fetchAndDisplayMarketData(year) {
    const loadingIndicator = document.getElementById('market-analyzer-loading');
    const errorDisplay = document.getElementById('market-analyzer-error');
    const resultsContainer = document.getElementById('market-analyzer-results');
    const tableContainer = document.getElementById('market-table-container');
    const tableTitle = document.getElementById('market-table-title');
    const outputExcelButton = document.getElementById('output-market-data-excel');

    loadingIndicator.style.display = 'block';
    errorDisplay.style.display = 'none';
    tableContainer.innerHTML = ''; // Clear previous table
    outputExcelButton.style.display = 'none'; // Hide button until data is loaded
    marketDataTableData = []; // Clear previous data for export

    tableTitle.textContent = `Market Data for Year ${year}`;

    try {
        const response = await fetch(`${API_BASE_URL}/api/market-analysis?year=${year}`);
        if (!response.ok) {
            const errData = await response.json().catch(() => ({ detail: "Failed to fetch market data."}));
            throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
        }
        const data = await response.json();

        if (data.errors && data.errors.length > 0) {
            errorDisplay.textContent = `API Error: ${data.errors.join(', ')}`;
            errorDisplay.style.display = 'block';
            loadingIndicator.style.display = 'none';
            return;
        }

        marketDataTableData = data.market_data || [];

        if (marketDataTableData.length === 0) {
            tableContainer.innerHTML = '<p>No market data found for the selected year.</p>';
        } else {
            const columnConfig = getMarketTableColumnConfig(year);
            const table = createSortableTable(marketDataTableData, columnConfig, 'market-analysis-table', currentMarketSortState, (newState) => {
                currentMarketSortState = newState;
                // No need to re-render for simple client-side sort, createSortableTable handles it internally
            });
            tableContainer.appendChild(table);
            outputExcelButton.style.display = 'inline-block'; // Show button
        }

    } catch (error) {
        console.error('Failed to fetch or display market data:', error);
        errorDisplay.textContent = `Error: ${error.message}`;
        errorDisplay.style.display = 'block';
    } finally {
        loadingIndicator.style.display = 'none';
    }
}


function createSortableTable(dataRows, columnConfig, tableId, currentSort, onSortChangeCallback) {
    let currentData = [...dataRows]; // Make a mutable copy for sorting

    const table = document.createElement('table');
    table.className = 'results-table';
    table.id = tableId;
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');
    const headerRow = document.createElement('tr');

    columnConfig.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.header;
        if (col.sortable) {
            th.classList.add('sortable');
            th.dataset.sortKey = col.key;
            if (currentSort && currentSort.key === col.key) {
                th.classList.add(currentSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
            }
            th.addEventListener('click', () => {
                const newDirection = (currentSort && currentSort.key === col.key && currentSort.direction === 'asc') ? 'desc' : 'asc';
                onSortChangeCallback({ key: col.key, direction: newDirection });
                sortAndRedrawTable(col.key, newDirection);
                // Update header styles
                headerRow.querySelectorAll('th.sortable').forEach(h => h.classList.remove('sort-asc', 'sort-desc'));
                th.classList.add(newDirection === 'asc' ? 'sort-asc' : 'sort-desc');
            });
        }
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Find the column configuration for the sort key to determine its type
    function getColumnFormat(sortKey) {
        const colConfig = columnConfig.find(c => c.key === sortKey);
        return colConfig ? colConfig.format : null;
    }

    function renderTableBody() {
        tbody.innerHTML = ''; // Clear existing rows
        currentData.forEach(rowData => {
            const tr = document.createElement('tr');
            columnConfig.forEach(col => {
                const td = document.createElement('td');
                let value = rowData[col.key];
                let currentFormat = col.format;
                let currentPrecision = undefined;

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
                        td.textContent = parseFloat(value).toFixed(currentPrecision === undefined ? 3 : currentPrecision);
                    } else {
                        td.textContent = value;
                    }
                } else {
                    td.textContent = value;
                }
                td.style.textAlign = (col.format === 'integer' || col.format?.startsWith('float') || col.format?.startsWith('percentage')) ? 'right' : 'left';
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    function sortAndRedrawTable(key, direction) {
        const columnFormat = getColumnFormat(key);
        const isNumericSort = columnFormat === 'integer' || (typeof columnFormat === 'string' && columnFormat.startsWith('float'));

        currentData.sort((a, b) => {
            let valA = a[key];
            let valB = b[key];

            const aIsNull = valA === null || typeof valA === 'undefined' || valA === '-' || String(valA).trim() === '';
            const bIsNull = valB === null || typeof valB === 'undefined' || valB === '-' || String(valB).trim() === '';

            if (aIsNull && bIsNull) return 0;
            if (aIsNull) return direction === 'asc' ? -1 : 1; // Nulls/empty first on asc, last on desc
            if (bIsNull) return direction === 'asc' ? 1 : -1; // Nulls/empty first on asc, last on desc

            if (isNumericSort) {
                valA = parseFloat(String(valA).replace(/,/g, '')); // Remove commas for parsing
                valB = parseFloat(String(valB).replace(/,/g, ''));

                const aIsNaN = isNaN(valA);
                const bIsNaN = isNaN(valB);

                if (aIsNaN && bIsNaN) return 0;
                if (aIsNaN) return direction === 'asc' ? 1 : -1; // NaN treated as larger
                if (bIsNaN) return direction === 'asc' ? -1 : 1; // NaN treated as larger

            } else { // String sort (already handled nulls)
                valA = String(valA).toLowerCase();
                valB = String(valB).toLowerCase();
            }

            if (valA < valB) return direction === 'asc' ? -1 : 1;
            if (valA > valB) return direction === 'asc' ? 1 : -1;
            return 0;
        });
        renderTableBody();
    }

    renderTableBody(); // Initial render
    table.appendChild(tbody);
    return table;
}

function exportMarketDataToCSV(data, columns, fileName) {
    if (!data || data.length === 0) {
        console.warn("No data provided to exportMarketDataToCSV");
        alert("No data to export.");
        return;
    }
    const headers = columns.map(col => col.header).join(',');
    const rows = data.map(rowData => {
        return columns.map(col => {
            let value = rowData[col.key];
            if (value === null || typeof value === 'undefined') {
                value = '';
            } else {
                value = String(value);
                if (value.includes(',')) {
                    value = `"${value.replace(/"/g, '""')}"`; // CSV quote and escape double quotes
                }
            }
            return value;
        }).join(',');
    });

    const csvContent = "\uFEFF" + [headers, ...rows].join('\n'); // Add BOM for Excel compatibility
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");

    if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", fileName);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    } else {
        navigator.msSaveBlob(blob, fileName);
    }
} 