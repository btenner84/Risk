document.addEventListener('DOMContentLoaded', function() {
    const contractInput = document.getElementById('contractInput');
    const trackButton = document.getElementById('trackButton');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsSection = document.getElementById('resultsSection');
    const errorSection = document.getElementById('errorSection');
    const contractTitle = document.getElementById('contractTitle');
    const contractSummary = document.getElementById('contractSummary');
    const lineageTableBody = document.getElementById('lineageTableBody');

    // Handle track button click
    trackButton.addEventListener('click', function() {
        const contractId = contractInput.value.trim().toUpperCase();
        if (contractId) {
            trackContract(contractId);
        }
    });

    // Handle enter key in input
    contractInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            const contractId = contractInput.value.trim().toUpperCase();
            if (contractId) {
                trackContract(contractId);
            }
        }
    });

    async function trackContract(contractId) {
        console.log('Tracking contract:', contractId);
        
        // Show loading, hide others
        showLoading();
        hideResults();
        hideError();

        try {
            const response = await fetch(`/api/contract-tracker/${contractId}`);
            const data = await response.json();

            if (response.ok && data.success) {
                displayResults(contractId, data.lineage);
            } else {
                showError(data.message || 'Contract not found');
            }
        } catch (error) {
            console.error('Error tracking contract:', error);
            showError('Failed to load contract data');
        } finally {
            hideLoading();
        }
    }

    function displayResults(contractId, lineage) {
        console.log('Displaying results for:', contractId, lineage);
        
        // Set title
        contractTitle.textContent = `Lineage for Contract ${contractId}`;
        
        // Set summary
        const years = lineage.length;
        const uniqueContracts = new Set(lineage.map(l => l.contract_id)).size;
        contractSummary.innerHTML = `
            <div class="stat-item">
                <span class="stat-value">${years}</span>
                <span class="stat-label">Years Tracked</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${uniqueContracts}</span>
                <span class="stat-label">Unique Contract IDs</span>
            </div>
        `;

        // Clear table body
        lineageTableBody.innerHTML = '';

        // Sort lineage by year (most recent first)
        lineage.sort((a, b) => b.year - a.year);

        // Populate table
        lineage.forEach(item => {
            const row = document.createElement('tr');
            
            // Highlight if this is the current year or if contract changed
            if (item.year === 2023) {
                row.classList.add('current-year');
            }

            row.innerHTML = `
                <td class="year-cell">${item.year}</td>
                <td class="contract-cell">
                    <span class="contract-id">${item.contract_id}</span>
                </td>
                <td class="parent-cell">${item.parent_organization || 'Unknown'}</td>
                <td class="status-cell">
                    <span class="status-badge ${getStatusClass(item.status)}">${item.status}</span>
                </td>
            `;

            lineageTableBody.appendChild(row);
        });

        showResults();
    }

    function getStatusClass(status) {
        switch(status?.toLowerCase()) {
            case 'active': return 'status-active';
            case 'terminated': return 'status-terminated';
            case 'merged': return 'status-merged';
            case 'split': return 'status-split';
            case 'renamed': return 'status-renamed';
            default: return 'status-unknown';
        }
    }

    function showLoading() {
        loadingIndicator.style.display = 'block';
    }

    function hideLoading() {
        loadingIndicator.style.display = 'none';
    }

    function showResults() {
        resultsSection.style.display = 'block';
    }

    function hideResults() {
        resultsSection.style.display = 'none';
    }

    function showError(message) {
        document.getElementById('errorMessage').textContent = message;
        errorSection.style.display = 'block';
    }

    function hideError() {
        errorSection.style.display = 'none';
    }
}); 