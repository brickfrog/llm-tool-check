document.addEventListener('DOMContentLoaded', function() {
    const tabs = document.querySelectorAll('.tab');
    const tabContents = document.querySelectorAll('.tab-content');

    // Tab switching functionality
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Remove active class from all tabs and content
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab
            this.classList.add('active');

            // Show corresponding content
            const contentId = this.id.replace('tab', 'content');
            document.getElementById(contentId).classList.add('active');
        });
    });

    // Nested tab switching functionality using event delegation
    document.addEventListener('click', function(event) {
        if (event.target.classList.contains('nested-tab')) {
            const nestedTab = event.target;

            // Find parent tab content to scope nested tab switching
            const parentTabContent = nestedTab.closest('.tab-content');
            const parentNestedTabs = parentTabContent.querySelectorAll('.nested-tab');
            const parentNestedContents = parentTabContent.querySelectorAll('.nested-tab-content');

            // Remove active class from all nested tabs and content in this parent
            parentNestedTabs.forEach(t => t.classList.remove('active'));
            parentNestedContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked nested tab
            nestedTab.classList.add('active');

            // Show corresponding nested content
            const contentId = nestedTab.id.replace('nested-tab', 'content');
            const contentElement = document.getElementById(contentId);
            if (contentElement) {
                contentElement.classList.add('active');
            }
        }
    });

    // Platform filtering functionality
    const filterCheckboxes = document.querySelectorAll('.filter-checkbox input[type="checkbox"]');

    function applyFilters() {
        const activeFilters = new Set();
        filterCheckboxes.forEach(checkbox => {
            if (checkbox.checked) {
                const platform = checkbox.id.replace('filter-', '');
                activeFilters.add(platform);
            }
        });

        // Filter all table rows in both tool support and structured output tables
        const allRows = document.querySelectorAll('table tbody tr[data-platform]');
        allRows.forEach(row => {
            const platform = row.getAttribute('data-platform');
            // Special handling for iointel-library rows when iointel filter is active
            if (activeFilters.has(platform) ||
                (platform === 'iointel-library' && activeFilters.has('iointel'))) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });

        // Hide/show columns based on visible data
        updateColumnVisibility();
    }

    function updateColumnVisibility() {
        // Get all provider columns
        const allProviderHeaders = document.querySelectorAll('th[data-provider]');

        allProviderHeaders.forEach(header => {
            const provider = header.getAttribute('data-provider');
            const colIndex = header.getAttribute('data-col-index');

            // Check if any visible row has non-dash content for this provider
            const visibleRows = document.querySelectorAll('table tbody tr[data-platform]:not([style*="display: none"])');
            let hasData = false;

            visibleRows.forEach(row => {
                const cell = row.querySelector(`td[data-provider="${provider}"]`);
                if (cell) {
                    const cellText = cell.textContent.trim();
                    // Check if cell has meaningful data (not just "-")
                    if (cellText !== '-' && cellText !== '') {
                        hasData = true;
                    }
                }
            });

            // Hide/show the column header and all cells in this column
            const allCellsInColumn = document.querySelectorAll(`th[data-provider="${provider}"], td[data-provider="${provider}"]`);
            allCellsInColumn.forEach(cell => {
                if (hasData) {
                    cell.style.display = '';
                } else {
                    cell.style.display = 'none';
                }
            });
        });
    }

    // Add event listeners to filter checkboxes
    filterCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', applyFilters);
    });

    // Apply initial filters
    applyFilters();
});
