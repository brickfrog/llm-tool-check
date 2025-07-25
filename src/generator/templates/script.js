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
            
            // Update column visibility for the new active tab
            updateColumnVisibility();
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
                // Update column visibility for the new active nested tab
                updateColumnVisibility();
            }
        }
    });

    // Platform filtering functionality
    const platformFilterCheckboxes = document.querySelectorAll('.filter-checkbox input[type="checkbox"][id^="filter-"]');
    const hideFailureCheckbox = document.getElementById('hide-failure-columns');

    function applyFilters() {
        const activeFilters = new Set();
        platformFilterCheckboxes.forEach(checkbox => {
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
        const hideFailures = hideFailureCheckbox && hideFailureCheckbox.checked;

        // Get the currently active tab content to scope the filtering
        const activeTabContent = document.querySelector('.tab-content.active');
        if (!activeTabContent) return;

        // Get provider columns only within the active tab
        const activeProviderHeaders = activeTabContent.querySelectorAll('th[data-provider]');

        activeProviderHeaders.forEach(header => {
            const provider = header.getAttribute('data-provider');
            const table = header.closest('table');
            
            // Check visible rows only within this specific table
            const visibleRows = table.querySelectorAll('tbody tr[data-platform]:not([style*="display: none"])');
            let hasData = false;
            let hasSuccess = false;

            visibleRows.forEach(row => {
                const cell = row.querySelector(`td[data-provider="${provider}"]`);
                if (cell) {
                    const cellSpan = cell.querySelector('span.cell');
                    if (cellSpan) {
                        // Check if this cell has any actual data (not "none" class)
                        if (!cellSpan.classList.contains('none')) {
                            hasData = true;
                        }
                        // Check if it has successful data
                        if (cellSpan.classList.contains('success') || cellSpan.classList.contains('partial')) {
                            hasSuccess = true;
                        }
                    }
                }
            });

            // Determine whether to show the column
            let shouldShow = hasData;
            if (hideFailures) {
                shouldShow = hasSuccess;
            }

            // Hide/show the column header and all cells in this specific table
            const tableCellsInColumn = table.querySelectorAll(`th[data-provider="${provider}"], td[data-provider="${provider}"]`);
            tableCellsInColumn.forEach(cell => {
                if (shouldShow) {
                    cell.style.display = '';
                } else {
                    cell.style.display = 'none';
                }
            });
        });
    }

    // Add event listeners to platform filter checkboxes
    platformFilterCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', applyFilters);
    });

    // Add event listener to hide failure columns checkbox
    if (hideFailureCheckbox) {
        hideFailureCheckbox.addEventListener('change', function() {
            updateColumnVisibility();
        });
    }

    // Apply initial filters
    applyFilters();
});
