document.addEventListener("DOMContentLoaded", function() {
    // Get reference to the content display area
    const contentDisplay = document.getElementById("content-display");

    // Define paths to HTML files for each module
    const modulePaths = {
        "crop-recommendation": "/templates/index.html",
        "crop-yield-prediction": "path/to/crop-yield-prediction.html",
        "advisory": "path/to/advisory.html",
        "blockchain": "path/to/blockchain.html"
    };

    // Function to load content into #content-display
    function loadContent(moduleId) {
        const url = modulePaths[moduleId];
        if (url) {
            fetch(url)
                .then(response => response.text())
                .then(html => {
                    contentDisplay.innerHTML = html;
                })
                .catch(error => {
                    contentDisplay.innerHTML = `<p>Error loading content: ${error}</p>`;
                });
        } else {
            contentDisplay.innerHTML = `<p>No content available for this module.</p>`;
        }
    }

    // Add event listeners to each module link
    document.querySelectorAll(".module").forEach(function(module) {
        module.addEventListener("click", function(event) {
            event.preventDefault();
            const moduleId = module.getAttribute("data-module");
            loadContent(moduleId);
        });
    });
});
