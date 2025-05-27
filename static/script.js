document.addEventListener("DOMContentLoaded", () => {
    const hamburger = document.querySelector(".hamburger");
    const navMenu = document.querySelector(".nav-menu");

    if (hamburger && navMenu) {
        hamburger.addEventListener("click", () => {
            hamburger.classList.toggle("active");
            navMenu.classList.toggle("active");
        });
    }

    // Get current path's file name (e.g., "index.html" or "predictor.html")
    const currentPage = window.location.pathname.split("/").pop();

    // If no file (e.g., it's just domain.com/), treat it as "index.html"
    const current = currentPage === "" ? "index.html" : currentPage;

    // Highlight the active link
    document.querySelectorAll(".nav-link").forEach(link => {
        const href = link.getAttribute("href");

        if (href && href.endsWith(current)) {
            link.classList.add("active");
        } else {
            link.classList.remove("active");
        }
    });
});
