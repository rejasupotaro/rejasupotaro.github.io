class NavComponent extends HTMLElement {
    constructor() {
        super();
    }

    async connectedCallback() {
        try {
            const response = await fetch('components/nav-template.html');
            const html = await response.text();
            this.innerHTML = html;

            // Update navigation links based on current page
            const currentPath = window.location.pathname;
            const navLinks = this.querySelectorAll('.nav-links a');
            
            navLinks.forEach(link => {
                const linkPath = link.getAttribute('href');
                if (currentPath.endsWith(linkPath)) {
                    link.classList.add('active');
                }
            });
        } catch (error) {
            console.error('Failed to load navigation:', error);
            this.innerHTML = '<p>Error loading navigation</p>';
        }
    }
}

customElements.define('nav-component', NavComponent); 