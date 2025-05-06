class NavComponent extends HTMLElement {
    constructor() {
        super();
    }

    async connectedCallback() {
        try {
            // Check if we're running on GitHub Pages
            const isGitHubPages = window.location.hostname.includes('github.io');
            const isInPages = window.location.pathname.includes('/pages/');
            
            // Calculate paths based on environment
            let baseUrl = '';
            let relativePath = isInPages ? '../' : './';
            
            if (isGitHubPages) {
                // For GitHub Pages, get the repository path
                const pathParts = window.location.pathname.split('/');
                // Remove empty strings and the current page
                const validParts = pathParts.filter(part => part && !part.includes('.html'));
                // Reconstruct the base URL
                baseUrl = validParts.length > 0 ? '/' + validParts.join('/') : '';
            }
            
            // Construct the full path for the template
            const templatePath = `${baseUrl}${relativePath}components/nav-template.html`;
            console.log('Loading template from:', templatePath); // Debug log
            const response = await fetch(templatePath);
            const html = await response.text();
            
            // Create a temporary div to parse the HTML
            const temp = document.createElement('div');
            temp.innerHTML = html;
            
            // Get the template content
            const template = temp.querySelector('#nav-template');
            if (template) {
                // Clone the template content and append it
                const content = template.content.cloneNode(true);
                
                // Update all href attributes to use the correct base path
                const links = content.querySelectorAll('a');
                links.forEach(link => {
                    const href = link.getAttribute('href');
                    if (href.startsWith('pages/')) {
                        // For GitHub Pages, ensure we have the correct path structure
                        const pagePath = isGitHubPages ? `${baseUrl}/${href}` : `${baseUrl}${relativePath}${href}`;
                        link.setAttribute('href', pagePath);
                    } else if (href === 'index.html') {
                        // For the logo link, use the appropriate root path
                        if (isGitHubPages) {
                            // On GitHub Pages, go to the repository root
                            link.setAttribute('href', baseUrl || '/');
                        } else {
                            // Locally, go to the current directory
                            link.setAttribute('href', isInPages ? '../' : './');
                        }
                    }
                });
                
                this.appendChild(content);
            } else {
                throw new Error('Template not found');
            }

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