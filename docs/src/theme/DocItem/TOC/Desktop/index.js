import React, {useEffect, useRef} from 'react';
import TOCDesktop from '@theme-original/DocItem/TOC/Desktop';

export default function TOCDesktopWrapper(props) {
  const observerRef = useRef(null);
  const isClickingRef = useRef(false);
  
  useEffect(() => {
    // Force re-initialize the TOC highlighting after component mounts
    const initTOCHighlighting = () => {
      // Get all headings
      const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
      const tocLinks = document.querySelectorAll('.theme-doc-toc-desktop .table-of-contents__link');
      
      if (headings.length === 0 || tocLinks.length === 0) return;
      
      // Add click event listeners to TOC links
      tocLinks.forEach(link => {
        link.addEventListener('click', (e) => {
          // Remove all active classes first
          tocLinks.forEach(l => l.classList.remove('table-of-contents__link--active'));
          // Add active class to clicked link
          e.target.classList.add('table-of-contents__link--active');
          
          // Set flag to prevent observer from interfering
          isClickingRef.current = true;
          
          // Reset flag after scroll completes
          setTimeout(() => {
            isClickingRef.current = false;
          }, 1000);
        });
      });
      
      let currentActive = null;

      // Create proper intersection observer
      const observer = new IntersectionObserver(
        (entries) => {
          if (isClickingRef.current) return;
          
          // Remove all active classes
          tocLinks.forEach(link => link.classList.remove('table-of-contents__link--active'));
          
          // Find the most relevant heading based on intersection
          let bestEntry = null;
          
          for (const entry of entries) {
            if (entry.isIntersecting) {
              if (!bestEntry || entry.boundingClientRect.top < bestEntry.boundingClientRect.top) {
                bestEntry = entry;
              }
            }
          }
          
          // If no heading is intersecting, keep the last active one
          if (!bestEntry && currentActive) {
            bestEntry = { target: currentActive };
          }
          
          // If still no heading, use first one
          if (!bestEntry && headings.length > 0) {
            bestEntry = { target: headings[0] };
          }
          
          if (bestEntry && bestEntry.target.id) {
            currentActive = bestEntry.target;
            const activeLink = document.querySelector(`.theme-doc-toc-desktop .table-of-contents__link[href="#${bestEntry.target.id}"]`);
            if (activeLink) {
              activeLink.classList.add('table-of-contents__link--active');
            }
          }
        },
        {
          rootMargin: '-10% 0px -80% 0px',
          threshold: 0
        }
      );
      
      observerRef.current = observer;
      
      // Observe all headings
      headings.forEach(heading => {
        if (heading.id) {
          observer.observe(heading);
        }
      });
      
      // Simple initial highlighting - just activate first TOC link
      if (tocLinks.length > 0) {
        tocLinks[0].classList.add('table-of-contents__link--active');
      }
      
      return () => {
        if (observerRef.current) {
          observerRef.current.disconnect();
        }
      };
    };
    
    // Initialize after a short delay to ensure DOM is ready
    const timeoutId = setTimeout(initTOCHighlighting, 100);
    
    return () => {
      clearTimeout(timeoutId);
      if (observerRef.current) {
        observerRef.current.disconnect();
      }
    };
  }, []);
  
  return <TOCDesktop {...props} />;
}