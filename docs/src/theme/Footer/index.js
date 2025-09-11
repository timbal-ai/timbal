import React from 'react';
import Link from '@docusaurus/Link';
import {
  //faGlobe,
  faGithub,
  faLinkedin,
} from '@fortawesome/free-brands-svg-icons';
import {
    faGlobe,
} from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="custom-footer">
      <div className="social-icons">
        <Link to="https://timbal.ai"><FontAwesomeIcon icon={faGlobe} /></Link>
        <Link to="https://github.com/timbal-ai/timbal"><FontAwesomeIcon icon={faGithub} /></Link>
        <Link to="https://www.linkedin.com/company/timbal-ai/"><FontAwesomeIcon icon={faLinkedin} /></Link>
      </div>
    </footer>
  );
}