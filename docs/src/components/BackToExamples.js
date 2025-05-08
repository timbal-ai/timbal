import React from 'react';
import Link from '@docusaurus/Link';

export default function BackToExamples() {
  return (
    <div style={{ marginBottom: '1.5rem', marginLeft: 0, paddingLeft: 0 }}>
      <Link
        to="/docs/examples/"
        style={{
          display: 'inline-block',
          padding: '4px 12px',
          background: 'transparent',
          color: 'var(--ifm-color-primary)',
          border: '1.5px solid var(--ifm-color-primary)',
          borderRadius: '5px',
          textDecoration: 'none',
          fontWeight: 500,
          fontSize: '0.98rem',
          boxShadow: 'none',
          transition: 'background 0.2s, color 0.2s',
        }}
        onMouseOver={e => {
          e.target.style.background = 'var(--ifm-color-primary)';
          e.target.style.color = '#fff';
        }}
        onMouseOut={e => {
          e.target.style.background = 'transparent';
          e.target.style.color = 'var(--ifm-color-primary)';
        }}
      >
        ← Back to Examples
      </Link>
    </div>
  );
}