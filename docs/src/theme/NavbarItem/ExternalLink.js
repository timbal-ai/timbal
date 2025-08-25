import React from 'react';
import OriginalExternalLink from '@theme-original/NavbarItem/ExternalLink';

export default function ExternalLink(props) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', gap: '0.25em' }}>
      <OriginalExternalLink {...props} />
    </span>
  );
} 