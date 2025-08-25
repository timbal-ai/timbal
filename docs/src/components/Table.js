import React from 'react';

export default function Table({children, className}) {
  return (
    <table className={className} style={{width: '100%'}}>
      {children}
    </table>
  );
}