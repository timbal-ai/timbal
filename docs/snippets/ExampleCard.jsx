export const ExampleCard = ({ title, href = "#" }) => (
  <a
    href={href}
    style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      padding: '8px 16px',
      textAlign: 'center',
      backgroundColor: 'white',
      border: '1px solid #e1e5e9',
      borderRadius: '6px',
      textDecoration: 'none',
      color: 'black',
      fontSize: '14px',
      fontWeight: 'normal',
      transition: 'all 0.2s'
    }}
    onMouseEnter={(e) => e.target.style.borderColor = 'black'}
    onMouseLeave={(e) => e.target.style.borderColor = '#e1e5e9'}
  >
    {title}
  </a>
);

export const ExampleGrid = ({ children }) => (
  <div style={{
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
    gap: '12px',
    marginTop: '24px'
  }}>
    {children}
  </div>
);

