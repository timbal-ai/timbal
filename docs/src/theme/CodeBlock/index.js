import React, { useState } from 'react';
import { Highlight, Prism } from 'prism-react-renderer';

// Make Prism available globally (needed for language extension)
(typeof global !== 'undefined' ? global : window).Prism = Prism;

// Monokai theme
const monokaiTheme = {
  plain: {
    color: '#f8f8f2',
    backgroundColor: '#272822',
  },
  styles: [
    {
      types: ['comment', 'prolog', 'doctype', 'cdata'],
      style: {
        color: '#75715e',
      },
    },
    {
      types: ['punctuation'],
      style: {
        color: '#f8f8f2',
      },
    },
    {
      types: ['namespace'],
      style: {
        opacity: 0.7,
      },
    },
    {
      types: ['property', 'tag', 'constant', 'symbol', 'deleted'],
      style: {
        color: '#f92672',
      },
    },
    {
      types: ['boolean', 'number'],
      style: {
        color: '#ae81ff',
      },
    },
    {
      types: ['selector', 'attr-name', 'string', 'char', 'builtin', 'inserted'],
      style: {
        color: '#e6db74',
      },
    },
    {
      types: ['operator', 'entity', 'url', 'variable'],
      style: {
        color: '#f8f8f2',
      },
    },
    {
      types: ['atrule', 'attr-value', 'function', 'class-name'],
      style: {
        color: '#a6e22e',
      },
    },
    {
      types: ['keyword'],
      style: {
        color: '#66d9ef',
      },
    },
    {
      types: ['regex', 'important'],
      style: {
        color: '#fd971f',
      },
    },
    {
      types: ['important', 'bold'],
      style: {
        fontWeight: 'bold',
      },
    },
    {
      types: ['italic'],
      style: {
        fontStyle: 'italic',
      },
    },
    {
      types: ['entity'],
      style: {
        cursor: 'help',
      },
    },
  ],
};

// Custom highlighting for specific keywords
const highlightMap = {
  Agent: 'custom-highlight-green',
  Workflow: 'custom-highlight-green',
  Tool: 'custom-highlight-green',
  StartEvent: 'custom-highlight-green',
  ChunkEvent: 'custom-highlight-green',
  OutputEvent: 'custom-highlight-green',
  kwargs: 'custom-highlight-orange',
  from: 'custom-highlight-pink',
  for: 'custom-highlight-pink',
  in: 'custom-highlight-pink',
  if: 'custom-highlight-pink',
  and: 'custom-highlight-pink',
  elif: 'custom-highlight-pink',
  else: 'custom-highlight-pink',
  type: 'custom-highlight-blue',
  break: 'custom-highlight-pink',
  pass: 'custom-highlight-pink',
  isinstance: 'custom-highlight-blue',
  try: 'custom-highlight-pink',
  except: 'custom-highlight-pink',
  Exception: 'custom-highlight-blue',
  import: 'custom-highlight-pink',
  await: 'custom-highlight-pink',
  async: 'custom-highlight-pink',
  def: 'custom-highlight-blue',
  class: 'custom-highlight-blue',
  as: 'custom-highlight-pink',
  return: 'custom-highlight-pink',
  '*': 'custom-highlight-pink',
  '+': 'custom-highlight-pink',
  '=': 'custom-highlight-pink',
  '{': 'custom-highlight-purple-pink',
  '}': 'custom-highlight-purple-pink',
  '(': 'custom-highlight-yellow',
  ')': 'custom-highlight-yellow',
  '[': 'custom-highlight-purple-pink',
  ']': 'custom-highlight-purple-pink',
  str: 'custom-highlight-blue',
  float: 'custom-highlight-blue',
  int: 'custom-highlight-blue',
  bool: 'custom-highlight-blue',
};

const regex = new RegExp(
  `(${Object.keys(highlightMap)
    .map(word =>
      /^[a-zA-Z0-9_]+$/.test(word)
        ? `\\b${word}\\b`
        : word.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')
    )
    .join('|')})`,
  'g'
);

export default function CodeBlock(props) {
  const { title, children, highlight } = props;
  const [copied, setCopied] = useState(false);

  // Parse highlight prop to get line numbers
  const highlightedLines = highlight ? highlight.split(',').map(num => parseInt(num.trim())) : [];

  return (
    <div
      className="custom-codeblock-container"
      style={{
        borderRadius: '12px',
        overflow: 'hidden',
        border: '1px solid #333',
        marginBottom: '1.5em',
        position: 'relative',
      }}
    >
      {title && (
        <div
          style={{
            background: '#2d2d2d',
            color: '#fff',
            padding: '0.3em 1em',
            fontSize: '0.85em',
            borderBottom: '1.5px solid #444',
            letterSpacing: '0.01em',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span>{title}</span>
        </div>
      )}
      <div style={{ position: 'relative' }}>
        <Highlight 
          theme={monokaiTheme} 
          {...props}
        >
          {({ className, style, tokens, getLineProps, getTokenProps }) => {
            const codeString = tokens.map(
              line => line.map(token => token.content).join('')
            ).join('\n');

            // Extract function parameters from def lines
            const functionParams = new Set();
            tokens.forEach(line => {
              const lineContent = line.map(token => token.content).join('');
              const defMatch = lineContent.match(/def\s+\w+\s*\(([^)]*)\)/);
              if (defMatch) {
                const params = defMatch[1];
                // Extract parameter names (handle type hints like a: float)
                const paramMatches = params.match(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*[^,)]+)?/g);
                if (paramMatches) {
                  paramMatches.forEach(param => {
                    const paramName = param.split(':')[0].trim();
                    if (paramName && paramName !== 'self') {
                      functionParams.add(paramName);
                    }
                  });
                }
              }
            });


            return (
              <>
                <button
                  className={`copy-icon-btn${copied ? ' copied' : ''}`}
                  onClick={() => {
                    navigator.clipboard.writeText(codeString);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000);
                  }}
                  aria-label="Copy code"
                  type="button"
                  style={{
                    position: 'absolute',
                    top: title ? 16 : 12,
                    right: '1em',
                    zIndex: 10,
                    background: 'rgba(30,30,30,0.7)',
                    border: 'none',
                    borderRadius: '6px',
                    padding: '4px',
                    cursor: 'pointer',
                    pointerEvents: 'auto',
                    transition: 'opacity 0.2s, background 0.2s',
                  }}
                >
                  {copied ? (
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><path fill="#fff" d="m9.55 15.15l8.475-8.475q.3-.3.7-.3t.7.3t.3.713t-.3.712l-9.175 9.2q-.3.3-.7.3t-.7-.3L4.55 13q-.3-.3-.288-.712t.313-.713t.713-.3t.712.3z"/></svg>
                  ) : (
                    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><g fill="none" stroke="#fff" strokeWidth="1.5"><path d="M6 11c0-2.828 0-4.243.879-5.121C7.757 5 9.172 5 12 5h3c2.828 0 4.243 0 5.121.879C21 6.757 21 8.172 21 11v5c0 2.828 0 4.243-.879 5.121C19.243 22 17.828 22 15 22h-3c-2.828 0-4.243 0-5.121-.879C6 20.243 6 18.828 6 16z"/><path d="M6 19a3 3 0 0 1-3-3v-6c0-3.771 0-5.657 1.172-6.828S7.229 2 11 2h4a3 3 0 0 1 3 3"/></g></svg>
                  )}
                </button>
                <pre className={className} style={{
                  ...style,
                  border: 'none',
                  borderRadius: 0,
                  margin: 0,
                  position: 'relative',
                }}>
                  {tokens.map((line, i) => {
                    const isHighlighted = highlightedLines.includes(i + 1);
                    
                    const lineStyle = {
                      ...getLineProps({ line }).style,
                      ...(isHighlighted ? {
                        backgroundColor: 'rgba(255, 255, 255, 0.1)',
                        borderLeft: '3px solid #bcb6ff',
                        paddingLeft: '8px',
                        marginLeft: '-8px',
                      } : {})
                    };

                    // Check if line is a comment
                    const lineContent = line.map(token => token.content).join('');
                    if (/^\s*#/.test(lineContent)) {
                      return (
                        <div 
                          key={i} 
                          {...getLineProps({ line })}
                          style={lineStyle}
                        >
                          {line.map((token, key) => (
                            <span
                              key={key}
                              className="custom-highlight-comment"
                            >
                              {token.content}
                            </span>
                          ))}
                        </div>
                      );
                    }

                    return (
                      <div 
                        key={i} 
                        {...getLineProps({ line })}
                        style={lineStyle}
                      >
                        {line.map((token, key) => {
                          const tokenProps = getTokenProps({ token });
                          let content = tokenProps.children;

                          // Skip highlighting for whitespace/spaces FIRST
                          if (typeof content === 'string' && (/^\s+$/.test(content) || content === ' ' || content === '\t')) {
                            return <span key={key} {...tokenProps} />;
                          }

                          // For bash blocks, keep white
                          if (props.language === 'bash') {
                            return (
                              <span key={key} style={{ color: '#ffffff' }}>
                                {content}
                              </span>
                            );
                          }

                          // Check if we're inside parentheses for kwargs detection (functions and classes)
                          if (typeof content === 'string' && content.trim() && !content.includes('=') && !content.includes('(') && !content.includes(')')) {
                            // Check if the next token is = and we're inside parentheses
                            const nextToken = key < line.length - 1 ? line[key + 1] : null;
                            const nextContent = nextToken ? getTokenProps({ token: nextToken }).children : '';
                            
                            if (nextContent === '=' || nextContent.includes('=')) {
                              // Check if we're inside parentheses by counting them across all previous lines and current line
                              let openCount = 0;
                              let closeCount = 0;
                              
                              // Count parentheses in all previous lines
                              for (let lineIndex = 0; lineIndex < i; lineIndex++) {
                                const prevLine = tokens[lineIndex];
                                for (let tokenIndex = 0; tokenIndex < prevLine.length; tokenIndex++) {
                                  const tokenContent = getTokenProps({ token: prevLine[tokenIndex] }).children;
                                  if (tokenContent === '(') openCount++;
                                  if (tokenContent === ')') closeCount++;
                                }
                              }
                              
                              // Count parentheses in current line up to this token
                              for (let tokenIndex = 0; tokenIndex < key; tokenIndex++) {
                                const tokenContent = getTokenProps({ token: line[tokenIndex] }).children;
                                if (tokenContent === '(') openCount++;
                                if (tokenContent === ')') closeCount++;
                              }
                              
                              if (openCount > closeCount) {
                                return <span key={key} className="custom-highlight-orange">{content}</span>;
                              }
                            }
                          }

                          // Check if this is a function name being used as a value (like handler=add)
                          if (typeof content === 'string' && content.trim()) {
                            // Check if previous token was = and we're inside parentheses
                            const prevToken = key > 0 ? line[key - 1] : null;
                            const prevContent = prevToken ? getTokenProps({ token: prevToken }).children : '';
                            
                            if (prevContent === '=' || prevContent.includes('=')) {
                              // Check if we're inside parentheses
                              let openCount = 0;
                              let closeCount = 0;
                              
                              // Count parentheses in all previous lines
                              for (let lineIndex = 0; lineIndex < i; lineIndex++) {
                                const prevLine = tokens[lineIndex];
                                for (let tokenIndex = 0; tokenIndex < prevLine.length; tokenIndex++) {
                                  const tokenContent = getTokenProps({ token: prevLine[tokenIndex] }).children;
                                  if (tokenContent === '(') openCount++;
                                  if (tokenContent === ')') closeCount++;
                                }
                              }
                              
                              // Count parentheses in current line up to this token
                              for (let tokenIndex = 0; tokenIndex < key; tokenIndex++) {
                                const tokenContent = getTokenProps({ token: line[tokenIndex] }).children;
                                if (tokenContent === '(') openCount++;
                                if (tokenContent === ')') closeCount++;
                              }
                              
                              if (openCount > closeCount) {
                                // Check if this looks like a function name (not a string, number, or built-in)
                                if (!content.includes('"') && !content.includes("'") && 
                                    !/^\d+(\.\d+)?$/.test(content) && 
                                    !['True', 'False', 'None'].includes(content) &&
                                    /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(content)) {
                                  return <span key={key} className="custom-highlight-green-fn">{content}</span>;
                                }
                              }
                            }
                          }

                          // Check if this token comes after 'from' in import statements
                          if (typeof content === 'string' && content.trim()) {
                            // Look for 'from' in previous tokens on this line
                            let hasFromBefore = false;
                            let hasImportAfter = false;
                            
                            for (let tokenIndex = 0; tokenIndex < key; tokenIndex++) {
                              const tokenContent = getTokenProps({ token: line[tokenIndex] }).children;
                              if (tokenContent === 'from') {
                                hasFromBefore = true;
                                break;
                              }
                            }
                            
                            // Check if 'import' appears after this token
                            for (let tokenIndex = key + 1; tokenIndex < line.length; tokenIndex++) {
                              const tokenContent = getTokenProps({ token: line[tokenIndex] }).children;
                              if (tokenContent === 'import') {
                                hasImportAfter = true;
                                break;
                              }
                            }

                            // If we found 'from' before and 'import' after, highlight the identifier part only
                            if (hasFromBefore && hasImportAfter) {
                              const trimmed = content.trim();
                              if (/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(trimmed)) {
                                // Replace only the identifier part, keep the whitespace
                                const highlightedContent = content.replace(
                                  /([a-zA-Z_][a-zA-Z0-9_]*)/,
                                  '<span class="custom-highlight-green">$1</span>'
                                );
                                return (
                                  <span
                                    key={key}
                                    dangerouslySetInnerHTML={{ __html: highlightedContent }}
                                  />
                                );
                              }
                            }
                          }

                          // Check if this is a function parameter
                          if (typeof content === 'string' && functionParams.has(content.trim())) {
                            return <span key={key} className="custom-highlight-orange">{content}</span>;
                          }


                          // Special handling for 'async' keyword based on context
                          if (typeof content === 'string' && content.trim() === 'async') {
                            // Check what comes after async
                            let nextToken = '';
                            for (let tokenIndex = key + 1; tokenIndex < line.length; tokenIndex++) {
                              const nextTokenContent = getTokenProps({ token: line[tokenIndex] }).children;
                              if (nextTokenContent && nextTokenContent.trim()) {
                                nextToken = nextTokenContent.trim();
                                break;
                              }
                            }
                            
                            // async def -> blue, async for -> pink
                            const colorClass = nextToken === 'def' ? 'custom-highlight-blue' : 'custom-highlight-pink';
                            const highlightedContent = content.replace(
                              /(async)/,
                              `<span class="${colorClass}">$1</span>`
                            );
                            return (
                              <span
                                key={key}
                                dangerouslySetInnerHTML={{ __html: highlightedContent }}
                              />
                            );
                          }

                          // Handle f-strings - check if this token contains an f-string
                          if (typeof content === 'string' && /^f['"]/.test(content)) {
                            // Split the f-string to highlight only the 'f' prefix
                            const match = content.match(/^(f)(['"].*['"]?)$/);
                            if (match) {
                              return (
                                <span key={key}>
                                  <span style={{ color: '#66d9ef' }}>{match[1]}</span>
                                  <span style={{ color: '#e6db74' }}>{match[2]}</span>
                                </span>
                              );
                            }
                          }

                          // Apply custom highlighting (but skip if token is already a string or comment)
                          if (typeof content === 'string' && !token.types.includes('string') && !token.types.includes('comment')) {
                            const highlightedContent = content.replace(
                              regex,
                              (match) => `<span class="${highlightMap[match]}">${match}</span>`
                            );

                            if (highlightedContent !== content) {
                              return (
                                <span
                                  key={key}
                                  dangerouslySetInnerHTML={{ __html: highlightedContent }}
                                />
                              );
                            }
                          }

                          // Default token rendering
                          return (
                            <span key={key} {...tokenProps} />
                          );
                        })}
                      </div>
                    );
                  })}
                </pre>
              </>
            );
          }}
        </Highlight>
      </div>
    </div>
  );
}