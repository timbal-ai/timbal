import React from 'react';
import { Highlight, themes, Prism } from 'prism-react-renderer';

// Make Prism available globally (needed for language extension)
(typeof global !== 'undefined' ? global : window).Prism = Prism;

export default function PrismHighlight({ children, ...props }) {
  // Map of words to their custom CSS classes
  const highlightMap = {
    Agent: 'custom-highlight-pink',
    from: 'custom-highlight-orange',
    // Add more words and classes as needed
  };

  // Build a regex to match any of the words
  const regex = new RegExp(`\\b(${Object.keys(highlightMap).join('|')})\\b`, 'g');

  // Convert children to string if it's not already
  const childrenStr = typeof children === 'string' ? children : String(children);

  // Replace each word with a span with the appropriate class
  const highlighted = childrenStr.replace(
    regex,
    (match) => `<span class="${highlightMap[match]}">${match}</span>`
  );

  const code = props.code || (typeof props.children === 'string' ? props.children : String(props.children));
  if (typeof code !== 'string') {
    throw new Error('CodeBlock: code must be a string');
  }

  return (
    <Highlight 
      theme={themes.github}
      code={highlighted}
      language={props.language || 'python'}
    >
      {({ className, style, tokens, getLineProps, getTokenProps }) => (
        <pre className={className} style={style}>
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })}>
              {line.map((token, key) => (
                <span key={key} {...getTokenProps({ token })} />
              ))}
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  );
} 