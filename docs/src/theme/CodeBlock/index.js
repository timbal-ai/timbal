import React, { useState } from 'react';
import { Highlight, themes, Prism } from 'prism-react-renderer';

// Make Prism available globally (needed for language extension)
(typeof global !== 'undefined' ? global : window).Prism = Prism;

// Custom theme with black background
const customTheme = {
  ...themes.github,
  plain: {
    ...themes.github.plain,
    backgroundColor: '#2d2d2d',
    color: '#ffffff',
  },
  styles: themes.github.styles.map(style => ({
    ...style,
    types: style.types,
  })),
};

// Map of words to their custom CSS classes
const highlightMap = {
  from: 'custom-highlight-pink',
  import: 'custom-highlight-pink',
  '=' : 'custom-highlight-pink',
  timbal: 'custom-highlight-green',
  Agent: 'custom-highlight-green',
  steps: 'custom-highlight-green',
  perplexity: 'custom-highlight-green',
  InMemorySaver: 'custom-highlight-green',
  fal: 'custom-highlight-green',
  text_to_image: 'custom-highlight-green',
  RunContext: 'custom-highlight-green',
  twilio: 'custom-highlight-green',
  whatsapp: 'custom-highlight-green',
  model: 'custom-highlight-orange',
  query: 'custom-highlight-orange',
  context: 'custom-highlight-orange',
  system_prompt: 'custom-highlight-orange',
  tools: 'custom-highlight-orange',
  parent_id: 'custom-highlight-orange',
  id: 'custom-highlight-orange',
  model_id: 'custom-highlight-orange',
  to: 'custom-highlight-orange',
  message: 'custom-highlight-orange',
  resource: 'custom-highlight-orange',
  max_results: 'custom-highlight-orange',
  message_id: 'custom-highlight-orange',
  thread_id: 'custom-highlight-orange',
  subject: 'custom-highlight-orange',
  body: 'custom-highlight-orange',
  template_sid: 'custom-highlight-orange',
  template_params: 'custom-highlight-orange',
  scope: 'custom-highlight-orange',
  folder: 'custom-highlight-orange',
  destination: 'custom-highlight-orange',
  text: 'custom-highlight-orange',
  voice_id: 'custom-highlight-orange',
  memory_id: 'custom-highlight-orange',
  location: 'custom-highlight-orange',
  memory_window_size: 'custom-highlight-orange',
  '(': 'custom-highlight-yellow-dark',
  ')': 'custom-highlight-yellow-dark',
  '[': 'custom-highlight-purple-pink',
  ']': 'custom-highlight-purple-pink',
  '{': 'custom-highlight-yellow-dark',
  '}': 'custom-highlight-yellow-dark',
  ',': '#fffff',
  'Flow': 'custom-highlight-green',
  await: 'custom-highlight-pink',
  def: 'custom-highlight-blue',
  async: 'custom-highlight-blue',
  '-': '#fffff',
  '>': '#fffff',
  ':': '#fffff',
  '.' : '#fffff',
  state: 'custom-highlight-green',
  savers: 'custom-highlight-green',
  sharepoint: 'custom-highlight-green',
  Tool: 'custom-highlight-green',
  list_directory: 'custom-highlight-green-fn',
  gen_images: 'custom-highlight-green-fn',
  download_file: 'custom-highlight-green-fn',
  TimbalPlatformSaver: 'custom-highlight-green',
  sql: 'custom-highlight-green',
  postgres: 'custom-highlight-green',
  return: 'custom-highlight-pink',
  for: 'custom-highlight-pink',
  in: 'custom-highlight-pink',
  if: 'custom-highlight-pink',
  datetime: 'custom-highlight-green',
  elevenlabs: 'custom-highlight-green',
  gmail: 'custom-highlight-green',
  JSONLSaver: 'custom-highlight-green',
  messages: 'custom-highlight-green',
  types: 'custom-highlight-green',
  File: 'custom-highlight-green',
  postgres_query: 'custom-highlight-green-fn',
  PGConfig: 'custom-highlight-green',
  get_message: 'custom-highlight-green-fn',
  get_thread: 'custom-highlight-green-fn',
  get_datetime: 'custom-highlight-green-fn',
  send_message: 'custom-highlight-green-fn',
  run_sql_query: 'custom-highlight-green-fn',
  create_draft_message: 'custom-highlight-green-fn',
  send_whatsapp_message: 'custom-highlight-green-fn',
  send_whatsapp_template: 'custom-highlight-green-fn',
  validate: 'custom-highlight-green-fn',
  complete: 'custom-highlight-green-fn',
  print: 'custom-highlight-green-fn',
  search: 'custom-highlight-green-fn',
  search_internet: 'custom-highlight-green-fn',
  stt: 'custom-highlight-green-fn',
  tts: 'custom-highlight-green-fn',
  run: 'custom-highlight-green-fn',
  search_web: 'custom-highlight-green-fn',
  set_input: 'custom-highlight-green-fn',
  extract_text: 'custom-highlight-green-fn',
  text_processor: 'custom-highlight-green-fn',
  convert_to_database: 'custom-highlight-green-fn',
  get_weather: 'custom-highlight-green-fn',
  get_time: 'custom-highlight-green-fn',
  add_step: 'custom-highlight-green-fn',
  add_link: 'custom-highlight-green-fn',
  set_output: 'custom-highlight-green-fn',
  parse_documentation: 'custom-highlight-green-fn',
  create_database: 'custom-highlight-green-fn',
  add_llm: 'custom-highlight-green-fn',
  compile: 'custom-highlight-green-fn',
  str: 'custom-highlight-green'

  // Add more words and classes as needed
};

// Build a regex to match any of the words or symbols
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

// Regex to match text between double quotes
const quoteRegex = /"([^"]*)"/g;

// Regex to match numbers
const numberRegex = /\\b\\d+(?:\\.\\d+)?\\b/g;

export default function CodeBlock(props) {
  const { title, children } = props;
  const [copied, setCopied] = useState(false);

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
      <Highlight 
        theme={customTheme} 
        {...props}
      >
        {({ className, style, tokens, getLineProps, getTokenProps }) => {
          // Extract the code string from tokens
          const codeString = tokens.map(
            line => line.map(token => token.content).join('')
          ).join('\n');

          const handleCopy = () => {
            navigator.clipboard.writeText(codeString);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          };

          let promptCount = 0;
          let stateSaverTotal = 0;
          let audioFileTotal = 0;
          let dbConfigTotal = 0;
          let sqlTotal = 0;
          tokens.forEach(line => {
            line.forEach(token => {
              const content = getTokenProps({ token }).children;
              if (typeof content === 'string' && content.trim() === 'state_saver') {
                stateSaverTotal++;
              }
              if (typeof content === 'string' && content.trim() === 'audio_file') {
                audioFileTotal++;
              }
              if (typeof content === 'string' && content.trim() === 'db_config') {
                dbConfigTotal++;
              }
              if (typeof content === 'string' && content.trim() === 'sql') {
                sqlTotal++;
              }
            });
          });
          let stateSaverCount = 0;
          let audioFileCount = 0;
          let dbConfigCount = 0;
          let sqlCount = 0;
          // At the top of your render function:
          let bracketStack = [];
          const getBracketColor = (depth) => {
            const mod = (depth - 1) % 3;
            if (mod === 0) return 'custom-highlight-yellow-dark'; // yellow class
            if (mod === 1) return { color: '#ff69b4' }; // pink
            return { color: '#42a5f5' }; // blue
          };

          return (
            <pre className={className} style={{
              ...style,
              border: 'none',
              borderRadius: 0,
              color: props.language === 'bash' ? '#ffffff' : style.color,
              margin: 0,
              position: 'relative',
            }}>
              <button
                className={`copy-icon-btn${copied ? ' copied' : ''}`}
                onClick={handleCopy}
                aria-label="Copy code"
                type="button"
              >
                {copied ? (
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><path fill="#fff" d="m9.55 15.15l8.475-8.475q.3-.3.7-.3t.7.3t.3.713t-.3.712l-9.175 9.2q-.3.3-.7.3t-.7-.3L4.55 13q-.3-.3-.288-.712t.313-.713t.713-.3t.712.3z"/></svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"><g fill="none" stroke="#fff" strokeWidth="1.5"><path d="M6 11c0-2.828 0-4.243.879-5.121C7.757 5 9.172 5 12 5h3c2.828 0 4.243 0 5.121.879C21 6.757 21 8.172 21 11v5c0 2.828 0 4.243-.879 5.121C19.243 22 17.828 22 15 22h-3c-2.828 0-4.243 0-5.121-.879C6 20.243 6 18.828 6 16z"/><path d="M6 19a3 3 0 0 1-3-3v-6c0-3.771 0-5.657 1.172-6.828S7.229 2 11 2h4a3 3 0 0 1 3 3"/></g></svg>
                )}
              </button>
              {tokens.map((line, i) => {
                // Join the line's tokens into a string to check for comment
                const lineContent = line.map(token => getTokenProps({ token }).children).join('');
                if (/^\s*#/.test(lineContent)) {
                  // If the line is a comment, render it as a single gray, italic line, preserving indentation
                  return (
                    <div key={i} {...getLineProps({ line })}>
                      {line.map((token, key) => {
                        const tokenProps = getTokenProps({ token });
                        return (
                          <span
                            key={key}
                            className={props.language === 'bash' ? 'custom-highlight-comment' : 'custom-highlight-comment'}
                          >
                            {tokenProps.children}
                          </span>
                        );
                      })}
                    </div>
                  );
                }
                return (
                  <div key={i} {...getLineProps({ line })}>
                    {line.map((token, key) => {
                      const tokenProps = getTokenProps({ token });
                      let content = tokenProps.children;
                      // For bash blocks, just return the content in white
                      if (props.language === 'bash') {
                        return (
                          <span
                            key={key}
                            style={{ color: '#ffffff' }}
                          >
                            {content}
                          </span>
                        );
                      }
                      // Split the content into quoted and non-quoted segments
                      const segments = [];
                      let lastIndex = 0;
                      let match;
                      while ((match = quoteRegex.exec(content)) !== null) {
                        // Push non-quoted segment
                        if (match.index > lastIndex) {
                          const nonQuoted = content.slice(lastIndex, match.index);
                          // Apply word/symbol highlighting to non-quoted segment
                          let highlightedNonQuoted = nonQuoted.replace(
                            regex,
                            (m) => `<span class=\"${highlightMap[m]}\">${m}</span>`
                          );
                          // Now highlight numbers in the result (but not inside quotes)
                          highlightedNonQuoted = highlightedNonQuoted.replace(
                            /\\b\\d+(?:\\.\\d+)?\\b/g,
                            (num) => `<span class=\"custom-highlight-purple\">${num}</span>`
                          );
                          segments.push(highlightedNonQuoted);
                        }
                        // Push quoted segment with yellow highlight
                        segments.push(`<span class=\"custom-highlight-yellow\">${match[0]}</span>`);
                        lastIndex = match.index + match[0].length;
                      }
                      // Push any remaining non-quoted segment
                      if (lastIndex < content.length) {
                        const nonQuoted = content.slice(lastIndex);
                        let highlightedNonQuoted = nonQuoted.replace(
                          regex,
                          (m) => `<span class=\"${highlightMap[m]}\">${m}</span>`
                        );
                        // Now highlight numbers in the result (but not inside quotes)
                        highlightedNonQuoted = highlightedNonQuoted.replace(
                          /\\b\\d+(?:\\.\\d+)?\\b/g,
                          (num) => `<span class=\"custom-highlight-purple\">${num}</span>`
                        );
                        segments.push(highlightedNonQuoted);
                      }
                      const highlightedContent = segments.join('');
                      // Custom logic for 'prompt'
                      if (
                        typeof content === 'string' &&
                        content.trim() === 'prompt'
                      ) {
                        promptCount++;
                        if (promptCount === 1) {
                          // orange
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        if (promptCount === 2) {
                          // white
                          return <span key={key} style={{ color: '#fff' }}>{content}</span>;
                        }
                        // For 3rd and later, use default or another style
                        return <span key={key} {...tokenProps} />;
                      }
                      // --- audio_file logic ---
                      if (
                        typeof content === 'string' &&
                        content.trim() === 'audio_file'
                      ) {
                        audioFileCount++;
                        // If there are 3, highlight the second
                        if (audioFileTotal === 3 && audioFileCount === 2) {
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        // Otherwise, default
                        return <span key={key} {...tokenProps} />;
                      }
                      // --- state_saver logic ---
                      if (
                        typeof content === 'string' &&
                        content.trim() === 'state_saver'
                      ) {
                        stateSaverCount++;
                        // If there is only one, highlight it
                        if (stateSaverTotal === 1 && stateSaverCount === 1) {
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        // If there are 2, highlight the first
                        if (stateSaverTotal === 2 && stateSaverCount === 1) {
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        // If there are 3, highlight the second
                        if (stateSaverTotal === 3 && stateSaverCount === 2) {
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        // Otherwise, default
                        return <span key={key} {...tokenProps} />;
                      }
                      // --- db_config logic ---
                      if (
                        typeof content === 'string' &&
                        content.trim() === 'db_config'
                      ) {
                        dbConfigCount++;
                        // If there is only one, highlight it
                        if (dbConfigTotal === 1 && dbConfigCount === 1) {
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        // If there are 2, highlight the first
                        if (dbConfigTotal === 2 && dbConfigCount === 1) {
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        // If there are 3, highlight the second
                        if (dbConfigTotal === 3 && dbConfigCount === 2) {
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        // Otherwise, default
                        return <span key={key} {...tokenProps} />;
                      }
                      // --- sql logic ---
                      if (
                        typeof content === 'string' &&
                        content.trim() === 'sql'
                      ) {
                        sqlCount++;
                        // First occurrence is green
                        if (sqlCount === 1) {
                          return <span key={key} className="custom-highlight-green">{content}</span>;
                        }
                        // Next two occurrences are orange
                        if (sqlCount === 2 || sqlCount === 3) {
                          return <span key={key} className="custom-highlight-orange">{content}</span>;
                        }
                        // Otherwise, default
                        return <span key={key} {...tokenProps} />;
                      }
                      // --- Parenthesis coloring logic ---
                      if (content === '(' || content === '{' || content === '[') {
                        bracketStack.push(content);
                        const depth = bracketStack.length;
                        const color = getBracketColor(depth);
                        if (typeof color === 'string') {
                          return <span key={key} className={color}>{content}</span>;
                        } else {
                          return <span key={key} style={color}>{content}</span>;
                        }
                      }
                      if (content === ')' || content === '}' || content === ']') {
                        const depth = bracketStack.length;
                        const color = getBracketColor(depth);
                        let result;
                        if (typeof color === 'string') {
                          result = <span key={key} className={color}>{content}</span>;
                        } else {
                          result = <span key={key} style={color}>{content}</span>;
                        }
                        // Only pop if the stack is not empty and the last opened matches
                        const last = bracketStack[bracketStack.length - 1];
                        if (
                          (content === ')' && last === '(') ||
                          (content === '}' && last === '{') ||
                          (content === ']' && last === '[')
                        ) {
                          bracketStack.pop();
                        }
                        return result;
                      }
                      // Only use dangerouslySetInnerHTML if we have a match
                      if (highlightedContent !== content) {
                        return (
                          <span
                            key={key}
                            dangerouslySetInnerHTML={{ __html: highlightedContent }}
                          />
                        );
                      }
                      // Otherwise use regular children
                      return (
                        <span
                          key={key}
                          {...tokenProps}
                        />
                      );
                    })}
                  </div>
                );
              })}
            </pre>
          );
        }}
      </Highlight>
    </div>
  );
}
