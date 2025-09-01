import React, { useState } from 'react';
import Link from '@docusaurus/Link';

export default function TimbalExamples() {
  const [activeSection, setActiveSection] = useState('agents');

  const sections = [
    { id: 'agents', name: 'Agents', examples: [
      //{ title: 'Calling Agents', link: '/examples/agents/calling_agents' },
      //{ title: 'Agent System Prompt', link: '/examples/agents/agent_system_prompt' },
      { title: 'Adding a Tool', link: '/examples/agents/adding_tool' },
      { title: 'Adding a Workflow', link: '/examples/agents/adding_workflow' },
      { title: 'Supervisor Agent', link: '/examples/agents/supervisor_agent' },
      { title: 'Image Analysis', link: '/examples/agents/image_analysis' },
      { title: 'Using Voice', link: '/examples/agents/using_voice' },
      //{ title: 'Dynamic Context', link: '/examples/agents/dynamic_context' }
    ]},
    { id: 'workflows', name: 'Workflows', examples: [
      { title: 'Running Workflows', link: '/examples/workflows/running_workflows' },
      { title: 'Sequential Steps', link: '/examples/workflows/sequential_steps' },
      { title: 'Parallel Steps', link: '/examples/workflows/parallel_steps' },
      { title: 'Conditional Branching', link: '/examples/workflows/conditional_branching' },
      { title: 'Array Input', link: '/examples/workflows/array_input' },
      { title: 'Calling Agent', link: '/examples/workflows/calling_agent' },
      { title: 'Agent Step', link: '/examples/workflows/agent_step' },
      { title: 'Tool Step', link: '/examples/workflows/tool_step' },
      { title: 'Human Loop', link: '/examples/workflows/human_loop' }
    ]},
    { id: 'tools', name: 'Tools', examples: [
      { title: 'Calling Tools', link: '/examples/tools/calling_tools' },
      { title: 'Dynamic Tools', link: '/examples/tools/dynamic_tools' },
      { title: 'Tools with Workflows', link: '/examples/tools/tools_workflows' }
    ]},
    { id: 'memory', name: 'Memory', examples: [
      { title: 'Memory with LibSQL', link: '/examples/memory/memory_with_libsql' },
      { title: 'Memory with PostgreSQL', link: '/examples/memory/memory_with_postgresql' },
      { title: 'Memory with Upstash', link: '/examples/memory/memory_with_upstash' },
      { title: 'Memory with MemO', link: '/examples/memory/memory_with_memo' },
      { title: 'Streaming Working Memory', link: '/examples/memory/memory_with_memcache' },
      { title: 'Streaming Working Memory (Advanced)', link: '/examples/memory/memory_without_memcache_advanced' },
      { title: 'Streaming Structured Working Memory', link: '/examples/memory/structured_memory_streaming' },
      { title: 'Memory Processors', link: '/examples/memory/memory_processors' },
      { title: 'AI SDK useChat Hook', link: '/examples/memory/ai_sdk_usechat_hook' }
    ]},
    { id: 'rag', name: 'RAG', examples: [
      { title: 'Chunk Text', link: '/examples/rag/chunk_text' },
      { title: 'Chunk Markdown', link: '/examples/rag/chunk_markdown' },
      { title: 'Chunk HTML', link: '/examples/rag/chunk_html' },
      { title: 'Chunk JSON', link: '/examples/rag/chunk_json' },
      { title: 'Adjust Chunk Size', link: '/examples/rag/chunk_size' },
      { title: 'Adjust Chunk Delimiters', link: '/examples/rag/chunk_delimiters' },

      { title: 'Embed Text Chunk', link: '/examples/rag/embed_text_chunk' },
      { title: 'Embed Chunk Array', link: '/examples/rag/embed_chunk_array' },
      { title: 'Embed Text with Cohere', link: '/examples/rag/embed_text_with_cohere' },
      { title: 'Metadata Extraction', link: '/examples/rag/metadata_extraction' },
      { title: 'Upsert Embeddings', link: '/examples/rag/upsert_embeddings' },
      { title: 'Hybrid Vector Search', link: '/examples/rag/hybrid_vector_search' },
      
      { title: 'Retrieve Results', link: '/examples/rag/retrieve_results' },
      { title: 'Re-ranking Results', link: '/examples/rag/re_ranking_results' },
      { title: 'Re-ranking Results with Tools', link: '/examples/rag/re_ranking_with_tools' },
      { title: 'Re-ranking Results with Cohere', link: '/examples/rag/re_ranking_with_cohere' },
      { title: 'Re-ranking Results with ZeroEntropy', link: '/examples/rag/re_ranking_with_zeroentropy' },
      { title: 'Using the Vector Query Tool', link: '/examples/rag/using_vector_query_tool' },
      { title: 'Optimizing Information Density', link: '/examples/rag/optimizing_information_density' },
      { title: 'Metadata Filtering', link: '/examples/rag/metadata_filtering' },
      { title: 'Chain of Thought Prompting', link: '/examples/rag/chain_of_thought_prompting' }
    ]},
    { id: 'evals', name: 'Evals', examples: [
      { title: 'Answer Relevancy', link: '/examples/evals/answer_relevancy' },
      { title: 'Bias', link: '/examples/evals/bias' },
      { title: 'Completeness', link: '/examples/evals/completeness' },
      { title: 'Content Similarity', link: '/examples/evals/content_similarity' },
      { title: 'Context Position', link: '/examples/evals/context_position' },
      { title: 'Context Precision', link: '/examples/evals/context_precision' },
      { title: 'Context Relevancy', link: '/examples/evals/context_relevancy' },
      { title: 'Contextual Recall', link: '/examples/evals/contextual_recall' },
      { title: 'Faithfulness', link: '/examples/evals/faithfulness' },
      { title: 'Hallucination', link: '/examples/evals/hallucination' },
      { title: 'Keyword Coverage', link: '/examples/evals/keyword_coverage' },
      { title: 'Prompt Alignment', link: '/examples/evals/prompt_alignment' },
      { title: 'Summarization', link: '/examples/evals/summarization' },
      { title: 'Textual Difference', link: '/examples/evals/textual_difference' },
      { title: 'Tone Consistency', link: '/examples/evals/tone_consistency' },
      { title: 'Toxicity', link: '/examples/evals/toxicity' },
      { title: 'LLM as a Judge', link: '/examples/evals/llm_as_judge' },
      { title: 'Native JavaScript', link: '/examples/evals/native_javascript' }
    ]},
    { id: 'voice', name: 'Voice', examples: [
      { title: 'Text-to-Speech', link: '/examples/voice/tts' },
      { title: 'Speech-to-Text', link: '/examples/voice/stt' },
      { title: 'Turn Taking', link: '/examples/voice/turn_taking' },
      { title: 'Speech-to-Speech', link: '/examples/voice/sts' },
    ]}
  ];

  const currentSection = sections.find(s => s.id === activeSection);

  return (
    <div>
      <p>The Examples section is a short list of example projects demonstrating basic AI engineering with Timbal, including text generation, structured output, streaming responses, retrieval-augmented generation (RAG), and voice.</p>
      
      {/* Filter/Category Tags */}
      <div style={{display: 'flex', gap: '0.5rem', marginBottom: '2rem', marginTop: '3rem', flexWrap: 'wrap'}}>
        {sections.map(section => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id)}
            style={{
              padding: '0.5rem 1rem',
              borderRadius: '1rem',
              border: '1px solid var(--ifm-color-emphasis-300)',
              background: activeSection === section.id ? '#ffffff' : 'transparent',
              color: activeSection === section.id ? '#000000' : 'var(--ifm-color-emphasis-700)',
              cursor: 'pointer',
              fontSize: '0.875rem',
              transition: 'all 0.2s ease',
              fontWeight: activeSection === section.id ? '600' : '400',
              boxShadow: activeSection === section.id ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
            }}
          >
            {section.name}
          </button>
        ))}
      </div>

      {/* Example Cards Grid */}
      <div style={{display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem'}}>
        {currentSection.examples.map((example, index) => (
          <Link 
            key={index}
            to={example.link}
            style={{
              border: '1px solid var(--ifm-color-emphasis-300)',
              borderRadius: '0.5rem',
              padding: '0.75rem',
              background: 'var(--ifm-background-color)',
              transition: 'all 0.2s ease',
              textDecoration: 'none',
              display: 'block'
            }}
            onMouseEnter={(e) => {
              e.target.style.borderColor = 'var(--ifm-color-primary)';
              e.target.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
            }}
            onMouseLeave={(e) => {
              e.target.style.borderColor = 'var(--ifm-color-emphasis-300)';
              e.target.style.boxShadow = 'none';
            }}
          >
            <h3 style={{
              margin: '0',
              fontSize: '1rem',
              color: 'var(--ifm-color-emphasis-900)',
              textAlign: 'center',
              fontWeight: '400'
            }}>
              {example.title}
            </h3>
          </Link>
        ))}
      </div>
    </div>
  );
}
