import React, { useMemo, useState, useEffect } from 'react';
import { Marked } from 'marked';

// Configure marked to be safe and clean
const markedInstance = new Marked({
  gfm: true,
  breaks: true,
});

const KEYWORDS = [
  'Searching...',
  'Analysing...',
  'Drafting...'
];

export default function StreamingAnswer({ text, timings, citations, onViewCitations, isStreaming }) {
  const [keywordIndex, setKeywordIndex] = useState(0);
  const [quoteExpanded, setQuoteExpanded] = useState(false);

  useEffect(() => {
    if (!text && isStreaming) {
      const interval = setInterval(() => {
        setKeywordIndex((prev) => (prev + 1) % KEYWORDS.length);
      }, 5000);
      return () => clearInterval(interval);
    }
  }, [text, isStreaming]);

  // Parse response text to identify first answer, status indicator, and final answer segments
  const parsedResponse = useMemo(() => {
    const result = {
      firstAnswerBody: '',
      firstAnswerQuote: '',
      refiningText: '',
      finalAnswerBody: '',
      finalAnswerQuote: '',
      hasFinalAnswer: false,
      hasRefining: false
    };

    if (!text) return result;

    const refiningIndex = text.indexOf('*Refining answer using additional retrieved contexts...*');
    const finalAnswerHeaderIndex = text.indexOf('# Synthesized Final Answer');
    const fallbackHeaderIndex = text.indexOf('# Synthesized Answer (Fallback)');
    
    const hasRefining = refiningIndex !== -1;
    const finalHeaderIndex = finalAnswerHeaderIndex !== -1 ? finalAnswerHeaderIndex : fallbackHeaderIndex;
    const hasFinalAnswer = finalHeaderIndex !== -1;
    
    result.hasRefining = hasRefining;
    result.hasFinalAnswer = hasFinalAnswer;

    // Helper to split a section into body and quote
    const splitBodyAndQuote = (sectionText) => {
      // Ordered H3 → H2 → H1 → bold — most specific first so that `###`
      // never accidentally matches inside a longer `#` search.
      // Plain-text triggers (no leading `#` or `**`) are intentionally
      // excluded: they match naturally-occurring prose and would incorrectly
      // split the answer body at innocent sentences.
      const triggers = [
        '### Verbatim Regulatory Quote',
        '### Verbatim Regulator Quote',
        '## Verbatim Regulatory Quote',
        '## Verbatim Regulator Quote',
        '# Verbatim Regulatory Quote',
        '# Verbatim Regulator Quote',
        '**Verbatim Regulatory Quote**',
        '**Verbatim Regulator Quote**',
        '# exact regulation quote',
        '# verbatim quote',
        '# exact quote',
      ];
      
      let splitIndex = -1;
      let matchedTrigger = '';
      
      for (const trigger of triggers) {
        const idx = sectionText.toLowerCase().indexOf(trigger.toLowerCase());
        if (idx !== -1) {
          splitIndex = idx;
          matchedTrigger = sectionText.substring(idx, idx + trigger.length);
          break;
        }
      }
      
      if (splitIndex === -1) {
        return { body: sectionText, quote: '' };
      }
      
      const body = sectionText.substring(0, splitIndex).trim();
      const quote = sectionText.substring(splitIndex + matchedTrigger.length).trim();
      return { body, quote };
    };

    if (!hasRefining && !hasFinalAnswer) {
      // Standard single pass: everything belongs to the first answer
      const split = splitBodyAndQuote(text);
      result.firstAnswerBody = split.body;
      result.firstAnswerQuote = split.quote;
    } else {
      // Map-Reduce flow: separate first answer, status indicator, and final answer
      let firstAnswerEndIndex = refiningIndex;
      if (firstAnswerEndIndex === -1) {
        firstAnswerEndIndex = finalHeaderIndex;
      }
      
      // Look for the preceding horizontal line separator '---'
      const separatorIdx = text.lastIndexOf('---', firstAnswerEndIndex);
      if (separatorIdx !== -1 && (firstAnswerEndIndex - separatorIdx) < 20) {
        firstAnswerEndIndex = separatorIdx;
      }
      
      const firstAnswerSection = text.substring(0, firstAnswerEndIndex).trim();
      const splitFirst = splitBodyAndQuote(firstAnswerSection);
      result.firstAnswerBody = splitFirst.body;
      result.firstAnswerQuote = splitFirst.quote;
      
      if (hasRefining) {
        result.refiningText = '*Refining answer using additional retrieved contexts...*';
      }
      
      if (hasFinalAnswer) {
        const headerLength = finalAnswerHeaderIndex !== -1 ? '# Synthesized Final Answer'.length : '# Synthesized Answer (Fallback)'.length;
        const finalSection = text.substring(finalHeaderIndex + headerLength).trim();
        const splitFinal = splitBodyAndQuote(finalSection);
        result.finalAnswerBody = splitFinal.body;
        result.finalAnswerQuote = splitFinal.quote;
      }
    }

    return result;
  }, [text]);

  const firstAnswerHtml = useMemo(() => {
    return { __html: markedInstance.parse(parsedResponse.firstAnswerBody || '') };
  }, [parsedResponse.firstAnswerBody]);

  const refiningHtml = useMemo(() => {
    return { __html: markedInstance.parse(parsedResponse.refiningText ? '---\n' + parsedResponse.refiningText : '') };
  }, [parsedResponse.refiningText]);

  const finalAnswerHtml = useMemo(() => {
    const header = parsedResponse.hasFinalAnswer ? '### Synthesized Final Answer\n' : '';
    return { __html: markedInstance.parse(header + parsedResponse.finalAnswerBody) };
  }, [parsedResponse.finalAnswerBody, parsedResponse.hasFinalAnswer]);

  const quoteHtml = useMemo(() => {
    const quoteText = parsedResponse.hasFinalAnswer ? parsedResponse.finalAnswerQuote : parsedResponse.firstAnswerQuote;
    return { __html: markedInstance.parse(quoteText || '') };
  }, [parsedResponse.hasFinalAnswer, parsedResponse.finalAnswerQuote, parsedResponse.firstAnswerQuote]);

  return (
    <div className="streaming-answer-container" style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '16px',
      padding: '20px',
      backgroundColor: 'var(--bg-card)',
      border: '1px solid var(--border-color)',
      borderRadius: '12px',
      marginBottom: '16px',
      animation: 'fadeIn 0.3s ease'
    }}>
      {/* Bot Icon and Title */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
        <span style={{ fontSize: '20px' }}>🤖</span>
        <span style={{ fontWeight: '600', fontSize: '14px', letterSpacing: '0.5px', textTransform: 'uppercase', color: 'var(--text-secondary)' }}>
          Regulatory Assistant
        </span>
        {isStreaming && (
          <>
            <span className="pulse-dot" style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: 'var(--accent-primary)',
              marginLeft: '6px'
            }} />
            {!text && (
              <span style={{
                fontSize: '11px',
                color: 'var(--text-muted)',
                marginLeft: '12px',
                fontWeight: 'normal',
                textTransform: 'none',
                animation: 'fadeIn 0.2s ease'
              }}>
                {KEYWORDS[keywordIndex]}
              </span>
            )}
          </>
        )}
      </div>

      {/* Main Answer Content / Simple Pulsing Indicator */}
      {!text && isStreaming ? (
        <div className="pipeline-loader">
          <div className="pipeline-pulse-dot"></div>
          <span className="pipeline-status-text">{KEYWORDS[keywordIndex]}</span>
        </div>
      ) : (
        <>
          {/* First Answer Body */}
          <div 
            className="markdown-content"
            dangerouslySetInnerHTML={firstAnswerHtml}
          />
          
          {/* Refining Status Indicator */}
          {parsedResponse.hasRefining && (
            <div 
              className="markdown-content refining-status"
              style={{ color: 'var(--text-muted, #64748b)', fontStyle: 'italic', margin: '12px 0' }}
              dangerouslySetInnerHTML={refiningHtml}
            />
          )}

          {/* Final Answer Body */}
          {parsedResponse.hasFinalAnswer && (
            <div 
              className="markdown-content final-answer"
              style={{ borderTop: '1px solid var(--border-color, #e2e8f0)', paddingTop: '16px', marginTop: '16px' }}
              dangerouslySetInnerHTML={finalAnswerHtml}
            />
          )}
          
          {/* Verbatim Quote Dropdown */}
          {((!parsedResponse.hasRefining && parsedResponse.firstAnswerQuote) || 
            (parsedResponse.hasFinalAnswer && parsedResponse.finalAnswerQuote)) && (
            <div style={{
              marginTop: '16px',
              border: '1px solid var(--border-color, #e2e8f0)',
              borderRadius: '8px',
              overflow: 'hidden',
              backgroundColor: 'var(--bg-hover, #f8fafc)',
              transition: 'all 0.2s ease'
            }}>
              <button
                onClick={() => setQuoteExpanded(!quoteExpanded)}
                style={{
                  width: '100%',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '10px 14px',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  fontWeight: '600',
                  fontSize: '13px',
                  color: 'var(--text-secondary, #475569)',
                  textAlign: 'left',
                  outline: 'none'
                }}
              >
                <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  📜 Verbatim Regulatory Quote Reference
                </span>
                <span style={{ 
                  fontSize: '10px', 
                  transform: quoteExpanded ? 'rotate(180deg)' : 'rotate(0deg)', 
                  transition: 'transform 0.2s ease',
                  color: 'var(--text-muted, #94a3b8)'
                }}>
                  ▼
                </span>
              </button>
              
              {quoteExpanded && (
                <div style={{
                  padding: '14px',
                  borderTop: '1px solid var(--border-color, #e2e8f0)',
                  fontSize: '13px',
                  lineHeight: '1.6',
                  color: 'var(--text-primary, #1e293b)',
                  backgroundColor: 'var(--bg-card, #ffffff)'
                }}>
                  <div 
                    className="markdown-content quote-section"
                    dangerouslySetInnerHTML={quoteHtml}
                  />
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Citations toggle and stage timings */}
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        borderTop: '1px solid var(--border-color)',
        paddingTop: '12px',
        marginTop: '8px',
        fontSize: '12px',
        color: 'var(--text-muted)'
      }}>
        <div style={{ display: 'flex', gap: '12px' }}>
          {citations && citations.length > 0 && (
            <button 
              onClick={onViewCitations}
              className="btn btn-secondary"
              style={{ padding: '6px 12px', fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px' }}
            >
              <span>📎</span> View Sources ({citations.length})
            </button>
          )}
        </div>

        {timings && Object.keys(timings).length > 0 && (
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
            {Object.entries(timings).map(([stage, time]) => (
              <span key={stage} style={{ 
                padding: '2px 6px', 
                backgroundColor: 'var(--bg-hover)', 
                borderRadius: '4px',
                fontFamily: 'var(--font-mono)',
                fontSize: '10px'
              }}>
                {stage}: {typeof time === 'number' ? `${time.toFixed(0)}ms` : time}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
