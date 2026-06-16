import React, { useMemo } from 'react';
import { Marked } from 'marked';

// Configure marked to be safe and clean
const markedInstance = new Marked({
  gfm: true,
  breaks: true,
});

export default function StreamingAnswer({ text, timings, citations, onViewCitations, isStreaming }) {
  const htmlContent = useMemo(() => {
    try {
      return { __html: markedInstance.parse(text || '') };
    } catch (e) {
      console.error('Failed to parse markdown:', e);
      return { __html: text || '' };
    }
  }, [text]);

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
          <span className="pulse-dot" style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            backgroundColor: 'var(--accent-primary)',
            marginLeft: '6px'
          }} />
        )}
      </div>

      {/* Main Answer Content */}
      <div 
        className="markdown-content"
        dangerouslySetInnerHTML={htmlContent}
      />

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
