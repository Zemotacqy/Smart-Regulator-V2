import React, { useState } from 'react';

export default function ViolationCard({ audit, index }) {
  const [expanded, setExpanded] = useState(false);

  // Status mapping
  const statusConfig = {
    COMPLIANT: {
      label: 'COMPLIANT',
      color: 'var(--color-compliant)',
      bgColor: 'var(--color-compliant-bg)',
      icon: '✅'
    },
    'NON-COMPLIANT': {
      label: 'NON-COMPLIANT',
      color: 'var(--color-non-compliant)',
      bgColor: 'var(--color-non-compliant-bg)',
      icon: '❌'
    },
    'NEEDS REVIEW': {
      label: 'NEEDS REVIEW',
      color: 'var(--color-needs-review)',
      bgColor: 'var(--color-needs-review-bg)',
      icon: '⚠️'
    }
  };

  const status = audit.status || 'NEEDS REVIEW';
  const config = statusConfig[status] || statusConfig['NEEDS REVIEW'];

  return (
    <div 
      className="violation-card" 
      style={{
        backgroundColor: 'var(--bg-card)',
        border: `1px solid ${expanded ? 'var(--border-focus)' : 'var(--border-color)'}`,
        borderRadius: '12px',
        padding: '16px 20px',
        marginBottom: '12px',
        cursor: 'pointer',
        transition: 'all 0.2s ease',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px'
      }}
      onClick={() => setExpanded(!expanded)}
    >
      {/* Top Header line */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '18px' }}>{config.icon}</span>
          <span style={{ fontWeight: '600', fontSize: '15px', color: 'var(--text-primary)' }}>
            {audit.section_reference || `Rule check #${index + 1}`}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <span 
            className="badge" 
            style={{ 
              backgroundColor: config.bgColor, 
              color: config.color,
              border: `1px solid ${config.color}33`,
              fontSize: '11px',
              padding: '4px 10px'
            }}
          >
            {config.label}
          </span>
          <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            {expanded ? '▲' : '▼'}
          </span>
        </div>
      </div>

      {/* Main summary line */}
      <div style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
        {audit.explanation || 'No reasoning provided.'}
      </div>

      {/* Expanded Details */}
      {expanded && (
        <div 
          style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            gap: '16px', 
            marginTop: '8px', 
            paddingTop: '16px', 
            borderTop: '1px solid var(--border-color)',
            animation: 'fadeIn 0.2s ease'
          }}
          onClick={(e) => e.stopPropagation()} // Prevent collapse when clicking details
        >
          {/* Quoted Entity Text */}
          {audit.quoted_entity_text && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ fontSize: '11px', fontWeight: '600', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>
                🔍 Quoted Entity Text
              </div>
              <div style={{ 
                backgroundColor: 'var(--bg-app)', 
                border: '1px solid var(--border-color)', 
                borderRadius: '6px', 
                padding: '12px', 
                fontSize: '13px', 
                lineHeight: '1.5',
                color: 'var(--text-secondary)',
                fontStyle: 'italic'
              }}>
                "{audit.quoted_entity_text}"
              </div>
            </div>
          )}

          {/* Regulation Reference / Context */}
          {audit.regulation_text && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <div style={{ fontSize: '11px', fontWeight: '600', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.5px' }}>
                📖 Relevant Regulation Text
              </div>
              <div style={{ 
                backgroundColor: 'var(--bg-hover)', 
                border: '1px solid var(--border-color)', 
                borderRadius: '6px', 
                padding: '12px', 
                fontSize: '13px', 
                lineHeight: '1.5',
                color: 'var(--text-secondary)',
                fontFamily: 'var(--font-sans)'
              }}>
                {audit.regulation_text}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
