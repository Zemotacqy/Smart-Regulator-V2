import React from 'react';

export default function SourcePanel({ citations, isOpen, onClose }) {
  return (
    <div className={`context-panel ${isOpen ? '' : 'collapsed'}`}>
      <div className="context-header">
        <div className="context-title">
          <span>📚</span> Source Citations ({citations.length})
        </div>
        <button className="close-panel-btn" onClick={onClose} aria-label="Close panel">
          ✕
        </button>
      </div>
      <div className="context-body">
        {citations.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '14px', textAlign: 'center', marginTop: '40px' }}>
            No citations for the active response. Ask a query to view sources.
          </div>
        ) : (
          citations.map((cit, idx) => (
            <div className="citation-card" key={cit.node_id || idx}>
              <div className="citation-meta">
                <span className="badge badge-doc">{cit.file_name || 'Document'}</span>
                {cit.breadcrumb && (
                  <span className="badge badge-breadcrumb">{cit.breadcrumb}</span>
                )}
              </div>
              {cit.title && <div className="citation-title">{cit.title}</div>}
              <div className="citation-text">{cit.text_content}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
