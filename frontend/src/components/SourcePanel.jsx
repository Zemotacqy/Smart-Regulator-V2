import React, { useState } from 'react';

export default function SourcePanel({ citations, isOpen, onClose }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [copiedId, setCopiedId] = useState(null);

  // Helper to determine the document type icon
  const getDocIcon = (fileName = '') => {
    const name = fileName.toLowerCase();
    if (name.includes('act')) return '🏛️';
    if (name.includes('regulation')) return '⚖️';
    if (name.includes('circular') || name.includes('framework') || name.includes('guidelines') || name.includes('direction')) return '📄';
    if (name.includes('sandbox')) return '🚀';
    return '📚';
  };

  // Helper to copy verbatim text to clipboard
  const handleCopy = (text, id) => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Helper to format breadcrumbs as a visual step-by-step trail, removing redundancy
  const renderBreadcrumbTrail = (breadcrumbStr, title) => {
    if (!breadcrumbStr) return null;
    let steps = breadcrumbStr.split(' > ').map(step => step.trim()).filter(Boolean);
    
    // Shorten long Authority name
    steps = steps.map(step => {
      if (step.includes("International Financial Services Centres Authority")) {
        return step.replace("International Financial Services Centres Authority", "IFSCA").replace("Regulations,", "Regs");
      }
      return step;
    });

    // If the last step is exactly the title of the card, omit it to avoid redundancy
    if (steps.length > 1 && title && steps[steps.length - 1].toLowerCase() === title.toLowerCase()) {
      steps = steps.slice(0, -1);
    }
    
    return (
      <div className="breadcrumb-trail">
        {steps.map((step, idx) => (
          <React.Fragment key={idx}>
            {idx > 0 && <span className="breadcrumb-separator">➔</span>}
            <span className="breadcrumb-step" title={step}>
              {step}
            </span>
          </React.Fragment>
        ))}
      </div>
    );
  };

  // Helper to extract a friendly title if missing
  const getCleanTitle = (cit) => {
    if (cit.title && cit.title.trim()) return cit.title;
    if (cit.breadcrumb) {
      const steps = cit.breadcrumb.split(' > ').map(s => s.trim()).filter(Boolean);
      if (steps.length > 0) {
        return steps[steps.length - 1];
      }
    }
    return 'Regulatory Provision';
  };

  // Filter citations based on search query
  const filteredCitations = (citations || []).filter(cit => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    const matchesFile = (cit.file_name || '').toLowerCase().includes(query);
    const matchesBreadcrumb = (cit.breadcrumb || '').toLowerCase().includes(query);
    const matchesTitle = (cit.title || '').toLowerCase().includes(query);
    const matchesContent = (cit.text_content || '').toLowerCase().includes(query);
    return matchesFile || matchesBreadcrumb || matchesTitle || matchesContent;
  });

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

      {/* Search Input Bar */}
      {citations && citations.length > 0 && (
        <div className="citation-search-container">
          <input
            type="text"
            className="citation-search-input"
            placeholder="Search citations by keyword, section..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      )}

      <div className="context-body">
        {citations.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '14px', textAlign: 'center', marginTop: '40px' }}>
            No citations for the active response. Ask a query to view sources.
          </div>
        ) : filteredCitations.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: '14px', textAlign: 'center', marginTop: '40px' }}>
            No citations match "{searchQuery}"
          </div>
        ) : (
          filteredCitations.map((cit, idx) => {
            const cardTitle = getCleanTitle(cit);
            const cardId = cit.node_id || idx;
            return (
              <div className="citation-card" key={cardId}>
                {/* Card Header: Section title and document name */}
                <div className="citation-card-top">
                  <div className="citation-title" title={cardTitle}>{cardTitle}</div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span className="badge badge-doc" title={cit.file_name}>
                      {getDocIcon(cit.file_name)} <span className="badge-text">{cit.file_name || 'Document'}</span>
                    </span>
                    <button 
                      className="copy-citation-btn"
                      onClick={() => handleCopy(cit.text_content, cardId)}
                      title="Copy verbatim text to clipboard"
                    >
                      {copiedId === cardId ? '✓ Copied' : '📋 Copy'}
                    </button>
                  </div>
                </div>

                {/* Card Body: Actual Verbatim Text (Always visible, scrollable if long) */}
                <div className="citation-text-container">
                  <div className="citation-text">{cit.text_content}</div>
                </div>

                {/* Card Footer: Breadcrumb pathway (Where to find it / Locator) */}
                {cit.breadcrumb && (
                  <div className="citation-card-footer">
                    <span className="find-label">📍 Section Locator:</span>
                    {renderBreadcrumbTrail(cit.breadcrumb, cardTitle)}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
