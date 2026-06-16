import React, { useState } from 'react';
import ViolationCard from '../components/ViolationCard';

export default function CompliancePage({ documents }) {
  const [file, setFile] = useState(null);
  const [selectedDocFilter, setSelectedDocFilter] = useState('');
  const [isAuditing, setIsAuditing] = useState(false);
  const [audits, setAudits] = useState([]);
  const [errorMessage, setErrorMessage] = useState('');
  const [progressMsg, setProgressMsg] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.name.toLowerCase().endsWith('.pdf')) {
        setFile(droppedFile);
        setErrorMessage('');
      } else {
        setErrorMessage('Only PDF files are supported.');
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setErrorMessage('');
    }
  };

  const handleRunAudit = async () => {
    if (!file || isAuditing) {
      return;
    }

    setIsAuditing(true);
    setAudits([]);
    setErrorMessage('');
    setProgressMsg('Extracting text and chunking document...');

    const formData = new FormData();
    formData.append('file', file);
    if (selectedDocFilter) {
      formData.append('doc_filter', selectedDocFilter);
    }

    try {
      const response = await fetch('/api/compliance', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Failed to start compliance audit.');
      }

      // Read SSE stream
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || ''; // Keep the last incomplete block in the buffer

        for (const part of parts) {
          if (!part.trim()) continue;

          // Parse SSE lines
          const lines = part.split('\n');
          let eventType = '';
          let eventDataString = '';

          for (const line of lines) {
            if (line.startsWith('event:')) {
              eventType = line.replace('event:', '').trim();
            } else if (line.startsWith('data:')) {
              eventDataString = line.replace('data:', '').trim();
            }
          }

          if (eventType === 'audit' && eventDataString) {
            try {
              const auditResult = JSON.parse(eventDataString);
              setAudits((prev) => [...prev, auditResult]);
              setProgressMsg(`Audited Section: ${auditResult.section_reference || 'checking...'}`);
            } catch (err) {
              console.error('Failed to parse audit event:', err);
            }
          } else if (eventType === 'chunk_error' && eventDataString) {
            console.warn('Chunk level audit failure:', eventDataString);
          } else if (eventType === 'error' && eventDataString) {
            try {
              const errorObj = JSON.parse(eventDataString);
              setErrorMessage(errorObj.error || 'Pipeline error encountered.');
            } catch (_) {
              setErrorMessage('Pipeline error encountered.');
            }
            setIsAuditing(false);
          } else if (eventType === 'done') {
            setProgressMsg('Compliance audit complete.');
            setIsAuditing(false);
          }
        }
      }
    } catch (err) {
      console.error('Compliance checker failed:', err);
      setErrorMessage(err.message || 'Network error encountered during compliance check.');
      setIsAuditing(false);
    }
  };

  return (
    <div className="page-container">
      <h2 style={{ marginBottom: '24px' }}>Compliance Document Audit</h2>

      {/* Upload Box container */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '24px',
        marginBottom: '32px'
      }}>
        {/* Left Side File upload */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <div 
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            style={{
              border: `2px dashed ${isDragOver ? 'var(--border-focus)' : 'var(--border-color)'}`,
              borderRadius: '12px',
              padding: '40px 20px',
              textAlign: 'center',
              backgroundColor: isDragOver ? 'var(--bg-hover)' : 'var(--bg-card)',
              transition: 'all 0.2s ease',
              cursor: 'pointer',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '12px'
            }}
            onClick={() => document.getElementById('compliance-file-input').click()}
          >
            <span style={{ fontSize: '40px' }}>📄</span>
            <div>
              <p style={{ fontWeight: '600', fontSize: '14px', marginBottom: '4px' }}>
                {file ? file.name : 'Drag & drop PDF here'}
              </p>
              <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : 'or click to browse local files'}
              </p>
            </div>
            <input 
              id="compliance-file-input"
              type="file"
              accept=".pdf"
              style={{ display: 'none' }}
              onChange={handleFileChange}
            />
          </div>

          {errorMessage && (
            <div style={{ color: 'var(--color-non-compliant)', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>⚠️</span> {errorMessage}
            </div>
          )}
        </div>

        {/* Right Side Options & Actions */}
        <div style={{
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-color)',
          borderRadius: '12px',
          padding: '24px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          gap: '16px'
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            <label style={{ fontSize: '13px', fontWeight: '600', color: 'var(--text-secondary)' }}>
              Audit Against Regulation Scope
            </label>
            <select 
              value={selectedDocFilter}
              onChange={(e) => setSelectedDocFilter(e.target.value)}
              disabled={isAuditing}
              style={{
                backgroundColor: 'var(--bg-card)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '8px',
                padding: '12px',
                fontSize: '14px',
                outline: 'none',
                cursor: 'pointer'
              }}
            >
              <option value="">🔍 All Ingested Regulations</option>
              {documents.map((doc) => (
                <option key={doc.doc_id} value={doc.doc_id}>
                  {doc.file_name.replace('.pdf', '')}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <button 
              className="btn btn-primary"
              disabled={!file || isAuditing}
              onClick={handleRunAudit}
              style={{ width: '100%', padding: '12px 24px', height: '48px' }}
            >
              {isAuditing ? 'Auditing in Progress...' : '▶ Run Compliance Check'}
            </button>
            
            {isAuditing && (
              <div style={{ 
                fontSize: '12px', 
                color: 'var(--text-muted)', 
                display: 'flex', 
                alignItems: 'center', 
                gap: '8px',
                alignSelf: 'center'
              }}>
                <span className="pulse-dot" style={{ backgroundColor: 'var(--color-needs-review)' }} />
                <span>{progressMsg}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Results Container Section */}
      <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
        <div style={{
          borderBottom: '1px solid var(--border-color)',
          paddingBottom: '12px',
          marginBottom: '16px',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <span style={{ fontSize: '14px', fontWeight: '600', textTransform: 'uppercase', color: 'var(--text-secondary)', letterSpacing: '0.5px' }}>
            Live Audit Results ({audits.length})
          </span>
          {audits.length > 0 && (
            <div style={{ display: 'flex', gap: '12px', fontSize: '12px' }}>
              <span style={{ color: 'var(--color-compliant)' }}>
                ✅ Compliant: {audits.filter(a => a.status === 'COMPLIANT').length}
              </span>
              <span style={{ color: 'var(--color-non-compliant)' }}>
                ❌ Non-Compliant: {audits.filter(a => a.status === 'NON-COMPLIANT').length}
              </span>
              <span style={{ color: 'var(--color-needs-review)' }}>
                ⚠️ Needs Review: {audits.filter(a => a.status === 'NEEDS REVIEW' || !['COMPLIANT', 'NON-COMPLIANT'].includes(a.status)).length}
              </span>
            </div>
          )}
        </div>

        {audits.length === 0 ? (
          <div style={{ 
            color: 'var(--text-muted)', 
            fontSize: '14px', 
            textAlign: 'center', 
            padding: '80px 20px',
            backgroundColor: 'var(--bg-surface)',
            border: '1px dashed var(--border-color)',
            borderRadius: '8px'
          }}>
            No compliance audits run yet. Upload a document to scan it against regulation boundaries.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {audits.map((audit, idx) => (
              <ViolationCard key={idx} audit={audit} index={idx} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
