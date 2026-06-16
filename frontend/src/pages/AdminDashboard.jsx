import React, { useState, useEffect } from 'react';
import IngestionLog from '../components/IngestionLog';

export default function AdminDashboard() {
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState({
    doc_count: 0,
    node_count: 0,
    rel_count: 0,
    flagged_count: 0
  });
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const [isStreamingLogs, setIsStreamingLogs] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

  // Fetch document list and stats
  const fetchData = async () => {
    try {
      const [docsRes, statsRes] = await Promise.all([
        fetch('/api/admin/documents'),
        fetch('/api/admin/stats')
      ]);

      if (docsRes.ok) {
        const docsData = await docsRes.json();
        setDocuments(docsData);
      }
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
      }
    } catch (err) {
      console.error('Failed to fetch admin dashboard data:', err);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

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
        setUploadMessage('');
      } else {
        setUploadMessage('Only PDF files are supported.');
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setUploadMessage('');
    }
  };

  const handleUpload = async () => {
    if (!file || isUploading) {
      return;
    }

    setIsUploading(true);
    setUploadMessage('Uploading document and queueing ingestion pipeline...');
    setIsStreamingLogs(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/admin/ingest', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(errText || 'Failed to submit document.');
      }

      const resData = await response.json();
      setUploadMessage(`Document "${file.name}" queued successfully!`);
      setFile(null);
      
      // Refresh database data after some delay
      setTimeout(() => {
        fetchData();
      }, 5000);

    } catch (err) {
      console.error('Upload failed:', err);
      setUploadMessage(`Error: ${err.message}`);
      setIsStreamingLogs(false);
      setIsUploading(false);
    }
  };

  return (
    <div className="page-container">
      <h2 style={{ marginBottom: '24px' }}>Document Corpus Management</h2>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1.2fr 1fr',
        gap: '32px',
        height: '100%'
      }}>
        {/* Left Column: Corpus Document List */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '24px',
            flexGrow: 1
          }}>
            <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              📚 Corpus ({stats.doc_count} documents)
            </h3>
            
            {/* Needs Repair Alert */}
            {stats.flagged_count > 0 && (
              <div style={{
                backgroundColor: 'var(--color-needs-review-bg)',
                border: '1px solid var(--color-needs-review)',
                borderRadius: '8px',
                padding: '12px 16px',
                marginBottom: '16px',
                fontSize: '13px',
                color: 'var(--color-needs-review)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <span>⚠️ <strong>{stats.flagged_count} AST nodes need repair</strong> (failed classifier boundaries)</span>
                <span style={{ fontSize: '11px', textTransform: 'uppercase', fontWeight: 'bold' }}>operator attention required</span>
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px', maxHeight: '450px', overflowY: 'auto' }}>
              {documents.length === 0 ? (
                <div style={{ color: 'var(--text-muted)', fontSize: '14px', textAlign: 'center', padding: '40px 0' }}>
                  No documents found in database.
                </div>
              ) : (
                documents.map((doc) => (
                  <div key={doc.doc_id} style={{
                    backgroundColor: 'var(--bg-card)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '8px',
                    padding: '14px 18px',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '8px',
                    transition: 'border-color 0.2s'
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: '600', fontSize: '14px', color: 'var(--text-primary)' }}>
                        {doc.file_name}
                      </span>
                      <span className="badge badge-doc" style={{ fontSize: '10px' }}>
                        ✅ ACTIVE
                      </span>
                    </div>
                    <div style={{ display: 'flex', gap: '16px', fontSize: '11px', color: 'var(--text-muted)' }}>
                      <span>Type: {doc.doc_type || 'Unknown'}</span>
                      {doc.publish_date && <span>Published: {new Date(doc.publish_date).toLocaleDateString()}</span>}
                      {doc.ingested_at && <span>Ingested: {new Date(doc.ingested_at).toLocaleDateString()}</span>}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Database Info Cards */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gap: '16px'
          }}>
            {[
              { label: 'Total AST Nodes', value: stats.node_count, icon: '🌲' },
              { label: 'Relationships', value: stats.rel_count, icon: '🔗' },
              { label: 'Flagged Needs Repair', value: stats.flagged_count, icon: '⚠️', color: stats.flagged_count > 0 ? 'var(--color-needs-review)' : 'var(--text-primary)' }
            ].map((stat, idx) => (
              <div key={idx} style={{
                backgroundColor: 'var(--bg-surface)',
                border: '1px solid var(--border-color)',
                borderRadius: '10px',
                padding: '16px',
                textAlign: 'center',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px'
              }}>
                <div style={{ fontSize: '20px' }}>{stat.icon}</div>
                <div style={{ fontSize: '11px', color: 'var(--text-secondary)', fontWeight: '600', textTransform: 'uppercase' }}>
                  {stat.label}
                </div>
                <div style={{ fontSize: '22px', fontWeight: '700', color: stat.color || 'var(--text-primary)' }}>
                  {stat.value}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Column: Ingest Form, Live Console, Evaluation Metrics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Ingest New Document form */}
          <div style={{
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '24px'
          }}>
            <h3 style={{ fontSize: '15px', fontWeight: '600', marginBottom: '14px' }}>
              📥 Ingest New Regulation
            </h3>
            
            <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
              <div 
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                style={{
                  flexGrow: 1,
                  border: `2px dashed ${isDragOver ? 'var(--border-focus)' : 'var(--border-color)'}`,
                  borderRadius: '8px',
                  padding: '16px',
                  textAlign: 'center',
                  backgroundColor: isDragOver ? 'var(--bg-hover)' : 'var(--bg-card)',
                  cursor: 'pointer',
                  fontSize: '13px'
                }}
                onClick={() => document.getElementById('admin-file-input').click()}
              >
                <span>📄</span> {file ? file.name : 'Select or drop PDF'}
                <input 
                  id="admin-file-input"
                  type="file"
                  accept=".pdf"
                  style={{ display: 'none' }}
                  onChange={handleFileChange}
                />
              </div>

              <button 
                className="btn btn-primary"
                disabled={!file || isUploading}
                onClick={handleUpload}
                style={{ height: '48px', padding: '0 20px', flexShrink: 0 }}
              >
                Ingest
              </button>
            </div>
            {uploadMessage && (
              <div style={{ marginTop: '10px', fontSize: '12px', color: 'var(--text-secondary)' }}>
                {uploadMessage}
              </div>
            )}
          </div>

          {/* Live Ingestion Log component */}
          <div style={{
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '24px'
          }}>
            <IngestionLog isStreaming={isStreamingLogs} />
          </div>

          {/* Evaluation metrics card */}
          <div style={{
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            padding: '24px'
          }}>
            <h3 style={{ fontSize: '15px', fontWeight: '600', marginBottom: '16px' }}>
              📊 Evaluation Metrics (Judge: Mistral-Nemo)
            </h3>
            <div style={{
              display: 'grid',
              gridTemplateColumns: '1fr 1fr',
              gap: '12px',
              fontSize: '13px'
            }}>
              {[
                { name: 'Recall @ 5', value: '67.03%', desc: 'Dense + Sparse hybrid candidates' },
                { name: 'Recall @ 10', value: '73.63%', desc: 'Targeting legal cross-references' },
                { name: 'MRR', value: '0.5117', desc: 'Mean Reciprocal Rank' },
                { name: 'Faithfulness', value: '83.08%', desc: 'Context adherence index' },
                { name: 'Context Recall', value: '96.48%', desc: 'Relevant parts retrieved' },
                { name: 'Answer Relevance', value: '91.87%', desc: 'Goal alignment index' },
                { name: 'Citation Precision', value: '95.71%', desc: 'Verified accurate sources' }
              ].map((metric, idx) => (
                <div key={idx} style={{
                  backgroundColor: 'var(--bg-card)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '6px',
                  padding: '10px 14px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <div>
                    <div style={{ fontWeight: '600', color: 'var(--text-primary)' }}>{metric.name}</div>
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>{metric.desc}</div>
                  </div>
                  <div style={{ 
                    fontSize: '16px', 
                    fontWeight: '700', 
                    color: 'var(--accent-secondary)',
                    fontFamily: 'var(--font-mono)'
                  }}>
                    {metric.value}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
