import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, Navigate, useLocation } from 'react-router-dom';
import { PAGES } from './config/pages';
import QAPage from './pages/QAPage';
import CompliancePage from './pages/CompliancePage';
import AdminDashboard from './pages/AdminDashboard';
import SourcePanel from './components/SourcePanel';

function AppLayout() {
  const location = useLocation();
  const [documents, setDocuments] = useState([]);
  const [citations, setCitations] = useState([]);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);

  // Fetch documents list from admin API on mount
  useEffect(() => {
    const fetchDocs = async () => {
      try {
        const response = await fetch('/api/admin/documents');
        if (response.ok) {
          const data = await response.json();
          setDocuments(data);
        }
      } catch (err) {
        console.error('Failed to load documents list:', err);
      }
    };
    fetchDocs();
  }, []);

  return (
    <div className="app-container">
      {/* Left Sidebar Navigation */}
      <aside className={`sidebar ${leftPanelOpen ? '' : 'collapsed'}`}>
        <div className="brand-section">
          <span className="brand-logo">⚖️</span>
          <span className="brand-title">Smart Regulator</span>
        </div>

        <nav style={{ flexGrow: 1 }}>
          <ul className="nav-menu">
            {PAGES.map((page) => {
              const isActive = location.pathname === page.path;
              return (
                <li key={page.id} className="nav-item">
                  <Link 
                    to={page.path} 
                    className={`nav-link ${isActive ? 'active' : ''}`}
                  >
                    <span className="nav-icon">{page.icon}</span>
                    <span>{page.name}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="sidebar-footer">
          <div>IFSCA RAG Console v2.0</div>
          <div style={{ fontSize: '10px', marginTop: '4px' }}>GPU Accelerated Core</div>
        </div>
      </aside>

      {/* Main Shell: Main Content + Right Sidebar */}
      <div className="main-wrapper">
        <main className="content-panel">
          {/* Header Panel */}
          <header className="panel-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <button 
                onClick={() => setLeftPanelOpen(!leftPanelOpen)}
                style={{
                  background: 'none',
                  border: 'none',
                  fontSize: '18px',
                  cursor: 'pointer',
                  color: 'var(--text-secondary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  padding: '6px',
                  borderRadius: '6px',
                  transition: 'background-color 0.2s',
                  lineHeight: '1'
                }}
                onMouseOver={(e) => e.target.style.backgroundColor = 'var(--bg-hover)'}
                onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}
                title={leftPanelOpen ? "Collapse navigation" : "Expand navigation"}
              >
                ☰
              </button>
              <h1 className="panel-title" style={{ margin: 0, fontSize: '20px' }}>
                {PAGES.find(p => p.path === location.pathname)?.name || 'Dashboard'}
              </h1>
            </div>
            <div className="header-actions">
              <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>
                Database Connected: <strong>{documents.length} Docs</strong>
              </span>
            </div>
          </header>

          {/* Router Content Container */}
          <div style={{ flexGrow: 1, overflow: 'hidden', position: 'relative' }}>
            <Routes>
              <Route 
                path="/qa" 
                element={
                  <QAPage 
                    documents={documents} 
                    setCitations={setCitations} 
                    setRightPanelOpen={setRightPanelOpen} 
                  />
                } 
              />
              <Route 
                path="/compliance" 
                element={<CompliancePage documents={documents} />} 
              />
              <Route 
                path="/admin" 
                element={<AdminDashboard />} 
              />
              <Route path="*" element={<Navigate to="/qa" replace />} />
            </Routes>

            {/* Toggle context panel overlay button */}
            <button 
              className="toggle-context-btn"
              onClick={() => setRightPanelOpen(!rightPanelOpen)}
              title={rightPanelOpen ? 'Collapse source panel' : 'Expand source panel'}
            >
              {rightPanelOpen ? '👉' : '👈'}
            </button>
          </div>
        </main>

        {/* Right Source Citation Panel */}
        <SourcePanel 
          citations={citations} 
          isOpen={rightPanelOpen} 
          onClose={() => setRightPanelOpen(false)} 
        />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AppLayout />
    </BrowserRouter>
  );
}
