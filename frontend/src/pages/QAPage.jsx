import React, { useState, useEffect, useRef } from 'react';
import StreamingAnswer from '../components/StreamingAnswer';

export default function QAPage({ documents, setCitations, setRightPanelOpen }) {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [selectedDocFilter, setSelectedDocFilter] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  const messagesEndRef = useRef(null);

  // Auto scroll to bottom
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || isStreaming) {
      return;
    }

    const userMsg = {
      id: Date.now().toString(),
      role: 'user',
      text: query
    };

    const botMsgId = (Date.now() + 1).toString();
    const botMsgPlaceholder = {
      id: botMsgId,
      role: 'bot',
      text: '',
      timings: {},
      citations: [],
      isStreaming: true
    };

    setMessages((prev) => [...prev, userMsg, botMsgPlaceholder]);
    setQuery('');
    setIsStreaming(true);

    // Build URL with query params
    let url = `/api/qa?query=${encodeURIComponent(userMsg.text)}`;
    if (selectedDocFilter) {
      url += `&doc_filter=${encodeURIComponent(selectedDocFilter)}`;
    }

    const eventSource = new EventSource(url);

    eventSource.addEventListener('token', (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prev) => 
          prev.map((msg) => {
            if (msg.id === botMsgId) {
              return { ...msg, text: msg.text + data.token };
            }
            return msg;
          })
        );
      } catch (err) {
        console.error('Failed to parse token event:', err);
      }
    });

    eventSource.addEventListener('citations', (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prev) => 
          prev.map((msg) => {
            if (msg.id === botMsgId) {
              return { ...msg, citations: data };
            }
            return msg;
          })
        );
      } catch (err) {
        console.error('Failed to parse citations event:', err);
      }
    });

    eventSource.addEventListener('timings', (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prev) => 
          prev.map((msg) => {
            if (msg.id === botMsgId) {
              return { ...msg, timings: data };
            }
            return msg;
          })
        );
      } catch (err) {
        console.error('Failed to parse timings event:', err);
      }
    });

    eventSource.addEventListener('done', () => {
      setMessages((prev) => 
        prev.map((msg) => {
          if (msg.id === botMsgId) {
            // Update parent layout citations to show immediately when completed
            if (msg.citations && msg.citations.length > 0) {
              setCitations(msg.citations);
              setRightPanelOpen(true);
            }
            return { ...msg, isStreaming: false };
          }
          return msg;
        })
      );
      setIsStreaming(false);
      eventSource.close();
    });

    eventSource.addEventListener('error', (event) => {
      let errorMsg = 'An error occurred during query generation.';
      try {
        if (event.data) {
          const parsed = JSON.parse(event.data);
          if (parsed.error) {
            errorMsg = parsed.error;
          }
        }
      } catch (_) {}
      
      setMessages((prev) => 
        prev.map((msg) => {
          if (msg.id === botMsgId) {
            return { 
              ...msg, 
              text: msg.text + `\n\n❌ **Error**: ${errorMsg}`, 
              isStreaming: false 
            };
          }
          return msg;
        })
      );
      setIsStreaming(false);
      eventSource.close();
    });
  };

  const handleViewCitations = (msgCitations) => {
    setCitations(msgCitations);
    setRightPanelOpen(true);
  };

  return (
    <div className="page-container" style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '24px 32px' }}>
      {/* Scrollable conversation history */}
      <div style={{ flexGrow: 1, overflowY: 'auto', marginBottom: '24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {messages.length === 0 ? (
          <div style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            height: '100%',
            color: 'var(--text-secondary)',
            textAlign: 'center',
            gap: '16px',
            marginTop: '80px'
          }}>
            <span style={{ fontSize: '48px' }}>🤖</span>
            <div>
              <h2 style={{ marginBottom: '8px' }}>IFSCA Regulatory Assistant</h2>
              <p style={{ maxWidth: '480px', fontSize: '14px', color: 'var(--text-muted)' }}>
                Ask questions about IFSCA regulations, banking standards, fintech framework, or capital markets.
                Responses will stream in real time with precise source citations.
              </p>
            </div>
          </div>
        ) : (
          messages.map((msg) => {
            if (msg.role === 'user') {
              return (
                <div key={msg.id} style={{
                  alignSelf: 'flex-end',
                  backgroundColor: 'var(--bg-hover)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '16px 16px 2px 16px',
                  padding: '12px 20px',
                  maxWidth: '70%',
                  fontSize: '15px',
                  lineHeight: '1.5',
                  color: 'var(--text-primary)',
                  boxShadow: '0 2px 8px rgba(0, 0, 0, 0.15)',
                  animation: 'fadeIn 0.2s ease'
                }}>
                  {msg.text}
                </div>
              );
            } else {
              return (
                <StreamingAnswer
                  key={msg.id}
                  text={msg.text}
                  timings={msg.timings}
                  citations={msg.citations}
                  onViewCitations={() => handleViewCitations(msg.citations)}
                  isStreaming={msg.isStreaming}
                />
              );
            }
          })
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input container at bottom */}
      <form onSubmit={handleSubmit} style={{
        backgroundColor: 'transparent',
        padding: '0',
        display: 'flex',
        flexDirection: 'column',
        gap: '8px',
        marginTop: 'auto'
      }}>
        {/* Document scope row (minimal metadata look) */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', fontSize: '12px', color: 'var(--text-muted)', paddingLeft: '8px' }}>
          <span>Targeting Scope:</span>
          <select 
            value={selectedDocFilter}
            onChange={(e) => setSelectedDocFilter(e.target.value)}
            disabled={isStreaming}
            style={{
              backgroundColor: 'transparent',
              color: 'var(--text-secondary)',
              border: 'none',
              fontSize: '12px',
              fontWeight: '500',
              outline: 'none',
              cursor: 'pointer',
              textDecoration: 'underline'
            }}
          >
            <option value="">All Regulations</option>
            {documents.map((doc) => (
              <option key={doc.doc_id} value={doc.doc_id}>
                {doc.file_name.replace('.pdf', '')}
              </option>
            ))}
          </select>
        </div>

        {/* Input box row */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          backgroundColor: 'var(--bg-card)', 
          border: '1px solid var(--border-color)', 
          borderRadius: '24px', 
          padding: '4px 6px 4px 16px',
          boxShadow: '0 2px 12px rgba(0, 0, 0, 0.03)'
        }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={isStreaming ? 'Generating response...' : 'Ask a regulatory compliance question...'}
            disabled={isStreaming}
            style={{
              flexGrow: 1,
              backgroundColor: 'transparent',
              color: 'var(--text-primary)',
              border: 'none',
              fontSize: '14px',
              outline: 'none',
              height: '38px'
            }}
          />
          <button 
            type="submit"
            className="btn btn-primary"
            disabled={!query.trim() || isStreaming}
            style={{ 
              borderRadius: '20px', 
              padding: '8px 20px', 
              height: '36px',
              fontSize: '13px',
              flexShrink: 0
            }}
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
