import React, { useEffect, useRef, useState } from 'react';

export default function IngestionLog({ isStreaming }) {
  const [logs, setLogs] = useState([]);
  const logEndRef = useRef(null);

  useEffect(() => {
    if (!isStreaming) {
      return;
    }

    const eventSource = new EventSource('/api/admin/ingest/logs?follow=true');

    eventSource.addEventListener('log', (event) => {
      try {
        const data = JSON.parse(event.data);
        setLogs((prev) => [...prev, data]);
      } catch (err) {
        console.error('Failed to parse log event:', err);
      }
    });

    eventSource.addEventListener('done', () => {
      eventSource.close();
    });

    eventSource.addEventListener('error', (err) => {
      console.error('EventSource failed:', err);
      eventSource.close();
    });

    return () => {
      eventSource.close();
    };
  }, [isStreaming]);

  // Auto scroll
  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  const getLevelColor = (level) => {
    switch (level?.toUpperCase()) {
      case 'ERROR':
        return '#f87171'; // light red
      case 'WARNING':
        return '#fbbf24'; // light amber
      case 'INFO':
        return '#60a5fa'; // light blue
      case 'DEBUG':
        return '#9ca3af'; // gray
      default:
        return 'var(--text-primary)';
    }
  };

  const formatLogMsg = (log) => {
    const { timestamp, level, event, ...extras } = log;
    const timeStr = timestamp ? new Date(timestamp).toLocaleTimeString() : '';
    const extraStr = Object.keys(extras).length > 0 
      ? ` | ${Object.entries(extras).map(([k, v]) => `${k}=${v}`).join(' ')}`
      : '';
    return {
      timeStr,
      level,
      msg: `${event}${extraStr}`
    };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ fontSize: '12px', fontWeight: '600', textTransform: 'uppercase', color: 'var(--text-muted)' }}>
          📟 Ingestion Pipeline Log Console
        </div>
        {isStreaming && (
          <span className="badge badge-doc" style={{ animation: 'pulse 1.5s infinite', fontSize: '10px' }}>
            Streaming Live
          </span>
        )}
      </div>

      <div 
        style={{
          backgroundColor: '#04060a',
          border: '1px solid var(--border-color)',
          borderRadius: '8px',
          padding: '16px',
          fontFamily: 'var(--font-mono)',
          fontSize: '12px',
          lineHeight: '1.6',
          height: '240px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
          boxShadow: 'inset 0 2px 8px rgba(0, 0, 0, 0.5)'
        }}
      >
        {logs.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
            {isStreaming ? 'Connecting to log stream...' : 'Log console inactive. Upload a document to view live logs.'}
          </div>
        ) : (
          logs.map((log, idx) => {
            const formatted = formatLogMsg(log);
            return (
              <div key={idx} style={{ wordBreak: 'break-all', whiteSpace: 'pre-wrap' }}>
                <span style={{ color: 'var(--text-muted)', marginRight: '8px' }}>[{formatted.timeStr}]</span>
                <span style={{ color: getLevelColor(formatted.level), fontWeight: '600', marginRight: '8px' }}>
                  {formatted.level}
                </span>
                <span style={{ color: '#e2e8f0' }}>{formatted.msg}</span>
              </div>
            );
          })
        )}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}
