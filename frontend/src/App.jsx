import { useState, useEffect, useRef } from 'react'
import {
  MessageSquare,
  Search,
  FileText,
  UploadCloud,
  Trash2,
  Zap,
  Check,
  Copy,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  AlertCircle,
  Sparkles,
  Database,
  Cpu,
  Send,
  Layers,
  Sliders,
  X,
  FolderOpen,
  ArrowRight,
  Info
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './App.css'

export default function App() {
  // Navigation & View
  const [activeTab, setActiveTab] = useState('chat') // 'chat' | 'inspector' | 'knowledge'
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // System & Health State
  const [backendHealth, setBackendHealth] = useState({
    status: 'connecting',
    llm_provider: 'ollama',
    llm_model: 'llama3',
    total_documents: 0,
    embedding_provider: 'sentence-transformers'
  })
  const [isRefreshingStats, setIsRefreshingStats] = useState(false)

  // RAG Configuration
  const [topK, setTopK] = useState(5)
  const [temperature, setTemperature] = useState(0.2)

  // Chat State
  const [messages, setMessages] = useState([])
  const [inputQuery, setInputQuery] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [expandedCitations, setExpandedCitations] = useState({})
  const [copiedId, setCopiedId] = useState(null)
  const chatBottomRef = useRef(null)

  // Documents & Upload State
  const [documentsList, setDocumentsList] = useState([])
  const [isUploading, setIsUploading] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)

  // Vector Inspector State
  const [inspectorQuery, setInspectorQuery] = useState('')
  const [inspectorResults, setInspectorResults] = useState([])
  const [isSearchingVector, setIsSearchingVector] = useState(false)

  // Toast Notifications
  const [toast, setToast] = useState(null)

  const showToast = (message, type = 'info') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 4000)
  }

  // Fetch Health & Documents on mount
  const fetchStatus = async () => {
    try {
      setIsRefreshingStats(true)
      const res = await fetch('/api/stats')
      if (res.ok) {
        const data = await res.json()
        setBackendHealth({
          status: 'ready',
          llm_provider: data.llm_provider || 'ollama',
          llm_model: data.llm_model || 'llama3',
          total_documents: data.total_documents || 0,
          embedding_provider: data.embedding_provider || 'sentence-transformers',
          chunk_size: data.chunk_size,
          chunk_overlap: data.chunk_overlap
        })
      } else {
        setBackendHealth(prev => ({ ...prev, status: 'error' }))
      }
    } catch {
      setBackendHealth(prev => ({ ...prev, status: 'offline' }))
    } finally {
      setIsRefreshingStats(false)
    }
  }

  const fetchDocuments = async () => {
    try {
      const res = await fetch('/api/documents')
      if (res.ok) {
        const data = await res.json()
        setDocumentsList(data.files || [])
      }
    } catch (err) {
      console.error('Failed to fetch documents', err)
    }
  }

  useEffect(() => {
    fetchStatus()
    fetchDocuments()
    const interval = setInterval(fetchStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (activeTab === 'chat') {
      chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, isGenerating, activeTab])

  // Send Chat Query
  const handleSendMessage = async (textToSend) => {
    const query = (textToSend || inputQuery).trim()
    if (!query || isGenerating) return

    const userMessageId = 'user_' + Date.now()
    const assistantMessageId = 'ai_' + (Date.now() + 1)

    const userMessage = {
      id: userMessageId,
      role: 'user',
      content: query,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    setMessages(prev => [...prev, userMessage])
    setInputQuery('')
    setIsGenerating(true)

    // Add placeholder assistant message
    setMessages(prev => [
      ...prev,
      {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        isStreaming: true,
        sources: [],
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ])

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          top_k: topK,
          temperature: temperature
        })
      })

      if (!response.ok) {
        throw new Error(`API returned ${response.status}`)
      }

      const data = await response.json()

      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: data.answer || 'No response generated.',
                sources: data.sources || [],
                model: data.model,
                latency: data.latency_seconds,
                error: data.error,
                isStreaming: false
              }
            : msg
        )
      )
    } catch (err) {
      setMessages(prev =>
        prev.map(msg =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: `Error connecting to RAG backend: ${err.message}. Please ensure the FastAPI server and Ollama are running.`,
                isStreaming: false
              }
            : msg
        )
      )
      showToast('Failed to connect to backend', 'error')
    } finally {
      setIsGenerating(false)
    }
  }

  // File Upload Handler
  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return
    const file = files[0]

    const validExtensions = ['.pdf', '.docx', '.doc', '.txt']
    const hasValidExt = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))
    if (!hasValidExt) {
      showToast(`Unsupported format. Supported: ${validExtensions.join(', ')}`, 'error')
      return
    }

    setIsUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })

      if (res.ok) {
        const result = await res.json()
        showToast(`Indexed "${file.name}" (${result.chunks_created || 0} chunks added)`, 'success')
        fetchStatus()
        fetchDocuments()
      } else {
        const errorData = await res.json()
        showToast(errorData.detail || 'Upload failed', 'error')
      }
    } catch (err) {
      showToast(`Upload error: ${err.message}`, 'error')
    } finally {
      setIsUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  // Clear Vector Database
  const handleClearDatabase = async () => {
    if (!window.confirm('Are you sure you want to clear the entire vector database? This will delete all indexed document chunks.')) {
      return
    }
    try {
      const res = await fetch('/api/clear', { method: 'POST' })
      if (res.ok) {
        showToast('Vector store wiped clean.', 'success')
        fetchStatus()
        fetchDocuments()
        setMessages([])
      }
    } catch (err) {
      showToast(`Failed to clear: ${err.message}`, 'error')
    }
  }

  // Semantic Vector Search (Inspector)
  const handleVectorSearch = async (e) => {
    if (e) e.preventDefault()
    if (!inspectorQuery.trim() || isSearchingVector) return

    setIsSearchingVector(true)
    try {
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: inspectorQuery, top_k: topK })
      })
      if (res.ok) {
        const data = await res.json()
        setInspectorResults(data.results || [])
      } else {
        showToast('Vector search failed', 'error')
      }
    } catch (err) {
      showToast(`Search error: ${err.message}`, 'error')
    } finally {
      setIsSearchingVector(false)
    }
  }

  // Copy Message to Clipboard
  const handleCopy = (text, id) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const toggleCitations = (msgId) => {
    setExpandedCitations(prev => ({
      ...prev,
      [msgId]: !prev[msgId]
    }))
  }

  // Suggested Prompts
  const promptStarters = [
    {
      title: 'Summarize Document Findings',
      prompt: 'Provide a concise, high-level executive summary of the uploaded document, highlighting its primary findings and conclusions.',
      icon: <Sparkles size={16} />
    },
    {
      title: 'Analyze Strategic Initiatives',
      prompt: 'What strategic recommendations, initiatives, or action plans are outlined in this document?',
      icon: <ArrowRight size={16} />
    },
    {
      title: 'Extract Critical Metrics',
      prompt: 'Extract all key statistics, numerical data, percentages, and financial figures discussed in the text.',
      icon: <Database size={16} />
    },
    {
      title: 'Risk & Opportunity Assessment',
      prompt: 'What main risks, challenges, and opportunities are identified in these documents?',
      icon: <Info size={16} />
    }
  ]

  return (
    <div className="app-container">
      {/* Ambient background glows */}
      <div className="ambient-glow-1"></div>
      <div className="ambient-glow-2"></div>

      {/* Top Header */}
      <header className="top-header">
        <div className="brand-section">
          <div className="brand-icon">
            <Cpu size={22} />
          </div>
          <div className="brand-text">
            <h1>
              NexusRAG
              <span className="brand-badge">Neural RAG v2</span>
            </h1>
            <p className="brand-subtitle">Retrieval Augmented Generation & Knowledge Engine</p>
          </div>
        </div>

        {/* View Switcher Navigation */}
        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            <MessageSquare size={16} />
            <span>AI Chat</span>
          </button>
          <button
            className={`tab-btn ${activeTab === 'inspector' ? 'active' : ''}`}
            onClick={() => setActiveTab('inspector')}
          >
            <Search size={16} />
            <span>Vector Inspector</span>
          </button>
          <button
            className={`tab-btn ${activeTab === 'knowledge' ? 'active' : ''}`}
            onClick={() => setActiveTab('knowledge')}
          >
            <FolderOpen size={16} />
            <span>Knowledge Base ({documentsList.length})</span>
          </button>
        </nav>

        {/* Status Indicators & Drawer Toggle */}
        <div className="header-actions">
          <div className="status-pill" title="Model & Database Connection Status">
            <div className={`pulse-dot ${backendHealth.status === 'ready' ? '' : 'warning'}`}></div>
            <span>
              {backendHealth.status === 'ready'
                ? `${backendHealth.llm_model} • ${backendHealth.total_documents} chunks`
                : backendHealth.status === 'connecting'
                ? 'Connecting...'
                : 'Backend Offline'}
            </span>
          </div>

          <button
            className={`btn-icon-toggle ${sidebarOpen ? 'active' : ''}`}
            onClick={() => setSidebarOpen(!sidebarOpen)}
            title="Toggle RAG Tuning & Upload Sidebar"
          >
            <Sliders size={18} />
          </button>
        </div>
      </header>

      {/* Main Layout Area */}
      <div className="main-layout">
        {/* Sidebar Controls & Ingestion Drawer */}
        <aside className={`sidebar-panel ${sidebarOpen ? '' : 'collapsed'}`}>
          <div className="sidebar-header">
            <span className="sidebar-title">
              <Sliders size={17} />
              RAG Control Panel
            </span>
            <button
              className="meta-action-btn"
              onClick={fetchStatus}
              title="Refresh Stats"
            >
              <RefreshCw size={14} className={isRefreshingStats ? 'animate-spin' : ''} />
            </button>
          </div>

          {/* Quick Upload Dropzone */}
          <div className="sidebar-section">
            <div className="sidebar-label">
              <span>Ingest Documents</span>
              <span style={{ color: 'var(--accent-cyan)' }}>PDF, DOCX, TXT</span>
            </div>

            <div
              className={`dropzone-box ${isDragOver ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setIsDragOver(false)
                handleFileUpload(e.dataTransfer.files)
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: 'none' }}
                accept=".pdf,.docx,.doc,.txt"
                onChange={(e) => handleFileUpload(e.target.files)}
              />
              <div className="dropzone-icon">
                <UploadCloud size={20} />
              </div>
              <p className="dropzone-title">
                {isUploading ? 'Chunking & Embedding...' : 'Drop file to Index'}
              </p>
              <p className="dropzone-sub">Click to browse or drag file</p>
            </div>
          </div>

          {/* Hyperparameter Sliders */}
          <div className="sidebar-section">
            <div className="sidebar-label">
              <span>Retrieval Parameters</span>
            </div>

            <div className="control-group">
              <div className="control-header">
                <span>Top-K Chunks</span>
                <span className="control-val">{topK}</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="range-slider"
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                Chunks retrieved from ChromaDB
              </span>
            </div>

            <div className="control-group" style={{ marginTop: '1.2rem' }}>
              <div className="control-header">
                <span>LLM Temperature</span>
                <span className="control-val">{temperature}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                className="range-slider"
              />
              <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>
                Lower = strictly factual, Higher = creative
              </span>
            </div>
          </div>

          {/* Engine Info */}
          <div className="sidebar-section">
            <div className="sidebar-label">
              <span>Active Architecture</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span>LLM Engine:</span>
                <strong style={{ color: '#fff' }}>{backendHealth.llm_model} ({backendHealth.llm_provider})</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span>Embedder:</span>
                <strong style={{ color: '#fff' }}>all-MiniLM-L6-v2 (384d)</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span>Vector Store:</span>
                <strong style={{ color: '#fff' }}>ChromaDB</strong>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-muted)' }}>
                <span>Total Chunks:</span>
                <strong style={{ color: 'var(--accent-emerald)' }}>{backendHealth.total_documents}</strong>
              </div>
            </div>
          </div>

          {/* Wipe Storage */}
          <div className="sidebar-section" style={{ marginTop: 'auto' }}>
            <button
              onClick={handleClearDatabase}
              style={{
                width: '100%',
                padding: '8px 12px',
                background: 'rgba(244, 63, 94, 0.1)',
                border: '1px solid rgba(244, 63, 94, 0.25)',
                color: 'var(--accent-rose)',
                borderRadius: '8px',
                fontSize: '0.8rem',
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem'
              }}
            >
              <Trash2 size={14} />
              Reset Vector Store
            </button>
          </div>
        </aside>

        {/* ─── TAB 1: AI NEURAL CHAT ─── */}
        {activeTab === 'chat' && (
          <div className="view-container">
            <div className="chat-scroll-area">
              {messages.length === 0 ? (
                /* Welcome Hero Screen */
                <div className="welcome-hero">
                  <div className="hero-glow-badge">
                    <Sparkles size={14} />
                    <span>Neural RAG Assistant Online</span>
                  </div>
                  <h2 className="hero-title">
                    Ask anything about your documents
                  </h2>
                  <p className="hero-description">
                    Powered by local vector search and LLM synthesis. Every answer is grounded in your ingested data with full source transparency and citations.
                  </p>

                  <div className="starters-grid">
                    {promptStarters.map((starter, idx) => (
                      <div
                        key={idx}
                        className="starter-card"
                        onClick={() => handleSendMessage(starter.prompt)}
                      >
                        <div className="starter-card-icon">{starter.icon}</div>
                        <div className="starter-card-title">{starter.title}</div>
                        <div className="starter-card-sub">{starter.prompt}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                /* Chat Messages */
                messages.map((msg) => (
                  <div key={msg.id} className={`message-row ${msg.role}`}>
                    <div className={`avatar-badge ${msg.role === 'assistant' ? 'ai' : 'user'}`}>
                      {msg.role === 'assistant' ? <Cpu size={18} /> : 'U'}
                    </div>

                    <div className="message-bubble-wrap">
                      <div className="message-bubble">
                        {msg.isStreaming ? (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', color: 'var(--text-muted)' }}>
                            <div className="pulse-dot"></div>
                            <span>Synthesizing response from retrieved chunks...</span>
                          </div>
                        ) : msg.role === 'assistant' ? (
                          <div className="markdown-body">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {msg.content}
                            </ReactMarkdown>
                          </div>
                        ) : (
                          <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                        )}

                        {/* Citations Section */}
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="citations-container">
                            <button
                              className="citation-toggle-btn"
                              onClick={() => toggleCitations(msg.id)}
                            >
                              <Layers size={13} />
                              <span>
                                {expandedCitations[msg.id] ? 'Hide' : 'Show'} {msg.sources.length} Cited Sources
                              </span>
                              {expandedCitations[msg.id] ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                            </button>

                            {expandedCitations[msg.id] && (
                              <div className="citations-list">
                                {msg.sources.map((src, i) => (
                                  <div key={i} className="citation-card">
                                    <div className="citation-top">
                                      <div className="citation-source-tag">
                                        <FileText size={13} color="var(--accent-primary)" />
                                        <span>{src.filename}</span>
                                      </div>
                                      <div className="citation-score-badge">
                                        {Math.round(src.similarity_score * 100)}% Match
                                      </div>
                                    </div>
                                    <div className="citation-snippet">
                                      {src.content}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Message Meta Info */}
                      <div className="message-meta-bar">
                        <span>{msg.timestamp}</span>
                        {msg.latency && (
                          <>
                            <span>•</span>
                            <span style={{ color: 'var(--accent-cyan)' }}>
                              ⚡ {msg.latency}s
                            </span>
                          </>
                        )}
                        {msg.role === 'assistant' && !msg.isStreaming && (
                          <>
                            <span>•</span>
                            <button
                              className="meta-action-btn"
                              onClick={() => handleCopy(msg.content, msg.id)}
                            >
                              {copiedId === msg.id ? (
                                <>
                                  <Check size={12} color="var(--accent-emerald)" />
                                  <span style={{ color: 'var(--accent-emerald)' }}>Copied</span>
                                </>
                              ) : (
                                <>
                                  <Copy size={12} />
                                  <span>Copy</span>
                                </>
                              )}
                            </button>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
              <div ref={chatBottomRef} />
            </div>

            {/* Floating Input Dock */}
            <div className="input-dock">
              <div className="input-capsule">
                <textarea
                  className="input-textarea"
                  placeholder="Ask a question about your documents (e.g. 'What are the main findings in the report?')..."
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      handleSendMessage()
                    }
                  }}
                  rows={1}
                />

                <div className="input-toolbar">
                  <div className="input-left-hints">
                    <span>Press <kbd className="kbd-hint">Enter</kbd> to ask</span>
                    <span>•</span>
                    <span><kbd className="kbd-hint">Shift+Enter</kbd> for new line</span>
                  </div>

                  <button
                    className="btn-send"
                    onClick={() => handleSendMessage()}
                    disabled={!inputQuery.trim() || isGenerating}
                  >
                    <span>Ask AI</span>
                    <Send size={15} />
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ─── TAB 2: VECTOR INSPECTOR ─── */}
        {activeTab === 'inspector' && (
          <div className="inspector-view">
            <div className="inspector-header">
              <h2 className="inspector-title">
                <Search size={24} color="var(--accent-cyan)" />
                Vector Similarity Inspector
              </h2>
              <p className="inspector-sub">
                Test raw semantic search directly against ChromaDB embeddings. Inspect how your questions match indexed document chunks without LLM generation.
              </p>
            </div>

            <form onSubmit={handleVectorSearch} className="search-bar-wrap">
              <input
                type="text"
                className="search-input-field"
                placeholder="Type query to test cosine similarity (e.g. 'circular economy in tourism')..."
                value={inspectorQuery}
                onChange={(e) => setInspectorQuery(e.target.value)}
              />
              <button
                type="submit"
                className="btn-send"
                disabled={!inspectorQuery.trim() || isSearchingVector}
              >
                {isSearchingVector ? 'Searching...' : 'Inspect Vectors'}
                <Search size={15} />
              </button>
            </form>

            {inspectorResults.length > 0 && (
              <div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-dim)', marginBottom: '1rem' }}>
                  Found {inspectorResults.length} matching chunks (sorted by similarity):
                </div>

                {inspectorResults.map((item, idx) => {
                  const scorePercent = Math.round(item.similarity_score * 100)
                  return (
                    <div key={idx} className="vector-result-card">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
                          <FileText size={16} color="var(--accent-primary)" />
                          <span>{item.metadata?.filename || 'Unknown Document'}</span>
                        </div>
                        <div className="citation-score-badge">
                          {scorePercent}% Similarity
                        </div>
                      </div>

                      <div className="similarity-meter-wrap">
                        <div
                          className="similarity-meter-fill"
                          style={{ width: `${Math.max(5, scorePercent)}%` }}
                        ></div>
                      </div>

                      <div className="citation-snippet" style={{ maxHeight: '180px' }}>
                        {item.content}
                      </div>

                      <div style={{ display: 'flex', gap: '1rem', marginTop: '0.75rem', fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                        <span>Chunk ID: {item.id}</span>
                        {item.metadata?.chunk_index !== undefined && (
                          <span>Chunk #{item.metadata.chunk_index + 1} of {item.metadata.total_chunks}</span>
                        )}
                        <span>Distance: {Number(item.distance).toFixed(4)}</span>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {/* ─── TAB 3: KNOWLEDGE BASE & DOCUMENTS ─── */}
        {activeTab === 'knowledge' && (
          <div className="docs-view">
            <div className="inspector-header">
              <h2 className="inspector-title">
                <FolderOpen size={24} color="var(--accent-emerald)" />
                Knowledge Base & Document Store
              </h2>
              <p className="inspector-sub">
                Explore loaded source files currently indexed in the vector database. Upload additional documents to expand the AI's internal knowledge.
              </p>
            </div>

            {/* Ingestion Dropzone Banner */}
            <div
              className={`dropzone-box ${isDragOver ? 'drag-over' : ''}`}
              style={{ padding: '2rem 1.5rem', marginBottom: '2rem' }}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true) }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={(e) => {
                e.preventDefault()
                setIsDragOver(false)
                handleFileUpload(e.dataTransfer.files)
              }}
              onClick={() => fileInputRef.current?.click()}
            >
              <div className="dropzone-icon" style={{ width: '48px', height: '48px' }}>
                <UploadCloud size={24} />
              </div>
              <h3 style={{ color: '#fff', fontSize: '1.05rem', marginBottom: '0.35rem' }}>
                {isUploading ? 'Chunking and generating embeddings...' : 'Upload Knowledge Documents'}
              </h3>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                Supports PDF, DOCX, and TXT files. Automatic text cleaning, chunking (1000 chars, 200 overlap), and local vector indexing.
              </p>
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '1rem', color: '#fff' }}>
                Ingested Files ({documentsList.length})
              </h3>
              <span style={{ fontSize: '0.8rem', color: 'var(--accent-emerald)' }}>
                ● {backendHealth.total_documents} total chunks stored
              </span>
            </div>

            <div className="doc-grid">
              {documentsList.length === 0 ? (
                <div style={{ gridColumn: '1 / -1', padding: '3rem', textAlign: 'center', color: 'var(--text-dim)' }}>
                  No documents found in the knowledge directory. Drop a file above to get started!
                </div>
              ) : (
                documentsList.map((doc, idx) => (
                  <div key={idx} className="doc-item-card">
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.85rem' }}>
                      <div className="doc-icon-wrap">
                        <FileText size={22} />
                      </div>
                      <div style={{ overflow: 'hidden' }}>
                        <div style={{ fontWeight: 600, color: '#fff', fontSize: '0.92rem', wordBreak: 'break-all' }}>
                          {doc.name}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)', marginTop: '0.2rem' }}>
                          Size: {doc.size_formatted}
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                      <span>Modified: {doc.modified}</span>
                      <span className="brand-badge" style={{ fontSize: '0.65rem' }}>
                        {doc.extension.toUpperCase()}
                      </span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>

      {/* Floating Toast Notification */}
      {toast && (
        <div className="toast-notice">
          {toast.type === 'error' ? (
            <AlertCircle size={16} color="var(--accent-rose)" />
          ) : (
            <Check size={16} color="var(--accent-emerald)" />
          )}
          <span>{toast.message}</span>
        </div>
      )}
    </div>
  )
}
