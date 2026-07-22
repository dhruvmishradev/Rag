"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Key, 
  UploadCloud, 
  BookOpen, 
  Trash2, 
  Send, 
  ChevronDown, 
  ChevronUp, 
  FileText, 
  Sparkles, 
  AlertTriangle,
  FileCheck
} from "lucide-react";

// API Base URL
const API_BASE = "http://localhost:8000/api";

interface DocSource {
  page_content: string;
  metadata: {
    page?: number;
    source?: string;
    [key: string]: any;
  };
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  docs?: DocSource[];
}

export default function RAGDashboard() {
  // Config & State
  const [apiKey, setApiKey] = useState("");
  const [isApiKeyConfigured, setIsApiKeyConfigured] = useState(false);
  const [processed, setProcessed] = useState(false);
  const [currentFile, setCurrentFile] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  
  // Chat States
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  
  // UI Helpers
  const [expandedSources, setExpandedSources] = useState<{ [key: number]: boolean }>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, isQuerying]);

  // Check backend status on load
  useEffect(() => {
    fetchStatus();
  }, []);

  const fetchStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/status`);
      if (res.ok) {
        const data = await res.json();
        setProcessed(data.processed);
        setCurrentFile(data.current_file);
        setIsApiKeyConfigured(data.api_key_configured || false);
      }
    } catch (err) {
      console.error("Failed to fetch status:", err);
      setErrorMessage("Backend is offline. Make sure uvicorn is running on port 8000.");
    }
  };

  // Drag and Drop Handlers
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/pdf") {
        setUploadedFile(file);
        setErrorMessage(null);
      } else {
        setErrorMessage("Only PDF files are supported.");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.type === "application/pdf") {
        setUploadedFile(file);
        setErrorMessage(null);
      } else {
        setErrorMessage("Only PDF files are supported.");
      }
    }
  };

  // Ingest/Process Document
  const handleIngest = async () => {
    if (!uploadedFile) return;
    setIsProcessing(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", uploadedFile);
    if (!isApiKeyConfigured && apiKey) {
      formData.append("api_key", apiKey);
    }

    try {
      const response = await fetch(`${API_BASE}/process`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Failed to process document");
      }

      const data = await response.json();
      setProcessed(true);
      setCurrentFile(data.current_file);
      setChatHistory([]); // Reset history on new document
      setUploadedFile(null);
    } catch (err: any) {
      setErrorMessage(err.message || "An error occurred during ingestion.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Clear Database
  const handleClear = async () => {
    setIsProcessing(true);
    try {
      const response = await fetch(`${API_BASE}/clear`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ api_key: apiKey }),
      });

      if (response.ok) {
        setProcessed(false);
        setCurrentFile(null);
        setChatHistory([]);
        setUploadedFile(null);
      } else {
        throw new Error("Failed to clear database.");
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Could not clear database.");
    } finally {
      setIsProcessing(false);
    }
  };

  // Send Query to LLM
  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isQuerying || !processed) return;

    const userQuery = query.trim();
    setQuery("");
    setChatHistory(prev => [...prev, { role: "user", content: userQuery }]);
    setIsQuerying(true);
    setErrorMessage(null);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: userQuery,
          api_key: (!isApiKeyConfigured && apiKey) ? apiKey : undefined,
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Error query LLM");
      }

      const data = await response.json();
      setChatHistory(prev => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          docs: data.docs,
        },
      ]);
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to query the document assistant.");
    } finally {
      setIsQuerying(false);
    }
  };

  const toggleSources = (index: number) => {
    setExpandedSources(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <div className="app-container">
      {/* Full-Screen Processing Spinner */}
      <AnimatePresence>
        {isProcessing && (
          <motion.div 
            className="processing-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div className="spinner-container">
              <div className="spinner"></div>
              <div className="processing-text">
                Ingesting Document
                <span className="processing-subtext">Generating vector embeddings...</span>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Sidebar Controls */}
      <aside className="sidebar">
        <div className="logo-section">
          <BookOpen size={16} />
          <span>RAG Assistant</span>
        </div>

        {/* API Config Panel */}
        <div className="sidebar-section">
          <div className="sidebar-title">
            API Configuration
          </div>
          <div className="input-wrapper">
            <input 
              type="text"
              className="input-field"
              style={{
                borderColor: "var(--border-subtle)",
                color: "var(--text-secondary)",
                backgroundColor: "transparent",
                cursor: "default"
              }}
              value="Mistral API Key..."
              disabled={true}
              readOnly={true}
            />
          </div>
        </div>

        {/* Document Ingestion */}
        <div className="sidebar-section">
          <div className="sidebar-title">
            Ingest Document
          </div>

          <div 
            className={`dropzone ${isDragActive ? "active" : ""}`}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
          >
            <input 
              type="file" 
              id="file-upload" 
              accept=".pdf" 
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <label htmlFor="file-upload" style={{ cursor: "pointer", display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
              <UploadCloud size={20} style={{ color: "var(--text-secondary)" }} />
              <span className="dropzone-text">
                {uploadedFile ? uploadedFile.name : "Choose PDF file"}
              </span>
              <span className="dropzone-subtext">
                {uploadedFile ? `${(uploadedFile.size / 1024 / 1024).toFixed(2)} MB` : "or drag here"}
              </span>
            </label>
          </div>

          {uploadedFile && (
            <motion.button 
              className="btn btn-primary"
              onClick={handleIngest}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.15 }}
            >
              Process Document
            </motion.button>
          )}
        </div>

        {/* Database Status Panel */}
        <div className="sidebar-section" style={{ marginTop: "auto" }}>
          <div className="status-card">
            <div className="status-badge">
              <span className={`status-dot ${processed ? "active" : "inactive"}`}></span>
              <span>
                {processed ? "Active Index" : "Empty Index"}
              </span>
            </div>
            
            {processed && (
              <>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: 2 }}>
                  <FileCheck size={12} />
                  <span style={{ textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap" }}>
                    {currentFile || "Pre-loaded DB"}
                  </span>
                </div>
                <button 
                  className="btn btn-danger" 
                  onClick={handleClear}
                  style={{ marginTop: 6 }}
                >
                  Clear Database
                </button>
              </>
            )}
          </div>
        </div>
      </aside>

      {/* Main Panel Content */}
      <main className="main-content">
        {/* Top Header */}
        <header className="header">
          <div>
            <h1 className="header-title">Document Q&A</h1>
          </div>
        </header>

        {/* Alert for Errors */}
        {errorMessage && (
          <div style={{
            background: "rgba(239, 68, 68, 0.08)",
            borderBottom: "1px solid rgba(239, 68, 68, 0.15)",
            color: "var(--color-danger)",
            padding: "0.6rem 3rem",
            fontSize: "0.8rem",
            display: "flex",
            alignItems: "center",
            gap: 6
          }}>
            <AlertTriangle size={14} />
            {errorMessage}
          </div>
        )}

        {/* Chat / Welcome View */}
        {!processed ? (
          <div className="empty-state">
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, ease: "easeOut" }}
              style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <h2 className="empty-state-title">RAG Document Assistant</h2>
              </div>
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: 32 }}
                transition={{ delay: 0.25, duration: 0.6, ease: "easeOut" }}
                style={{ height: "1.5px", backgroundColor: "var(--text-primary)", borderRadius: "1px" }}
              />
              <motion.p 
                className="empty-state-desc"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.15, duration: 0.5 }}
                style={{ marginTop: "0.5rem" }}
              >
                An interface for querying PDF books using semantic retrieval.
                To get started, configure your API Key and upload a PDF. Once processed, you can query specific sections, terms, or summaries directly.
              </motion.p>
            </motion.div>
          </div>
        ) : (
          <div className="chat-messages">
            {chatHistory.length === 0 && (
              <div className="empty-state" style={{ padding: 0 }}>
                <motion.div
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, ease: "easeOut" }}
                  style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}
                >
                  <h3 className="empty-state-title" style={{ fontSize: "1.1rem" }}>Index Ready</h3>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: 24 }}
                    transition={{ delay: 0.2, duration: 0.5, ease: "easeOut" }}
                    style={{ height: "1.5px", backgroundColor: "var(--text-primary)", borderRadius: "1px" }}
                  />
                  <motion.p 
                    className="empty-state-desc"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.12, duration: 0.4 }}
                    style={{ marginTop: "0.25rem" }}
                  >
                    Ask a question to query your processed document. The retriever will find relevant chunks and Mistral will generate an answer based on them.
                  </motion.p>
                </motion.div>
              </div>
            )}
            
            {chatHistory.map((msg, index) => (
              <div 
                key={index}
                className={`message-row ${msg.role}`}
              >
                <div className="message-bubble">
                  <div className="message-header">
                    {msg.role === "user" ? "You" : "Assistant"}
                  </div>
                  <div>{msg.content}</div>

                  {/* Page numbers summary */}
                  {msg.role === "assistant" && msg.docs && msg.docs.length > 0 && (
                    <div style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.25rem", display: "flex", gap: "0.35rem", flexWrap: "wrap" }}>
                      <span>Cited:</span>
                      {Array.from(new Set(msg.docs.map(d => (d.metadata.page || 0) + 1))).sort((a,b) => a-b).map((page, idx, arr) => (
                        <span key={page} style={{ color: "var(--text-secondary)", fontWeight: 500 }}>
                          Page {page}{idx < arr.length - 1 ? "," : ""}
                        </span>
                      ))}
                    </div>
                  )}
                  
                  {/* Retrieved Sources expander */}
                  {msg.docs && msg.docs.length > 0 && (
                    <>
                      <div 
                        className="citations-toggle"
                        onClick={() => toggleSources(index)}
                      >
                        {expandedSources[index] ? "Hide sources" : "Show sources"}
                        {expandedSources[index] ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                      </div>

                      <AnimatePresence>
                        {expandedSources[index] && (
                          <motion.div 
                            className="citations-container"
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.15 }}
                            style={{ overflow: "hidden" }}
                          >
                            {msg.docs.map((doc, docIdx) => (
                              <div className="citation-card" key={docIdx}>
                                <div className="citation-meta">
                                  Source Chunk {docIdx + 1} — Page {(doc.metadata.page || 0) + 1}
                                </div>
                                <div className="citation-content">
                                  {doc.page_content}
                                </div>
                              </div>
                            ))}
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </>
                  )}
                </div>
              </div>
            ))}

            {/* In-chat querying loading bubble */}
            {isQuerying && (
              <div className="message-row assistant">
                <div className="message-bubble">
                  <div className="message-header">Assistant</div>
                  <div style={{ display: "flex", gap: 3, alignItems: "center", marginTop: 4 }}>
                    <motion.div 
                      className="status-dot" 
                      style={{ backgroundColor: "var(--text-secondary)", width: 4, height: 4 }}
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0 }}
                    />
                    <motion.div 
                      className="status-dot" 
                      style={{ backgroundColor: "var(--text-secondary)", width: 4, height: 4 }}
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
                    />
                    <motion.div 
                      className="status-dot" 
                      style={{ backgroundColor: "var(--text-secondary)", width: 4, height: 4 }}
                      animate={{ opacity: [0.3, 1, 0.3] }}
                      transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
                    />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}

        {/* Chat input block */}
        {processed && (
          <div className="chat-input-container">
            <form onSubmit={handleQuery} className="chat-input-form">
              <input 
                type="text" 
                className="chat-input"
                placeholder="Ask a question..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={isQuerying}
              />
              <button 
                type="submit" 
                className="send-btn"
                disabled={!query.trim() || isQuerying}
              >
                <Send size={14} />
              </button>
            </form>
          </div>
        )}
      </main>
    </div>
  );
}
