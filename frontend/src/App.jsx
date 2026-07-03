import { useState, useRef, useCallback, useEffect } from 'react'
import './index.css'

const API = 'http://localhost:8000'

// ─── Anomaly Banner ───────────────────────────────────────
function AnomalyBanner({ anomaly, summary }) {
  const status = summary?.overall_status || anomaly?.status || 'waiting'
  const message = summary?.overall_message || anomaly?.message || 'Start microphone or upload a file'

  const cfg = {
    normal:   { cls: 'banner-normal',   icon: '✅', title: 'System Normal' },
    anomaly:  { cls: 'banner-anomaly',  icon: <span className="warning-anim">🚨</span>, title: '⚠️ ANOMALY DETECTED' },
    warning:  { cls: 'banner-warning',  icon: '⚠️', title: 'PARTIAL ANOMALY' },
    silence:  { cls: 'banner-silence',  icon: '🔇', title: 'SILENCE' },
    waiting:  { cls: 'banner-waiting',  icon: '⏳', title: 'AWAITING INPUT' },
    no_ref:   { cls: 'banner-waiting',  icon: '⚙️', title: 'NOT CONFIGURED' },
  }
  const c = cfg[status] || cfg.waiting

  return (
    <div className={`anomaly-banner ${c.cls} ${status === 'anomaly' ? 'alert-card' : ''}`}>
      <span className="banner-icon">{c.icon}</span>
      <div className="banner-body">
        <div className="banner-title">{c.title}</div>
        <div className="banner-msg">{message}</div>
        <div className="banner-time" style={{fontSize: '0.7rem', marginTop: '4px', opacity: 0.6}}>
          Last check: {new Date().toLocaleTimeString()}
        </div>
      </div>
      {anomaly?.similarity !== undefined && status !== 'waiting' && status !== 'silence' && (
        <div className="banner-score">
          <div className="banner-score-val">{(anomaly.similarity * 100).toFixed(0)}%</div>
          <div className="banner-score-lbl">Match</div>
        </div>
      )}
      {summary && (
        <div className="banner-chips">
          <span className="chip chip-normal">✅ {summary.normal_count ?? 0}</span>
          <span className="chip chip-anomaly">🚨 {summary.anomaly_count ?? 0}</span>
          <span className="chip chip-silence">🔇 {summary.silence_count ?? 0}</span>
        </div>
      )}
    </div>
  )
}

// ─── Live Detection Box (per-second output) ───────────────
function LiveDetection({ anomaly }) {
  if (!anomaly || anomaly.status === 'waiting') return null
  const type = anomaly.current_type || anomaly.status
  const label = anomaly.current_label || '—'
  const conf  = anomaly.current_conf  || 0
  const sim   = anomaly.similarity    || 0
  const rms   = anomaly.rms          || 0

  const typeColor = {
    action:    '#06b6d4',
    normal:    '#22c55e',
    anomaly:   '#ef4444',
    silence:   '#64748b',
    uncertain: '#f97316',
  }
  const color = typeColor[type] || '#94a3b8'

  return (
    <div className={`live-detect ${type === 'anomaly' ? 'alert-card' : ''}`} style={{ borderColor: color }}>
      <div className="live-detect-label" style={{ color }}>{label}</div>
      <div className="live-detect-row">
        {conf > 0 && <span className="live-badge" style={{ background: color + '22', color }}>
          Model: {(conf * 100).toFixed(1)}%
        </span>}
        <span className="live-badge">Match: {(sim * 100).toFixed(0)}%</span>
        <span className="live-badge">RMS: {rms.toFixed(3)}</span>
      </div>
    </div>
  )
}

// ─── Anomaly History Section ──────────────────────────────
function AnomalyHistory({ history }) {
  if (!history || history.length === 0) {
    return <div className="empty-state">No anomalies recorded yet.</div>
  }

  return (
    <div style={{overflowX: 'auto'}}>
      <table className="history-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Type</th>
            <th>Confidence</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>
          {history.map((item, i) => (
            <tr key={i}>
              <td className="history-time">{item.time_str || new Date(item.timestamp * 1000).toLocaleTimeString()}</td>
              <td className={`history-label ${item.type === 'anomaly' ? 'history-status-red' : 'history-status-green'}`}>
                {item.label}
              </td>
              <td>{(item.confidence * 100).toFixed(1)}%</td>
              <td>{item.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Chatbot Component ────────────────────────────────────
function Chatbot() {
  const [isOpen, setIsOpen] = useState(false)
  const [messages, setMessages] = useState([{ text: "Hello! I'm your Audio Assistant. How can I help you today?", isBot: true }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async () => {
    if (!input.trim()) return
    const userMsg = input
    setInput('')
    setMessages(prev => [...prev, { text: userMsg, isBot: false }])
    setLoading(true)

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg })
      })
      const data = await res.json()
      setMessages(prev => [...prev, { text: data.reply, isBot: true }])
    } catch (e) {
      console.error('Chat error:', e)
      setMessages(prev => [...prev, { text: "Sorry, I'm having trouble connecting to the server.", isBot: true }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <div className="chat-toggle" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? '✕' : '💬'}
      </div>
      {isOpen && (
        <div className="chat-panel">
          <div className="chat-header">
            <span>Audio Assistant</span>
            <span className="chat-close" onClick={() => setIsOpen(false)}>✕</span>
          </div>
          <div className="chat-messages">
            {messages.map((m, i) => (
              <div key={i} className={`chat-msg ${m.isBot ? 'chat-msg-bot' : 'chat-msg-user'}`}>
                {m.text}
              </div>
            ))}
            {loading && <div className="chat-msg chat-msg-bot">...</div>}
            <div ref={chatEndRef} />
          </div>
          <div className="chat-input-area">
            <input 
              className="chat-input" 
              placeholder="Ask me anything..." 
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyPress={e => e.key === 'Enter' && handleSend()}
            />
            <button className="chat-send" onClick={handleSend}>➤</button>
          </div>
        </div>
      )}
    </>
  )
}

// ─── Running Sequence Log ─────────────────────────────────
function SequenceLog({ log }) {
  const endRef = useRef(null)
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [log])

  if (!log || log.length === 0) {
    return <div className="empty-state"><div className="empty-icon">📋</div><div className="empty-text">Sequence log will appear here during mic recording</div></div>
  }

  const typeIcon = { 
    action: '🎯', 
    normal: '✅', 
    anomaly: '🚨', 
    anomaly_sequence: '🔄', 
    silence: '🔇', 
    uncertain: '❓' 
  }

  return (
    <div className="seq-log">
      {log.map((entry, i) => {
        const isAnomaly = entry.type === 'anomaly' || entry.label === 'SEQUENCE_ANOMALY';
        const color = isAnomaly ? '#ef4444'
          : entry.type === 'action' ? '#06b6d4'
          : entry.type === 'normal' ? '#22c55e'
          : '#64748b'
        return (
          <div key={i} className={`seq-entry ${entry.type === 'anomaly' ? 'alert-card' : ''}`} style={{ borderLeftColor: color }}>
            <span className="seq-time">{entry.timestamp}s</span>
            <span className="seq-icon">{typeIcon[entry.type] || '•'}</span>
            <span className="seq-lbl" style={{ color }}>{entry.label}</span>
            {entry.conf > 0 && <span className="seq-conf">{entry.conf}%</span>}
            <span className="seq-sim">({entry.similarity}% match)</span>
          </div>
        )
      })}
      <div ref={endRef} />
    </div>
  )
}

// ─── Probability Bars ─────────────────────────────────────
function ProbBars({ probs }) {
  if (!probs || !Object.keys(probs).length) return <div className="empty-state"><div className="empty-text">Awaiting predictions…</div></div>
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1])
  return (
    <div className="prob-list">
      {sorted.map(([lbl, p]) => (
        <div key={lbl} className="prob-row">
          <span className="prob-lbl">{lbl}</span>
          <div className="prob-track"><div className="prob-fill" style={{ width: `${p * 100}%` }} /></div>
          <span className="prob-val">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

// ─── Waveform ─────────────────────────────────────────────
function Waveform({ active }) {
  const ref = useRef(); const anim = useRef()
  useEffect(() => {
    const c = ref.current; if (!c) return
    const draw = () => {
      const ctx = c.getContext('2d'), r = c.getBoundingClientRect()
      c.width = r.width * 2; c.height = r.height * 2; ctx.scale(2, 2)
      const [w, h] = [r.width, r.height], t = Date.now() / 1000, amp = active ? 28 : 6
      ctx.clearRect(0, 0, w, h)
      const g = ctx.createLinearGradient(0, 0, w, 0)
      g.addColorStop(0, active ? 'rgba(6,182,212,.9)' : 'rgba(6,182,212,.25)')
      g.addColorStop(1, active ? 'rgba(168,85,247,.9)' : 'rgba(168,85,247,.25)')
      ctx.strokeStyle = g; ctx.lineWidth = active ? 2.5 : 1.5; ctx.beginPath()
      for (let x = 0; x < w; x++) {
        const y = h / 2 + Math.sin(x * .03 + t * 2) * amp + Math.sin(x * .07 + t * 3.2) * (amp * .4) + (active ? Math.sin(x * .13 + t * 5) * 9 : 0)
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      }
      ctx.stroke()
      anim.current = requestAnimationFrame(draw)
    }
    draw(); return () => cancelAnimationFrame(anim.current)
  }, [active])
  return <div className="viz"><canvas ref={ref} /></div>
}

// ─── Main App ─────────────────────────────────────────────
export default function App() {
  const [listening, setListening] = useState(false)
  const [processing, setProcessing] = useState(false)
  const [anomaly, setAnomaly] = useState(null)
  const [anomalySummary, setAnomalySummary] = useState(null)
  const [probs, setProbs] = useState({})
  const [seqLog, setSeqLog] = useState([])
  const [history, setHistory] = useState([])
  const [stats, setStats] = useState({ windows: 0 })
  const [error, setError] = useState(null)
  const [statusMsg, setStatusMsg] = useState('')
  const [results, setResults] = useState(null)
  const fileRef = useRef()
  const pollRef = useRef()

  // Poll mic results and history
  useEffect(() => {
    const poll = async () => {
      try {
        const endpoints = [
          fetch(`${API}/history`).then(r => r.json()),
          fetch(`${API}/status`).then(r => r.json())
        ]
        
        if (listening) {
          endpoints.push(fetch(`${API}/mic-result`).then(r => r.json()))
          endpoints.push(fetch(`${API}/sequence-log`).then(r => r.json()))
        }

        const data = await Promise.all(endpoints)
        const historyData = data[0]
        const statusData = data[1]
        
        if (historyData.history) setHistory(historyData.history.reverse())
        if (statusData.windows_processed !== undefined) setStats(s => ({ ...s, windows: statusData.windows_processed }))
        
        if (listening && data[2] && data[3]) {
          const micData = data[2]
          const logData = data[3]
          if (micData.anomaly) setAnomaly(micData.anomaly)
          if (micData.all_probabilities) setProbs(micData.all_probabilities)
          if (logData.log) setSeqLog(logData.log)
        }
      } catch {}
    }
    
    poll()
    pollRef.current = setInterval(poll, 1000)
    return () => clearInterval(pollRef.current)
  }, [listening])

  const startMic = useCallback(async () => {
    setError(null); setAnomaly(null); setAnomalySummary(null); setSeqLog([])
    setStatusMsg('Starting microphone…')
    try {
      const r = await fetch(`${API}/start-mic`, { method: 'POST' })
      const d = await r.json()
      if (d.status === 'started') { setListening(true); setStatusMsg(`🎤 ${d.device || 'Microphone'}`) }
      else { setError(d.detail); setStatusMsg('') }
    } catch { setError('Cannot connect to backend on port 8000.'); setStatusMsg('') }
  }, [])

  const stopMic = useCallback(async () => {
    try { await fetch(`${API}/stop-mic`, { method: 'POST' }) } catch (e) { console.error('Stop mic error:', e) }
    setListening(false); setStatusMsg('⏹ Stopped.')
  }, [])

  const uploadFile = useCallback(async (file) => {
    if (!file) return
    setProcessing(true); setError(null); setResults(null)
    setAnomaly(null); setAnomalySummary(null); setSeqLog([])
    setStatusMsg(`Analyzing "${file.name}"…`)
    const fd = new FormData(); fd.append('file', file)
    try {
      const r = await fetch(`${API}/upload`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `Error ${r.status}`)
      const d = await r.json()
      if (d.status === 'error') throw new Error(d.error)
      setResults(d)
      setAnomalySummary(d.anomaly_summary)
      setStats({ windows: d.total_windows || 0 })
      if (d.windows?.length) {
        const last = d.windows[d.windows.length - 1]
        setProbs(last.all_probabilities || {})
        setAnomaly(last.anomaly || null)
        const log = d.windows.map(w => ({
          timestamp: w.timestamp?.toFixed ? w.timestamp.toFixed(1) : w.timestamp,
          label: w.anomaly?.current_label || '?',
          conf:  Math.round((w.anomaly?.current_conf || 0) * 100 * 10) / 10,
          type:  w.anomaly?.current_type || 'normal',
          similarity: Math.round((w.anomaly?.similarity || 1) * 100 * 10) / 10,
        }))
        setSeqLog(log)
      }
      setStatusMsg(`✅ Done — ${d.total_windows} windows analyzed`)
    } catch (e) { setError(e.message); setStatusMsg('') }
    finally { setProcessing(false); if (fileRef.current) fileRef.current.value = '' }
  }, [])

  const reset = useCallback(() => {
    setAnomaly(null); setAnomalySummary(null); setProbs({})
    setSeqLog([]); setStats({ windows: 0 }); setError(null)
    setStatusMsg(''); setResults(null)
  }, [])

  return (
    <>
      <header className="hdr">
        <div className="hdr-brand">
          <div className="hdr-logo">🎙️</div>
          <h1>AI Smart Home Monitoring System</h1>
        </div>
        <div className="hdr-badge">
          <span className={`dot ${listening ? 'dot-green' : processing ? 'dot-blue' : 'dot-dim'}`} />
          {listening ? 'Monitoring Live' : processing ? 'Processing…' : 'System Ready'}
        </div>
      </header>

      <div className="sticky-banner">
        <AnomalyBanner anomaly={anomaly} summary={anomalySummary} />
      </div>

      <main className="dash">

        {/* Controls */}
        <section className="card full">
          <div className="card-title">⚡ System Controls</div>
          <div className="ctrl-row">
            <button className="btn btn-primary" onClick={startMic} disabled={listening || processing}>🎤 Start Monitoring</button>
            <button className="btn btn-danger"  onClick={stopMic} disabled={!listening}>⏹ Stop</button>
            <button className="btn btn-sec"     onClick={() => fileRef.current?.click()} disabled={processing || listening}>
              {processing ? '⏳ Processing…' : '📁 Upload Session'}
            </button>
            {(results || anomaly || seqLog.length > 0) && (
              <button className="btn btn-sec" onClick={reset}>🔄 Clear Dashboard</button>
            )}
            <input ref={fileRef} type="file" accept=".wav,.mp3,.m4a,.flac,.ogg" style={{ display: 'none' }}
              onChange={e => { const f = e.target.files?.[0]; if (f) uploadFile(f) }} />
          </div>
          {error   && <div className="msg-error">⚠️ {error}</div>}
          {statusMsg && !error && <div className="msg-info">{statusMsg}</div>}
        </section>

        {/* Status Indicators */}
        <section className="card">
          <div className="card-title">🚥 System Status</div>
          <div style={{display: 'flex', alignItems: 'center', gap: '20px', padding: '10px'}}>
             <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                <div style={{width: '24px', height: '24px', borderRadius: '50%', background: (listening && anomaly?.is_anomaly) ? '#ef4444' : '#22c55e', boxShadow: (listening && anomaly?.is_anomaly) ? '0 0 15px #ef4444' : '0 0 15px #22c55e'}}></div>
                <span style={{fontSize: '0.7rem', marginTop: '5px', color: 'var(--text2)'}}>DETECTOR</span>
             </div>
             <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                <div style={{width: '24px', height: '24px', borderRadius: '50%', background: listening ? '#22c55e' : '#475569', boxShadow: listening ? '0 0 15px #22c55e' : 'none'}}></div>
                <span style={{fontSize: '0.7rem', marginTop: '5px', color: 'var(--text2)'}}>MIC ACTIVE</span>
             </div>
             <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                <div style={{width: '24px', height: '24px', borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 15px #22c55e'}}></div>
                <span style={{fontSize: '0.7rem', marginTop: '5px', color: 'var(--text2)'}}>BACKEND</span>
             </div>
          </div>
        </section>

        {/* Live detection */}
        {listening && (
          <section className="card">
            <div className="card-title">⚡ Live Analysis</div>
            <LiveDetection anomaly={anomaly} />
          </section>
        )}

        {/* Anomaly History Table */}
        <section className="card full">
          <div className="card-title">🕰️ Anomaly History Log</div>
          <AnomalyHistory history={history} />
        </section>

        {/* Stats */}
        <section className="card full">
          <div className="stats-row">
            <div className="stat"><div className="stat-n">{stats.windows}</div><div className="stat-l">Windows</div></div>
            <div className="stat"><div className="stat-n">{seqLog.filter(e => e.type === 'action').length}</div><div className="stat-l">Actions</div></div>
            <div className="stat"><div className="stat-n">{seqLog.filter(e => e.type === 'anomaly').length}</div><div className="stat-l">Anomalies</div></div>
            <div className="stat"><div className="stat-n">{history.length}</div><div className="stat-l">Saved Alerts</div></div>
          </div>
        </section>

        {/* Waveform */}
        <section className="card">
          <div className="card-title">〰️ Audio Stream</div>
          <Waveform active={listening} />
        </section>

        {/* Probabilities */}
        <section className="card">
          <div className="card-title">📊 Confidence Levels</div>
          <ProbBars probs={probs} />
        </section>

        {/* Detection Log */}
        <section className="card full">
          <div className="card-title">
            📋 Session Timeline
            {listening && <span className="live-pill">● LIVE</span>}
          </div>
          <SequenceLog log={seqLog} />
        </section>

      </main>

      <Chatbot />
    </>
  )
}
