import { useEffect, useState, useRef, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";
const STAFF_API_KEY = import.meta.env.VITE_API_KEY || "";

function staffApi() {
  const headers = {};
  if (STAFF_API_KEY) {
    headers["X-API-Key"] = STAFF_API_KEY;
  }
  return axios.create({ baseURL: API_BASE_URL, headers });
}

function getStaffName() {
  return localStorage.getItem("jewelguard_staff_name") || "staff";
}

function LiveFeed({ view, running, visible, sessionKey }) {
  const streamSrc = running
    ? `${API_BASE_URL}/frame_stream?view=${view}&s=${sessionKey}`
    : `${API_BASE_URL}/frame?view=${view}&s=${sessionKey}&t=0`;

  if (!visible) {
    return null;
  }

  return (
    <div className="camera-frame-inner">
      <img
        key={`${view}-${sessionKey}-${running ? "live" : "idle"}`}
        className="camera-feed"
        src={streamSrc}
        alt={`${view} monitoring feed`}
      />
    </div>
  );
}

function InfoCard({ title, value, subtitle }) {
  return (
    <div className="info-card">
      <p className="info-title">{title}</p>
      <h2 className="info-value">{value}</h2>
      {subtitle && <p className="info-subtitle">{subtitle}</p>}
    </div>
  );
}

function IncidentCard({ incident, onSelect, selected }) {
  const riskLevel = incident.risk_level || "LOW";
  const workflow = (incident.status || "open").toLowerCase();

  return (
    <div
      className={`incident-card ${riskLevel === "HIGH" ? "incident-high" : "incident-medium"}${selected ? " incident-card-selected" : ""}`}
      onClick={() => onSelect(incident)}
    >
      <div className="incident-top">
        <h3>{incident.person_id || incident.mask_status || "Alert"}</h3>
        <div className="incident-badges">
          <span className={`workflow-pill workflow-${workflow}`}>
            {workflow.toUpperCase()}
          </span>
          <span className={`risk-pill risk-${riskLevel.toLowerCase()}`}>
            {riskLevel}
          </span>
        </div>
      </div>
      <p className="incident-time">{incident.timestamp}</p>
      <p className="incident-desc">
        {incident.risk_description || "No description available."}
      </p>
      <div className="incident-meta">
        <span>Risk: {incident.risk_score}</span>
        <span>Zone: {incident.alert_zone || "general"}</span>
        <span>Wrist: {incident.wrist_near_case ? "YES" : "NO"}</span>
        <span>Mask: {incident.mask_status || "Unknown"}</span>
      </div>
    </div>
  );
}

function IncidentList({ title, incidents, emptyText, onSelect, onExport, selectedId }) {
  return (
    <section className="incident-list-section full-width-section">
      <div className="section-title-row">
        <h2>{title}</h2>
        <div className="section-title-actions">
          <span className="incident-count">{incidents.length} records</span>
          {onExport && (
            <button type="button" className="export-button" onClick={onExport}>
              Export CSV
            </button>
          )}
        </div>
      </div>
      {incidents.length > 0 ? (
        <div className="incident-list alert-list-grid">
          {incidents.map((incident) => (
            <IncidentCard
              key={incident.id}
              incident={incident}
              onSelect={onSelect}
              selected={selectedId === incident.id}
            />
          ))}
        </div>
      ) : (
        <div className="empty-box">{emptyText}</div>
      )}
    </section>
  );
}

function IncidentsWithEvidence({
  title,
  incidents,
  emptyText,
  selectedIncident,
  onSelect,
  onUpdated,
  onExport,
}) {
  return (
    <div className="incidents-page">
      <IncidentList
        title={title}
        incidents={incidents}
        emptyText={emptyText}
        onSelect={onSelect}
        onExport={onExport}
        selectedId={selectedIncident?.id}
      />
      <EvidenceDetails selectedIncident={selectedIncident} onUpdated={onUpdated} />
    </div>
  );
}

function EvidenceDetails({ selectedIncident, onUpdated }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState("");

  useEffect(() => {
    setNote(selectedIncident?.staff_note || "");
  }, [selectedIncident]);

  if (!selectedIncident) {
    return (
      <section className="incident-detail-section">
        <h2>Evidence Details</h2>
        <div className="empty-box">
          Select an incident to review evidence and update workflow status.
        </div>
      </section>
    );
  }

  const workflow = (selectedIncident.status || "open").toLowerCase();

  const runAction = async (action) => {
    try {
      setBusy(action);
      const api = staffApi();
      const payload = { staff_name: getStaffName(), staff_note: note || undefined };
      const response = await api.post(
        `/incidents/${selectedIncident.id}/${action}`,
        payload
      );
      onUpdated?.(response.data);
    } catch (error) {
      console.error(`${action} error:`, error);
    } finally {
      setBusy("");
    }
  };

  return (
    <section className="incident-detail-section">
      <h2>Evidence Details</h2>
      <div className="detail-card">
        <div className="detail-top-row">
          <h3>{selectedIncident.person_id || "Incident"} #{selectedIncident.id}</h3>
          <span className={`workflow-pill workflow-${workflow}`}>
            {workflow.toUpperCase()}
          </span>
        </div>
        <p>{selectedIncident.timestamp}</p>
        <p>{selectedIncident.risk_description}</p>
        <p className="detail-meta">
          Zone: {selectedIncident.alert_zone || "general"} · Risk:{" "}
          {selectedIncident.risk_score} ({selectedIncident.risk_level})
        </p>
        {selectedIncident.screenshot_path && (
          <img
            className="evidence-image"
            src={`${API_BASE_URL}/screenshot?path=${encodeURIComponent(selectedIncident.screenshot_path)}`}
            alt="Incident screenshot"
          />
        )}
        <label className="staff-note-label" htmlFor="staff-note">
          Staff note
        </label>
        <textarea
          id="staff-note"
          className="staff-note-input"
          rows={3}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Optional note for audit trail…"
        />
        <div className="incident-action-row">
          {workflow === "open" && (
            <button
              type="button"
              className="ack-button"
              disabled={Boolean(busy)}
              onClick={() => runAction("acknowledge")}
            >
              {busy === "acknowledge" ? "Saving…" : "Acknowledge"}
            </button>
          )}
          {workflow !== "resolved" && (
            <button
              type="button"
              className="resolve-button"
              disabled={Boolean(busy)}
              onClick={() => runAction("resolve")}
            >
              {busy === "resolve" ? "Saving…" : "Mark resolved"}
            </button>
          )}
        </div>
      </div>
    </section>
  );
}

function TopBar({
  health,
  status,
  staffName,
  onStaffNameChange,
  onStart,
  onStop,
  isStarting,
  isStopping,
  staffModeActive,
  onToggleStaffMode,
  staffModeBusy,
}) {
  const running = health?.engineRunning || status?.running;
  const mode = health?.mode || status?.mode || "—";
  const profile = health?.videoProfile;
  const runtime = health?.runtimeProfile || status?.runtimeProfile;
  const runtimeLabel =
    health?.runtimeLabel ||
    status?.runtimeLabel ||
    (runtime === "demo" ? "Demo" : runtime === "live" ? "Live" : null);

  return (
    <header className="top-bar">
      <div className="brand-block">
        <div className="brand-mark">VV</div>
        <div>
          <h1>{health?.productName || status?.productName || "VaultVision"}</h1>
          <p>
            {health?.productTagline ||
              status?.productTagline ||
              "Retail Surveillance System"}
          </p>
        </div>
      </div>

      <div className="status-pills">
        <span className={`status-pill ${running ? "pill-live" : "pill-off"}`}>
          {running ? "● LIVE" : "○ OFFLINE"}
        </span>
        {(health?.store || status?.storeName) && (
          <span className="status-pill pill-neutral">
            {health?.store || status?.storeName}
          </span>
        )}
        <span className="status-pill pill-neutral">
          {mode}{profile ? ` · ${profile}` : ""}
          {runtimeLabel ? ` · ${runtimeLabel}` : ""}
        </span>
        {(status?.flaggedCount ?? 0) > 0 && !staffModeActive && (
          <span className="status-pill pill-alert">
            {status.flaggedCount} flagged
          </span>
        )}
        {staffModeActive && (
          <span className="status-pill pill-staff-mode">STAFF MODE</span>
        )}
        {(status?.crowdActive && !staffModeActive) && (
          <span className="status-pill pill-crowd">
            Crowd {status.customerCount ?? 0}
          </span>
        )}
      </div>

      <div className="top-bar-actions">
        <button
          type="button"
          className={staffModeActive ? "staff-mode-button active" : "staff-mode-button"}
          onClick={onToggleStaffMode}
          disabled={staffModeBusy || !status?.running}
          title="Pause customer alerts during restock / closed floor"
        >
          {staffModeBusy ? "…" : staffModeActive ? "Staff mode ON" : "Staff mode"}
        </button>
        <input
          className="staff-inline-input"
          value={staffName}
          onChange={(e) => onStaffNameChange(e.target.value)}
          placeholder="Staff name"
          title="Used in audit log for dismiss & incidents"
        />
        <button
          className="start-button"
          onClick={onStart}
          disabled={isStarting}
        >
          {isStarting ? "…" : "Start"}
        </button>
        <button className="stop-button" onClick={onStop} disabled={isStopping}>
          {isStopping ? "…" : "Stop"}
        </button>
      </div>
    </header>
  );
}

function StaffActionsPanel({
  status,
  onDismiss,
  onMarkStaff,
  onUnmarkStaff,
  dismissingId,
  markingStaffId,
}) {
  const staffModeOn = Boolean(status?.staffModeActive);
  const flagged = staffModeOn ? [] : (status?.flaggedPeople ?? []);
  const dismissed = status?.dismissedIds ?? [];
  const staffIds = status?.staffIds ?? [];
  const tracked = status?.trackedPersonIds ?? [];
  const running = status?.running;
  const customersOnly = staffModeOn
    ? []
    : tracked.filter(
        (id) => !staffIds.includes(id) && !dismissed.includes(id)
      );
  const hasActivity =
    staffModeOn ||
    flagged.length > 0 ||
    dismissed.length > 0 ||
    staffIds.length > 0;

  return (
    <section
      className={`staff-actions-panel ${hasActivity ? "staff-actions-active" : ""}`}
    >
      <div className="staff-actions-head">
        <h3>Staff actions</h3>
        <span className="staff-actions-hint">Flags & staff marking</span>
      </div>

      {!running ? (
        <p className="staff-actions-idle">Start the engine to manage people.</p>
      ) : staffModeOn ? (
        <p className="staff-actions-idle staff-mode-note">
          Staff mode is on — everyone detected is auto-staff (no mask scan, no
          crowd alerts). Turn off staff mode when customers arrive.
        </p>
      ) : (
        <>
          {flagged.length > 0 ? (
            <ul className="flag-review-list compact">
              {flagged.map((person) => (
                <li key={person.personId} className="flag-review-row">
                  <div className="flag-review-info">
                    <strong>{person.personId}</strong>
                    <span>
                      {person.maskStatus} · {Math.round((person.confidence || 0) * 100)}%
                    </span>
                  </div>
                  <button
                    type="button"
                    className="flag-dismiss-action"
                    disabled={dismissingId === person.personId}
                    onClick={() => onDismiss(person.personId)}
                  >
                    {dismissingId === person.personId ? "…" : "Dismiss flag"}
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <p className="staff-actions-idle">No flagged customers.</p>
          )}

          {customersOnly.length > 0 && (
            <div className="staff-mark-block">
              <p className="staff-mark-label">Mark working staff (no mask scan):</p>
              <ul className="flag-review-list compact">
                {customersOnly.map((personId) => (
                  <li key={personId} className="flag-review-row staff-mark-row">
                    <strong>{personId}</strong>
                    <button
                      type="button"
                      className="staff-mark-action"
                      disabled={markingStaffId === personId}
                      onClick={() => onMarkStaff(personId)}
                    >
                      {markingStaffId === personId ? "…" : "Mark staff"}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {staffModeOn && tracked.length > 0 && (
        <div className="dismissed-chips">
          <span className="dismissed-label">Auto-staff:</span>
          {tracked.map((id) => (
            <span key={id} className="staff-chip staff-chip-readonly">
              {id}
            </span>
          ))}
        </div>
      )}

      {!staffModeOn && staffIds.length > 0 && (
        <div className="dismissed-chips">
          <span className="dismissed-label">Staff:</span>
          {staffIds.map((id) => (
            <button
              key={id}
              type="button"
              className="staff-chip"
              title="Click to resume customer tracking"
              onClick={() => onUnmarkStaff(id)}
            >
              {id} ×
            </button>
          ))}
        </div>
      )}

      {dismissed.length > 0 && (
        <div className="dismissed-chips">
          <span className="dismissed-label">Dismissed flags:</span>
          {dismissed.map((id) => (
            <span key={id} className="dismissed-chip">
              {id}
            </span>
          ))}
        </div>
      )}
    </section>
  );
}

function AlarmStrip({ status }) {
  if (!status?.alarmActive) {
    return null;
  }

  const level = status.alarmLevel || "FLAG";
  const labels = {
    FLAG: "Masked person flagged",
    CROWD_FLAG: "Flagged person in customer group",
    WRIST_NEAR: "Wrist near display case",
    WRIST_FLAG: "Critical — flagged + wrist at case",
    WRIST_INSIDE: "Critical — hand inside case",
  };

  const cssClass =
    level === "WRIST_INSIDE" || level === "WRIST_FLAG"
      ? "alarm-strip alarm-critical"
      : level === "WRIST_NEAR"
        ? "alarm-strip alarm-wrist"
        : "alarm-strip alarm-flag";

  return (
    <div className={cssClass}>
      <strong>{labels[level] || "Alert active"}</strong>
      <span>
        Risk {status.riskScore ?? 0} · Flagged {status.flaggedCount ?? 0}
      </span>
    </div>
  );
}

function MonitoringPanel({
  title,
  view,
  status,
  feedSessionKey,
  riskLevel,
  alertType,
  riskScore,
  reasons,
  metrics,
  onDismiss,
  onMarkStaff,
  onUnmarkStaff,
  dismissingId,
  markingStaffId,
}) {
  return (
    <main className="monitor-layout">
      <section className="camera-section">
        <div className="section-title-row">
          <h2>{title}</h2>
          <span className={status?.running ? "live-badge" : "offline-badge"}>
            {status?.running ? "LIVE" : "OFFLINE"}
          </span>
        </div>
        <div className="camera-frame">
          <LiveFeed
            view={view}
            running={Boolean(status?.running)}
            visible
            sessionKey={feedSessionKey}
          />
        </div>
      </section>

      <aside className="risk-section side-panel">
        <h2 className="side-panel-title">Risk overview</h2>

        <div className={`risk-box risk-${riskLevel.toLowerCase()} risk-box-compact`}>
          <p>{alertType}</p>
          <h1>{riskScore}</h1>
          <h2>{riskLevel}</h2>
        </div>

        <StaffActionsPanel
          status={status}
          onDismiss={onDismiss}
          onMarkStaff={onMarkStaff}
          onUnmarkStaff={onUnmarkStaff}
          dismissingId={dismissingId}
          markingStaffId={markingStaffId}
        />

        <div className="compact-grid">{metrics}</div>

        <div className="reasons-box reasons-box-compact">
          <h3>Active reasons</h3>
          {reasons.length > 0 ? (
            <ul>
              {reasons.map((reason, index) => (
                <li key={index}>{reason}</li>
              ))}
            </ul>
          ) : (
            <p className="reasons-clear">All clear — no warnings.</p>
          )}
        </div>
      </aside>
    </main>
  );
}

function playAlarmTone(level) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const tones =
      level === "WRIST_INSIDE" || level === "WRIST_FLAG"
        ? [880, 1100, 880, 1100]
        : level === "WRIST_NEAR"
          ? [660, 880, 660]
          : [520, 660];

    let start = ctx.currentTime;
    tones.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "square";
      osc.frequency.value = freq;
      gain.gain.value = 0.08;
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(start + i * 0.18);
      osc.stop(start + i * 0.18 + 0.15);
    });
  } catch (e) {
    console.warn("Alarm audio unavailable:", e);
  }
}

function App() {
  const [activeTab, setActiveTab] = useState("entrance");
  const [status, setStatus] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [engineMessage, setEngineMessage] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [feedSessionKey, setFeedSessionKey] = useState(0);
  const [dismissingId, setDismissingId] = useState("");
  const [markingStaffId, setMarkingStaffId] = useState("");
  const [staffModeBusy, setStaffModeBusy] = useState(false);
  const [health, setHealth] = useState(null);
  const [staffName, setStaffName] = useState(() => getStaffName());
  const prevAlarmLevel = useRef("NONE");
  const prevFlaggedCount = useRef(0);

  const isMonitorTab = activeTab === "entrance" || activeTab === "store";

  const handleAlarm = useCallback((nextStatus) => {
    if (!nextStatus?.running) {
      return;
    }
    const level = nextStatus.alarmLevel || "NONE";
    const flagged = Number(nextStatus.flaggedCount || 0);
    const levelRank = { NONE: 0, FLAG: 1, CROWD_FLAG: 2, WRIST_NEAR: 3, WRIST_FLAG: 4, WRIST_INSIDE: 5 };
    const prevRank = levelRank[prevAlarmLevel.current] ?? 0;
    const nextRank = levelRank[level] ?? 0;
    if (nextRank > prevRank || (level === "FLAG" && flagged > prevFlaggedCount.current)) {
      playAlarmTone(level);
    }
    prevAlarmLevel.current = level;
    prevFlaggedCount.current = flagged;
  }, []);

  const tabs = [
    { id: "entrance", label: "Entrance" },
    { id: "store", label: "In-Store" },
    { id: "active", label: "Active Alerts" },
    { id: "critical", label: "Critical" },
    { id: "identity", label: "Identity" },
    { id: "incidents", label: "Incidents" },
  ];

  const fetchHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setHealth(response.data);
    } catch (error) {
      console.error("Health fetch error:", error);
    }
  };

  const fetchStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status`);
      setStatus(response.data);
      handleAlarm(response.data);
    } catch (error) {
      console.error("Status fetch error:", error);
    }
  };

  const fetchIncidents = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/incidents`);
      setIncidents(response.data);
    } catch (error) {
      console.error("Incident fetch error:", error);
    }
  };

  const startEngine = async () => {
    try {
      setIsStarting(true);
      const response = await staffApi().post("/start");
      setEngineMessage(response.data.message || "Vision engine started");
      setFeedSessionKey((key) => key + 1);
      await fetchStatus();
      await fetchHealth();
    } catch (error) {
      setEngineMessage(
        error.response?.data?.error ||
          error.response?.data?.message ||
          "Failed to start vision engine"
      );
    } finally {
      setIsStarting(false);
    }
  };

  const stopEngine = async () => {
    try {
      setIsStopping(true);
      const response = await staffApi().post("/stop");
      setEngineMessage(response.data.message || "Vision engine stopped");
      setFeedSessionKey((key) => key + 1);
      await fetchStatus();
      await fetchHealth();
    } catch (error) {
      setEngineMessage("Failed to stop vision engine");
    } finally {
      setIsStopping(false);
    }
  };

  const dismissFalseFlag = async (personId) => {
    try {
      setDismissingId(personId);
      await staffApi().post("/flags/clear", {
        person_id: personId,
        staff_name: getStaffName(),
      });
      setEngineMessage(`Dismissed flag for ${personId}`);
      await fetchStatus();
    } catch (error) {
      setEngineMessage("Failed to dismiss flag");
    } finally {
      setDismissingId("");
    }
  };

  const markAsStaff = async (personId) => {
    try {
      setMarkingStaffId(personId);
      await staffApi().post("/staff/mark", {
        person_id: personId,
        staff_name: getStaffName(),
      });
      setEngineMessage(`Marked ${personId} as staff`);
      await fetchStatus();
    } catch (error) {
      setEngineMessage("Failed to mark staff");
    } finally {
      setMarkingStaffId("");
    }
  };

  const unmarkStaff = async (personId) => {
    try {
      setMarkingStaffId(personId);
      await staffApi().post("/staff/unmark", {
        person_id: personId,
        staff_name: getStaffName(),
      });
      setEngineMessage(`Removed staff mark from ${personId}`);
      await fetchStatus();
    } catch (error) {
      setEngineMessage("Failed to unmark staff");
    } finally {
      setMarkingStaffId("");
    }
  };

  const toggleStaffMode = async () => {
    const active = Boolean(status?.staffModeActive);
    try {
      setStaffModeBusy(true);
      const endpoint = active ? "/staff-mode/stop" : "/staff-mode/start";
      const response = await staffApi().post(endpoint, { staff_name: getStaffName() });
      setEngineMessage(response.data.message || (active ? "Staff mode off" : "Staff mode on"));
      await fetchStatus();
    } catch (error) {
      setEngineMessage("Failed to toggle staff mode");
    } finally {
      setStaffModeBusy(false);
    }
  };

  const handleStaffNameChange = (name) => {
    setStaffName(name);
    localStorage.setItem("jewelguard_staff_name", name);
  };

  const handleIncidentUpdated = (incident) => {
    setSelectedIncident(incident);
    setIncidents((rows) =>
      rows.map((row) => (row.id === incident.id ? incident : row))
    );
  };

  const exportIncidents = () => {
    window.open(`${API_BASE_URL}/incidents/export`, "_blank");
  };

  useEffect(() => {
    fetchStatus();
    fetchIncidents();
    fetchHealth();
    const statusInterval = setInterval(fetchStatus, status?.running ? 500 : 1000);
    const incidentInterval = setInterval(fetchIncidents, status?.running ? 5000 : 10000);
    const healthInterval = setInterval(fetchHealth, 10000);
    return () => {
      clearInterval(statusInterval);
      clearInterval(incidentInterval);
      clearInterval(healthInterval);
    };
  }, [status?.running]);

  useEffect(() => {
    if (!engineMessage) {
      return undefined;
    }
    const t = setTimeout(() => setEngineMessage(""), 5000);
    return () => clearTimeout(t);
  }, [engineMessage]);

  const riskLevel = status?.riskLevel ?? "LOW";
  const alertType = status?.alertType ?? "NORMAL";
  const riskScore = status?.riskScore ?? 0;
  const entranceReasons = status?.entranceReasons ?? [];
  const storeReasons = status?.storeReasons ?? [];

  const activeAlerts = incidents.filter(
    (i) => Number(i.risk_score || 0) >= 40 && (i.status || "open") !== "resolved"
  );

  const criticalAlerts = incidents.filter((incident) => {
    const zone = (incident.alert_zone || "").toLowerCase();
    const description = (incident.risk_description || "").toLowerCase();
    return (
      zone === "store" ||
      incident.risk_level === "HIGH" ||
      description.includes("inside protected case") ||
      description.includes("hand inside") ||
      description.includes("wrist")
    );
  });

  const identityWarnings = incidents.filter((incident) => {
    const zone = (incident.alert_zone || "").toLowerCase();
    const personId = incident.person_id || "";
    return zone === "entrance" || /^P-\d+$/i.test(personId);
  });

  return (
    <div className={`app ${isMonitorTab ? "app-monitor" : "app-scroll"}`}>
      <TopBar
        health={health}
        status={status}
        staffName={staffName}
        onStaffNameChange={handleStaffNameChange}
        onStart={startEngine}
        onStop={stopEngine}
        isStarting={isStarting}
        isStopping={isStopping}
        staffModeActive={Boolean(status?.staffModeActive)}
        onToggleStaffMode={toggleStaffMode}
        staffModeBusy={staffModeBusy}
      />

      {engineMessage && (
        <div className="toast-message" role="status">
          {engineMessage}
        </div>
      )}

      <AlarmStrip status={status} />

      <nav className="tabs tabs-compact">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={activeTab === tab.id ? "tab active-tab" : "tab"}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <div className="main-content">
        {activeTab === "entrance" && (
          <MonitoringPanel
            title="Entrance Monitoring"
            view="entrance"
            status={status}
            feedSessionKey={feedSessionKey}
            riskLevel={riskLevel}
            alertType={alertType}
            riskScore={riskScore}
            reasons={entranceReasons}
            onDismiss={dismissFalseFlag}
            onMarkStaff={markAsStaff}
            onUnmarkStaff={unmarkStaff}
            dismissingId={dismissingId}
            markingStaffId={markingStaffId}
            metrics={
              <>
                <InfoCard
                  title="Customers"
                  value={status?.customerCount ?? 0}
                  subtitle={status?.staffModeActive ? "Staff mode" : "in frame"}
                />
                <InfoCard
                  title="Crowd"
                  value={status?.crowdLevel ?? "NONE"}
                  subtitle={status?.crowdActive ? `${status.crowdSeconds ?? 0}s` : "clear"}
                />
                <InfoCard
                  title="Flagged"
                  value={status?.flaggedCustomerCount ?? status?.flaggedCount ?? 0}
                  subtitle={status?.alarmLevel ?? "NONE"}
                />
                <InfoCard
                  title="Staff"
                  value={status?.staffCount ?? 0}
                  subtitle={(status?.staffIds ?? []).length ? "marked" : "none"}
                />
                <InfoCard title="People" value={status?.totalPeople ?? 0} />
                <InfoCard title="Alert" value={alertType} />
              </>
            }
          />
        )}

        {activeTab === "store" && (
          <MonitoringPanel
            title="In-Store Monitoring"
            view="store"
            status={status}
            feedSessionKey={feedSessionKey}
            riskLevel={riskLevel}
            alertType={alertType}
            riskScore={riskScore}
            reasons={storeReasons}
            onDismiss={dismissFalseFlag}
            onMarkStaff={markAsStaff}
            onUnmarkStaff={unmarkStaff}
            dismissingId={dismissingId}
            markingStaffId={markingStaffId}
            metrics={
              <>
                <InfoCard
                  title="Customers"
                  value={status?.customerCount ?? 0}
                  subtitle={status?.crowdActive ? "group active" : "counting"}
                />
                <InfoCard
                  title="Crowd"
                  value={status?.crowdLevel ?? "NONE"}
                  subtitle={status?.crowdSeconds ? `${status.crowdSeconds}s` : "—"}
                />
                <InfoCard
                  title="Flagged"
                  value={status?.flaggedCustomerCount ?? 0}
                  subtitle={status?.faceCoveringDetected ? "YES" : "clear"}
                />
                <InfoCard title="Staff" value={status?.staffCount ?? 0} />
                <InfoCard title="People" value={status?.totalPeople ?? 0} />
                <InfoCard title="Risk" value={riskLevel} />
              </>
            }
          />
        )}

        {activeTab === "active" && (
          <IncidentsWithEvidence
            title="Active Alerts"
            incidents={activeAlerts}
            emptyText="No active alerts right now."
            selectedIncident={selectedIncident}
            onSelect={setSelectedIncident}
            onUpdated={handleIncidentUpdated}
          />
        )}

        {activeTab === "critical" && (
          <IncidentsWithEvidence
            title="Critical Alerts"
            incidents={criticalAlerts}
            emptyText="No critical alerts logged."
            selectedIncident={selectedIncident}
            onSelect={setSelectedIncident}
            onUpdated={handleIncidentUpdated}
          />
        )}

        {activeTab === "identity" && (
          <IncidentsWithEvidence
            title="Identity Warnings"
            incidents={identityWarnings}
            emptyText="No identity warnings logged."
            selectedIncident={selectedIncident}
            onSelect={setSelectedIncident}
            onUpdated={handleIncidentUpdated}
          />
        )}

        {activeTab === "incidents" && (
          <IncidentsWithEvidence
            title="Incident Evidence Log"
            incidents={incidents}
            emptyText="No incidents logged yet."
            selectedIncident={selectedIncident}
            onSelect={setSelectedIncident}
            onUpdated={handleIncidentUpdated}
            onExport={exportIncidents}
          />
        )}
      </div>
    </div>
  );
}

export default App;
