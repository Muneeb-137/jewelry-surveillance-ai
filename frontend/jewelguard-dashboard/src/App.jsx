import { useEffect, useState } from "react";
import axios from "axios";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

function InfoCard({ title, value, subtitle }) {
  return (
    <div className="info-card">
      <p className="info-title">{title}</p>
      <h2 className="info-value">{value}</h2>
      {subtitle && <p className="info-subtitle">{subtitle}</p>}
    </div>
  );
}

function IncidentCard({ incident, onSelect }) {
  const riskLevel = incident.risk_level || "LOW";

  return (
    <div
      className={`incident-card ${riskLevel === "HIGH" ? "incident-high" : "incident-medium"}`}
      onClick={() => onSelect(incident)}
    >
      <div className="incident-top">
        <h3>{incident.person_id || "Unknown Person"}</h3>
        <span className={`risk-pill risk-${riskLevel.toLowerCase()}`}>
          {riskLevel}
        </span>
      </div>

      <p className="incident-time">{incident.timestamp}</p>

      <p className="incident-desc">
        {incident.risk_description || "No description available."}
      </p>

      <div className="incident-meta">
        <span>Risk: {incident.risk_score}</span>
        <span>Case Activity: {incident.motion_level}</span>
        <span>Wrist: {incident.wrist_near_case ? "YES" : "NO"}</span>
        <span>Mask: {incident.mask_status || "Unknown"}</span>
      </div>
    </div>
  );
}

function IncidentList({ title, incidents, emptyText, onSelect }) {
  return (
    <section className="incident-list-section full-width-section">
      <div className="section-title-row">
        <h2>{title}</h2>
        <span className="incident-count">{incidents.length} records</span>
      </div>

      {incidents.length > 0 ? (
        <div className="incident-list alert-list-grid">
          {incidents.map((incident) => (
            <IncidentCard
              key={incident.id}
              incident={incident}
              onSelect={onSelect}
            />
          ))}
        </div>
      ) : (
        <div className="empty-box">{emptyText}</div>
      )}
    </section>
  );
}

function EvidenceDetails({ selectedIncident }) {
  if (!selectedIncident) {
    return (
      <section className="incident-detail-section">
        <h2>Evidence Details</h2>
        <div className="empty-box">
          Select an incident to view full description and screenshot.
        </div>
      </section>
    );
  }

  const riskLevel = selectedIncident.risk_level || "LOW";

  return (
    <section className="incident-detail-section">
      <h2>Evidence Details</h2>

      <div className="detail-card">
        <div className="incident-top">
          <h3>{selectedIncident.person_id}</h3>
          <span className={`risk-pill risk-${riskLevel.toLowerCase()}`}>
            {riskLevel}
          </span>
        </div>

        <p className="incident-time">{selectedIncident.timestamp}</p>

        <div className="detail-grid">
          <InfoCard title="Risk Score" value={selectedIncident.risk_score} />
          <InfoCard title="Mask Status" value={selectedIncident.mask_status} />
          <InfoCard title="Case Activity" value={selectedIncident.motion_level} />
          <InfoCard
            title="Wrist Near Case"
            value={selectedIncident.wrist_near_case ? "YES" : "NO"}
          />
          <InfoCard title="People Near Case" value={selectedIncident.people_near_case} />
          <InfoCard title="Loitering" value={`${selectedIncident.loitering_seconds}s`} />
        </div>

        <div className="full-description">
          <h3>Full Risk Description</h3>
          <p>{selectedIncident.risk_description}</p>
        </div>

        {selectedIncident.screenshot_path && (
          <div className="screenshot-box">
            <h3>Evidence Screenshot</h3>
            <img
              src={`${API_BASE_URL}/screenshot?path=${encodeURIComponent(
                selectedIncident.screenshot_path
              )}`}
              alt="Incident Screenshot"
            />
          </div>
        )}
      </div>
    </section>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("entrance");
  const [status, setStatus] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [engineMessage, setEngineMessage] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);

  const tabs = [
    { id: "entrance", label: "Entrance Monitoring" },
    { id: "store", label: "In-Store Monitoring" },
    { id: "active", label: "Active Alerts" },
    { id: "critical", label: "Critical Alerts" },
    { id: "identity", label: "Identity Warnings" },
    { id: "incidents", label: "Incident Evidence Log" },
  ];

  const fetchStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/status`);
      setStatus(response.data);
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
      const response = await axios.post(`${API_BASE_URL}/start`);
      setEngineMessage(response.data.message || "Vision engine started");
      await fetchStatus();
    } catch (error) {
      console.error("Start error:", error);
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
      const response = await axios.post(`${API_BASE_URL}/stop`);
      setEngineMessage(response.data.message || "Vision engine stopped");
      await fetchStatus();
    } catch (error) {
      console.error("Stop error:", error);
      setEngineMessage("Failed to stop vision engine");
    } finally {
      setIsStopping(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    fetchIncidents();

    const interval = setInterval(() => {
      fetchStatus();
      fetchIncidents();
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const riskScore = status?.riskScore ?? 0;
  const riskLevel = status?.riskLevel ?? "LOW";
  const alertType = status?.alertType ?? "NORMAL";
  const reasons = status?.reasons ?? [];
  const entranceReasons = reasons.filter((reason) => {
  const r = reason.toLowerCase();

  const isStoreReason =
    r.includes("jewelry display") ||
    r.includes("display case") ||
    r.includes("jewelry case") ||
    r.includes("wrist") ||
    r.includes("hand") ||
    r.includes("case activity") ||
    r.includes("protected case") ||
    r.includes("loitering");

  return !isStoreReason;
});

const storeReasons = reasons.filter((reason) => {
  const r = reason.toLowerCase();

  return (
    r.includes("jewelry display") ||
    r.includes("display case") ||
    r.includes("jewelry case") ||
    r.includes("wrist") ||
    r.includes("hand") ||
    r.includes("case activity") ||
    r.includes("protected case") ||
    r.includes("loitering")
  );
});

  const activeAlerts = incidents.filter((incident) => Number(incident.risk_score || 0) >= 40);

  const criticalAlerts = incidents.filter((incident) => {
    const description = (incident.risk_description || "").toLowerCase();

    return (
      incident.risk_level === "HIGH" ||
      description.includes("critical") ||
      description.includes("inside protected case") ||
      description.includes("hand inside")
    );
  });

  const identityWarnings = incidents.filter((incident) => {
    const description = (incident.risk_description || "").toLowerCase();
    const maskStatus = (incident.mask_status || "").toLowerCase();

    return (
      description.includes("identity") ||
      description.includes("face covering") ||
      description.includes("reduced identity") ||
      maskStatus.includes("masked")
    );
  });

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>JewelGuard AI</h1>
          <p>Explainable risk monitoring for jewelry retail</p>
        </div>

        <div className="controls">
          <button
            className="start-button"
            onClick={startEngine}
            disabled={isStarting}
          >
            {isStarting ? "Starting..." : "Start"}
          </button>

          <button
            className="stop-button"
            onClick={stopEngine}
            disabled={isStopping}
          >
            {isStopping ? "Stopping..." : "Stop"}
          </button>
        </div>
      </header>

      {engineMessage && <div className="message">{engineMessage}</div>}

      <nav className="tabs">
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

      {activeTab === "entrance" && (
        <main className="main-layout compact-dashboard">
          <section className="camera-section">
            <div className="section-title-row">
              <h2>Entrance Monitoring</h2>
              <span className={status?.running ? "live-badge" : "offline-badge"}>
                {status?.running ? "LIVE" : "OFFLINE"}
              </span>
            </div>

            <div className="camera-frame">
              <img
                className="camera-feed"
                src={`${API_BASE_URL}/frame_stream`}
                alt="Entrance Monitoring Feed"
              />
            </div>
          </section>

          <section className="risk-section compact-side-panel">
            <h2>Entrance Risk</h2>

            <div className={`risk-box risk-${riskLevel.toLowerCase()}`}>
              <p>{alertType}</p>
              <h1>{riskScore}</h1>
              <h2>{riskLevel}</h2>
            </div>

            <div className="compact-grid">
              <InfoCard
                title="Face Covering"
                value={status?.faceCoveringDetected ? "DETECTED" : "CLEAR"}
                subtitle={status?.maskStatus ?? "Unknown"}
              />
              <InfoCard
                title="Entrance Speed"
                value={`${Math.round(status?.entranceSpeed ?? 0)} px/s`}
                subtitle={status?.entranceFastApproach ? "Fast Approach" : "Normal"}
              />
              <InfoCard
                title="Person ID"
                value={status?.currentPersonId ?? "None"}
              />
              <InfoCard
                title="Alert Type"
                value={alertType}
              />
              <InfoCard
                title="Mask Confidence"
                value={status?.maskConfidence ?? 0}
              />
              <InfoCard
                title="People"
                value={status?.totalPeople ?? 0}
              />
            </div>

            <div className="reasons-box">
              <h3>Entrance Alert Reasons</h3>

              {entranceReasons.length > 0 ? (
                <ul>
                  {entranceReasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              ) : (
                <p>No entrance warning active.</p>
              )}
            </div>
          </section>
        </main>
      )}

      {activeTab === "store" && (
        <main className="main-layout compact-dashboard">
          <section className="camera-section">
            <div className="section-title-row">
              <h2>In-Store Monitoring</h2>
              <span className={status?.running ? "live-badge" : "offline-badge"}>
                {status?.running ? "LIVE" : "OFFLINE"}
              </span>
            </div>

            <div className="camera-frame">
              <img
                className="camera-feed"
                src={`${API_BASE_URL}/frame_stream`}
                alt="In-Store Monitoring Feed"
              />
            </div>
          </section>

          <section className="risk-section compact-side-panel">
            <h2>Store Risk</h2>

            <div className={`risk-box risk-${riskLevel.toLowerCase()}`}>
              <p>{alertType}</p>
              <h1>{riskScore}</h1>
              <h2>{riskLevel}</h2>
            </div>

            <div className="compact-grid">
              <InfoCard title="Person ID" value={status?.currentPersonId ?? "None"} />
              <InfoCard title="People Near Case" value={status?.peopleNearCase ?? 0} />
              <InfoCard title="Wrist Near Case" value={status?.wristNearCase ? "YES" : "NO"} />
              <InfoCard title="Wrist Inside Case" value={status?.wristInsideCase ? "YES" : "NO"} />
              <InfoCard
                title="Case Activity"
                value={status?.caseActivityLevel ?? status?.motionLevel ?? "NONE"}
                subtitle={`${Number(status?.caseActivityScore ?? status?.motionScore ?? 0).toFixed(2)}%`}
              />
              <InfoCard title="Loitering" value={`${status?.loiteringSeconds ?? 0}s`} />
              <InfoCard title="Repeated Activity" value={status?.repeatedHighMotion ?? 0} />
              <InfoCard title="Face Covering" value={status?.faceCoveringDetected ? "YES" : "NO"} />
              <InfoCard title="Risk Level" value={riskLevel} />
            </div>

            <div className="reasons-box">
              <h3>In-Store Alert Reasons</h3>

              {storeReasons.length > 0 ? (
                <ul>
                  {storeReasons.map((reason, index) => (
                    <li key={index}>{reason}</li>
                  ))}
                </ul>
              ) : (
                <p>No in-store warning active.</p>
              )}
            </div>
          </section>
        </main>
      )}

      {activeTab === "active" && (
        <main className="incident-layout">
          <IncidentList
            title="Active Alerts"
            incidents={activeAlerts}
            emptyText="No active alerts recorded yet."
            onSelect={setSelectedIncident}
          />
          <EvidenceDetails selectedIncident={selectedIncident} />
        </main>
      )}

      {activeTab === "critical" && (
        <main className="incident-layout">
          <IncidentList
            title="Critical Alerts"
            incidents={criticalAlerts}
            emptyText="No critical alerts recorded yet."
            onSelect={setSelectedIncident}
          />
          <EvidenceDetails selectedIncident={selectedIncident} />
        </main>
      )}

      {activeTab === "identity" && (
        <main className="incident-layout">
          <IncidentList
            title="Identity Warnings"
            incidents={identityWarnings}
            emptyText="No identity warnings recorded yet."
            onSelect={setSelectedIncident}
          />
          <EvidenceDetails selectedIncident={selectedIncident} />
        </main>
      )}

      {activeTab === "incidents" && (
        <main className="incident-layout">
          <IncidentList
            title="Incident Evidence Log"
            incidents={incidents}
            emptyText="No incidents recorded yet. Medium/high events will appear here."
            onSelect={setSelectedIncident}
          />
          <EvidenceDetails selectedIncident={selectedIncident} />
        </main>
      )}
    </div>
  );
}

export default App;
