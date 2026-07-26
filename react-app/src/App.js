import React, { useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import HotspotMap from './modules/HotspotMap';
import AnomalyDetection from './modules/AnomalyDetection';
import PatternTrends from './modules/PatternTrends';
import PredictiveRisk from './modules/PredictiveRisk';

const MODULES = {
  hotspots: HotspotMap,
  anomalies: AnomalyDetection,
  patterns: PatternTrends,
  risk: PredictiveRisk,
};

export default function App() {
  const [active, setActive] = useState('hotspots');
  const ActiveModule = MODULES[active];

  return (
    <div className="app-shell">
      <TopBar active={active} />
      <Sidebar active={active} onSelect={setActive} />
      <main className="app-main">
        <ActiveModule />
      </main>
    </div>
  );
}
