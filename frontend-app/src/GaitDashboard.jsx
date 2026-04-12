import React, { useMemo, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

export default function GaitDashboard({ richFeatures }) {
  // Added a 4th tab for Summary metrics
  const [activeTab, setActiveTab] = useState('angles');
  const series = richFeatures?.series;

  const chartData = useMemo(() => {
    if (!series) return [];
    
    return series.map((frameData, index) => ({
      frame: index + 1,
      leftKnee: frameData[0],  
      rightKnee: frameData[1], 
      leftHip: frameData[2],   
      rightHip: frameData[3],  
      leftAnkle: frameData[4], 
      rightAnkle: frameData[5],
      trunkLean: frameData[6],
      pelvisWidth: frameData[7],
      ankleDist: frameData[8],
      heelDist: frameData[9], 
      stepWidth: frameData[10]
    }));
  }, [series]);

  if (!series) return <div style={{ opacity: 0.5 }}>No time-series data available.</div>;

  const SummaryCard = ({ title, value, unit }) => (
    <div style={{
      background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: '8px', padding: '12px', flex: 1, minWidth: '120px', textAlign: 'center'
    }}>
      <div style={{ fontSize: '11px', textTransform: 'uppercase', opacity: 0.7, marginBottom: '4px' }}>{title}</div>
      <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#60a5fa' }}>
        {typeof value === 'number' ? value.toFixed(2) : value} <span style={{ fontSize: '12px', color: 'white', opacity: 0.7 }}>{unit}</span>
      </div>
    </div>
  );

  // List of joints to pull into the data table
  const jointStats = [
    { id: 'left_knee', label: 'Left Knee' },
    { id: 'right_knee', label: 'Right Knee' },
    { id: 'left_hip', label: 'Left Hip' },
    { id: 'right_hip', label: 'Right Hip' },
    { id: 'left_ankle', label: 'Left Ankle' },
    { id: 'right_ankle', label: 'Right Ankle' },
    { id: 'trunk_lean', label: 'Trunk Lean' },
  ];

  return (
    <div style={{ marginTop: '20px', padding: '20px', background: 'rgba(0,0,0,0.25)', borderRadius: '16px', border: '1px solid rgba(255,255,255,0.05)' }}>
      <h3 style={{ marginTop: 0, marginBottom: '16px', fontSize: '18px' }}>Advanced Clinical Dashboard</h3>
      
      <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginBottom: '20px' }}>
        <SummaryCard title="Left Knee ROM" value={richFeatures?.left_knee?.rom} unit="°" />
        <SummaryCard title="Right Knee ROM" value={richFeatures?.right_knee?.rom} unit="°" />
        <SummaryCard title="Trunk Lean (Avg)" value={richFeatures?.trunk_lean?.mean} unit="°" />
        <SummaryCard title="Step Variability" value={richFeatures?.step_variability} unit="" />
        <SummaryCard title="Pelvic Sway" value={richFeatures?.pelvis_sway} unit="" />
      </div>

      <div style={{ display: 'flex', gap: '8px', marginBottom: '16px', flexWrap: 'wrap' }}>
        {['angles', 'dynamics', 'posture', 'clinical_metrics'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            style={{
              padding: '8px 16px',
              borderRadius: '6px',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 'bold',
              backgroundColor: activeTab === tab ? '#3b82f6' : 'rgba(255,255,255,0.1)',
              color: activeTab === tab ? 'white' : 'rgba(255,255,255,0.7)',
            }}
          >
            {tab === 'angles' ? 'Joint Angles' : 
             tab === 'dynamics' ? 'Step Dynamics' : 
             tab === 'posture' ? 'Posture' : 'Clinical Metrics'}
          </button>
        ))}
      </div>
      
      <div style={{ width: '100%', minHeight: 350, background: 'rgba(0,0,0,0.2)', padding: '15px', borderRadius: '12px' }}>
        
        {/* --- TIME SERIES CHARTS --- */}
        {activeTab !== 'clinical_metrics' && (
          <ResponsiveContainer width="100%" height={350}>
            <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
              <XAxis dataKey="frame" stroke="rgba(255,255,255,0.4)" tick={{fill: 'rgba(255,255,255,0.6)'}} />
              <YAxis domain={['auto', 'auto']} stroke="rgba(255,255,255,0.4)" tick={{fill: 'rgba(255,255,255,0.6)'}} />
              <Tooltip 
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }}
                itemStyle={{ color: '#fff', fontWeight: 'bold' }}
              />
              <Legend wrapperStyle={{ paddingTop: '10px' }} />
              
              {activeTab === 'angles' && (
                <>
                  <Line type="monotone" dataKey="leftKnee" name="Left Knee" stroke="#60a5fa" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="rightKnee" name="Right Knee" stroke="#f472b6" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="leftHip" name="Left Hip" stroke="#34d399" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                  <Line type="monotone" dataKey="rightHip" name="Right Hip" stroke="#fbbf24" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                  <Line type="monotone" dataKey="leftAnkle" name="Left Ankle" stroke="#a78bfa" strokeWidth={1.5} dot={false} />
                  <Line type="monotone" dataKey="rightAnkle" name="Right Ankle" stroke="#f87171" strokeWidth={1.5} dot={false} />
                </>
              )}

              {activeTab === 'dynamics' && (
                <>
                  <Line type="monotone" dataKey="heelDist" name="Step Length Proxy" stroke="#34d399" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="stepWidth" name="Step Width Proxy" stroke="#fbbf24" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="ankleDist" name="Ankle Distance" stroke="#a78bfa" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                </>
              )}

              {activeTab === 'posture' && (
                <>
                  <Line type="monotone" dataKey="trunkLean" name="Trunk Lean Angle" stroke="#f87171" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="pelvisWidth" name="Pelvis Width Proxy" stroke="#60a5fa" strokeWidth={2} strokeDasharray="4 4" dot={false} />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        )}

        {/* --- CLINICAL METRICS TABLE --- */}
        {activeTab === 'clinical_metrics' && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.2)' }}>
                  <th style={{ padding: '12px' }}>Joint / Metric</th>
                  <th style={{ padding: '12px' }}>ROM (Range)</th>
                  <th style={{ padding: '12px' }}>Max Ext/Flex</th>
                  <th style={{ padding: '12px' }}>Min Ext/Flex</th>
                  <th style={{ padding: '12px' }}>Mean Angle</th>
                </tr>
              </thead>
              <tbody>
                {jointStats.map(stat => {
                  const data = richFeatures[stat.id];
                  if (!data) return null;
                  return (
                    <tr key={stat.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                      <td style={{ padding: '12px', fontWeight: 'bold' }}>{stat.label}</td>
                      <td style={{ padding: '12px', color: '#60a5fa', fontWeight: 'bold' }}>{data.rom?.toFixed(2)}°</td>
                      <td style={{ padding: '12px', color: '#f87171' }}>{data.max?.toFixed(2)}°</td>
                      <td style={{ padding: '12px', color: '#34d399' }}>{data.min?.toFixed(2)}°</td>
                      <td style={{ padding: '12px', opacity: 0.8 }}>{data.mean?.toFixed(2)}°</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

      </div>
    </div>
  );
}