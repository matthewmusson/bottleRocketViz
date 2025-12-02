import React, { useState, useEffect, useMemo, useCallback } from 'react';

// Physical Constants (0.8L Bottle Rocket - from constants.py)
const CONSTANTS = {
    g: 9.81,                    // m/s²
    rho_air: 1.225,             // kg/m³
    rho_water: 1000,            // kg/m³
    P_atm: 101325,              // Pa
    gamma: 1.4,                 // Adiabatic exponent
    tank_volume: 8e-4,          // 0.8L in m³
    d_bottle: 0.075,            // 7.5 cm
    d_nozzle: 0.026,            // 2.6 cm
    mass_empty: 0.0765,         // 76.5 g
};

const A_bottle = Math.PI * (CONSTANTS.d_bottle / 2) ** 2;
const A_nozzle = Math.PI * (CONSTANTS.d_nozzle / 2) ** 2;

// RK4 Integrator
function rk4Step(odes, t, y, dt, params) {
    const k1 = odes(t, y, params);
    const k2 = odes(t + dt / 2, y.map((yi, i) => yi + dt / 2 * k1[i]), params);
    const k3 = odes(t + dt / 2, y.map((yi, i) => yi + dt / 2 * k2[i]), params);
    const k4 = odes(t + dt, y.map((yi, i) => yi + dt * k3[i]), params);
    return y.map((yi, i) => yi + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

// Rocket ODEs: y = [v, h, m, V_water]
function rocketODEs(t, y, params) {
    const { V_air_0, C_d, P_0_abs } = params;
    let [v, h, m, V_water] = y;

    const drag_accel = -0.5 * CONSTANTS.rho_air * C_d * A_bottle * v * Math.abs(v) / m;

    if (V_water <= 0) {
        return [-CONSTANTS.g + drag_accel, v, 0, 0];
    }

    const V_air = CONSTANTS.tank_volume - V_water;
    const P_1 = P_0_abs * Math.pow(V_air_0 / V_air, CONSTANTS.gamma);
    const delta_P = P_1 - CONSTANTS.P_atm;

    if (delta_P <= 0) {
        return [-CONSTANTS.g + drag_accel, v, 0, 0];
    }

    const denom = CONSTANTS.rho_water * (1 - Math.pow(CONSTANTS.d_nozzle / CONSTANTS.d_bottle, 4));
    const v_e = Math.sqrt(2 * delta_P / denom);

    const dV_water_dt = -A_nozzle * v_e;
    const dm_dt = CONSTANTS.rho_water * dV_water_dt;
    const thrust_accel = (CONSTANTS.rho_water * A_nozzle * v_e * v_e) / m;
    const dv_dt = thrust_accel - CONSTANTS.g + drag_accel;

    return [dv_dt, v, dm_dt, dV_water_dt];
}

// Run simulation for given parameters
function runSimulation(fillRatio, C_d, pressurePSI) {
    const V_water_0 = fillRatio * CONSTANTS.tank_volume;
    const V_air_0 = CONSTANTS.tank_volume - V_water_0;
    const P_0_abs = pressurePSI * 6894.76 + CONSTANTS.P_atm;

    const m_0 = CONSTANTS.mass_empty + CONSTANTS.rho_water * V_water_0;
    let y = [0, 0, m_0, V_water_0];
    let t = 0;
    const dt = 0.002;
    const params = { V_air_0, C_d, P_0_abs };

    const trajectory = [{ t: 0, h: 0, v: 0 }];
    let maxH = 0;
    let maxHTime = 0;
    let burnoutTime = null;
    let burnoutAlt = null;
    let burnoutVel = null;

    while (t < 15 && y[1] >= 0) {
        y = rk4Step(rocketODEs, t, y, dt, params);
        t += dt;

        if (y[3] <= 0 && burnoutTime === null) {
            burnoutTime = t;
            burnoutAlt = y[1];
            burnoutVel = y[0];
        }

        if (y[1] > maxH) {
            maxH = y[1];
            maxHTime = t;
        }

        if (Math.floor(t * 200) > Math.floor((t - dt) * 200)) {
            trajectory.push({ t, h: Math.max(0, y[1]), v: y[0] });
        }
    }

    return { trajectory, maxH, maxHTime, burnoutTime, burnoutAlt, burnoutVel };
}

// Find optimal fill ratio
function findOptimal(C_d, pressurePSI) {
    let bestRatio = 0.33;
    let bestH = 0;
    const data = [];

    for (let r = 0.05; r <= 0.95; r += 0.02) {
        const { maxH } = runSimulation(r, C_d, pressurePSI);
        data.push({ ratio: r, maxH });
        if (maxH > bestH) {
            bestH = maxH;
            bestRatio = r;
        }
    }

    return { bestRatio, bestH, data };
}

// Color palette for trajectories
const COLORS = [
    '#00ffaa', '#ff6b6b', '#4ecdc4', '#ffe66d',
    '#a855f7', '#22d3ee', '#fb923c', '#f472b6'
];

export default function BottleRocketSim() {
    const [fillRatio, setFillRatio] = useState(0.33);
    const [dragCoeff, setDragCoeff] = useState(0.4);
    const [pressure, setPressure] = useState(60);
    const [showOptimal, setShowOptimal] = useState(true);
    const [compareMode, setCompareMode] = useState(false);
    const [compareRatios, setCompareRatios] = useState([0.25, 0.33, 0.5, 0.67]);

    // Main simulation
    const simResult = useMemo(() =>
        runSimulation(fillRatio, dragCoeff, pressure),
        [fillRatio, dragCoeff, pressure]
    );

    // Optimal calculation
    const optimal = useMemo(() =>
        findOptimal(dragCoeff, pressure),
        [dragCoeff, pressure]
    );

    // Compare trajectories
    const compareResults = useMemo(() => {
        if (!compareMode) return [];
        return compareRatios.map(r => ({
            ratio: r,
            ...runSimulation(r, dragCoeff, pressure)
        }));
    }, [compareMode, compareRatios, dragCoeff, pressure]);

    // SVG dimensions
    const width = 700;
    const height = 400;
    const padding = { top: 30, right: 30, bottom: 50, left: 60 };
    const plotW = width - padding.left - padding.right;
    const plotH = height - padding.top - padding.bottom;

    // Scales
    const maxT = compareMode
        ? Math.max(...compareResults.map(r => r.trajectory[r.trajectory.length - 1]?.t || 5), 5)
        : Math.max(simResult.trajectory[simResult.trajectory.length - 1]?.t || 5, 5);
    const maxH = compareMode
        ? Math.max(...compareResults.map(r => r.maxH), 10) * 1.1
        : Math.max(simResult.maxH * 1.1, 10);

    const scaleX = (t) => padding.left + (t / maxT) * plotW;
    const scaleY = (h) => padding.top + plotH - (h / maxH) * plotH;

    // Generate path
    const pathD = (trajectory) => {
        if (trajectory.length === 0) return '';
        return trajectory.map((p, i) =>
            `${i === 0 ? 'M' : 'L'} ${scaleX(p.t)} ${scaleY(p.h)}`
        ).join(' ');
    };

    // Grid lines
    const xTicks = Array.from({ length: 6 }, (_, i) => (i * maxT) / 5);
    const yTicks = Array.from({ length: 6 }, (_, i) => (i * maxH) / 5);

    return (
        <div style={{
            minHeight: '100vh',
            background: 'linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%)',
            color: '#e0e0e0',
            fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
            padding: '24px',
        }}>
            {/* Header */}
            <div style={{
                borderBottom: '1px solid #333',
                paddingBottom: '16px',
                marginBottom: '24px',
            }}>
                <h1 style={{
                    fontSize: '28px',
                    fontWeight: 300,
                    letterSpacing: '4px',
                    margin: 0,
                    background: 'linear-gradient(90deg, #00ffaa, #4ecdc4)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                }}>
                    BOTTLE ROCKET TRAJECTORY SIMULATOR
                </h1>
                <p style={{
                    fontSize: '11px',
                    color: '#666',
                    marginTop: '8px',
                    letterSpacing: '2px'
                }}>
                    BERNOULLI PROPULSION • ADIABATIC EXPANSION • RK4 INTEGRATION
                </p>
            </div>

            <div style={{ display: 'flex', gap: '24px', flexWrap: 'wrap' }}>
                {/* Controls Panel */}
                <div style={{
                    background: 'rgba(20, 20, 30, 0.8)',
                    border: '1px solid #2a2a3a',
                    borderRadius: '8px',
                    padding: '20px',
                    width: '280px',
                    flexShrink: 0,
                }}>
                    <div style={{
                        fontSize: '10px',
                        letterSpacing: '2px',
                        color: '#00ffaa',
                        marginBottom: '20px',
                        borderBottom: '1px solid #2a2a3a',
                        paddingBottom: '8px'
                    }}>
                        PARAMETERS
                    </div>

                    {/* Fill Ratio */}
                    <div style={{ marginBottom: '20px' }}>
                        <label style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            fontSize: '11px',
                            marginBottom: '8px'
                        }}>
                            <span>FILL RATIO</span>
                            <span style={{ color: '#00ffaa' }}>{(fillRatio * 100).toFixed(0)}%</span>
                        </label>
                        <input
                            type="range"
                            min="0.05"
                            max="0.95"
                            step="0.01"
                            value={fillRatio}
                            onChange={(e) => setFillRatio(parseFloat(e.target.value))}
                            style={{ width: '100%', accentColor: '#00ffaa' }}
                        />
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            fontSize: '9px',
                            color: '#555',
                            marginTop: '4px'
                        }}>
                            <span>5%</span>
                            <span>95%</span>
                        </div>
                    </div>

                    {/* Drag Coefficient */}
                    <div style={{ marginBottom: '20px' }}>
                        <label style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            fontSize: '11px',
                            marginBottom: '8px'
                        }}>
                            <span>DRAG COEFFICIENT (Cᴅ)</span>
                            <span style={{ color: '#4ecdc4' }}>{dragCoeff.toFixed(2)}</span>
                        </label>
                        <input
                            type="range"
                            min="0.1"
                            max="1.0"
                            step="0.01"
                            value={dragCoeff}
                            onChange={(e) => setDragCoeff(parseFloat(e.target.value))}
                            style={{ width: '100%', accentColor: '#4ecdc4' }}
                        />
                    </div>

                    {/* Pressure */}
                    <div style={{ marginBottom: '20px' }}>
                        <label style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            fontSize: '11px',
                            marginBottom: '8px'
                        }}>
                            <span>PRESSURE</span>
                            <span style={{ color: '#ffe66d' }}>{pressure} PSI</span>
                        </label>
                        <input
                            type="range"
                            min="20"
                            max="120"
                            step="5"
                            value={pressure}
                            onChange={(e) => setPressure(parseFloat(e.target.value))}
                            style={{ width: '100%', accentColor: '#ffe66d' }}
                        />
                    </div>

                    {/* Mode Toggle */}
                    <div style={{
                        marginTop: '24px',
                        paddingTop: '16px',
                        borderTop: '1px solid #2a2a3a'
                    }}>
                        <button
                            onClick={() => setCompareMode(!compareMode)}
                            style={{
                                width: '100%',
                                padding: '10px',
                                background: compareMode ? '#00ffaa' : 'transparent',
                                border: '1px solid #00ffaa',
                                color: compareMode ? '#0a0a0f' : '#00ffaa',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '10px',
                                letterSpacing: '2px',
                                fontFamily: 'inherit',
                                transition: 'all 0.2s'
                            }}
                        >
                            {compareMode ? '◉ COMPARE MODE' : '○ SINGLE MODE'}
                        </button>
                    </div>

                    {/* Optimal Indicator */}
                    <div style={{
                        marginTop: '20px',
                        padding: '12px',
                        background: 'rgba(0, 255, 170, 0.1)',
                        border: '1px solid rgba(0, 255, 170, 0.3)',
                        borderRadius: '4px',
                    }}>
                        <div style={{
                            fontSize: '9px',
                            color: '#666',
                            letterSpacing: '1px',
                            marginBottom: '8px'
                        }}>
                            OPTIMAL FILL RATIO
                        </div>
                        <div style={{
                            fontSize: '24px',
                            color: '#00ffaa',
                            fontWeight: 600
                        }}>
                            {(optimal.bestRatio * 100).toFixed(1)}%
                        </div>
                        <div style={{
                            fontSize: '10px',
                            color: '#888',
                            marginTop: '4px'
                        }}>
                            Max altitude: {optimal.bestH.toFixed(1)} m
                        </div>
                    </div>
                </div>

                {/* Main Visualization */}
                <div style={{ flex: 1, minWidth: '500px' }}>
                    {/* Trajectory Plot */}
                    <div style={{
                        background: 'rgba(10, 10, 15, 0.9)',
                        border: '1px solid #2a2a3a',
                        borderRadius: '8px',
                        padding: '16px',
                        marginBottom: '16px'
                    }}>
                        <svg width={width} height={height} style={{ display: 'block' }}>
                            {/* Grid */}
                            <defs>
                                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                                    <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1a1a2a" strokeWidth="0.5" />
                                </pattern>
                            </defs>
                            <rect
                                x={padding.left}
                                y={padding.top}
                                width={plotW}
                                height={plotH}
                                fill="url(#grid)"
                            />

                            {/* Axis lines */}
                            {xTicks.map((t, i) => (
                                <g key={`x-${i}`}>
                                    <line
                                        x1={scaleX(t)} y1={padding.top}
                                        x2={scaleX(t)} y2={padding.top + plotH}
                                        stroke="#2a2a3a" strokeWidth="1"
                                    />
                                    <text
                                        x={scaleX(t)} y={height - 20}
                                        fill="#666" fontSize="10" textAnchor="middle"
                                    >
                                        {t.toFixed(1)}s
                                    </text>
                                </g>
                            ))}
                            {yTicks.map((h, i) => (
                                <g key={`y-${i}`}>
                                    <line
                                        x1={padding.left} y1={scaleY(h)}
                                        x2={padding.left + plotW} y2={scaleY(h)}
                                        stroke="#2a2a3a" strokeWidth="1"
                                    />
                                    <text
                                        x={padding.left - 10} y={scaleY(h) + 4}
                                        fill="#666" fontSize="10" textAnchor="end"
                                    >
                                        {h.toFixed(0)}m
                                    </text>
                                </g>
                            ))}

                            {/* Axis labels */}
                            <text
                                x={width / 2} y={height - 5}
                                fill="#888" fontSize="11" textAnchor="middle"
                            >
                                Time (s)
                            </text>
                            <text
                                x={15} y={height / 2}
                                fill="#888" fontSize="11" textAnchor="middle"
                                transform={`rotate(-90, 15, ${height / 2})`}
                            >
                                Altitude (m)
                            </text>

                            {/* Trajectories */}
                            {compareMode ? (
                                compareResults.map((result, i) => (
                                    <g key={i}>
                                        <path
                                            d={pathD(result.trajectory)}
                                            fill="none"
                                            stroke={COLORS[i % COLORS.length]}
                                            strokeWidth="2"
                                            opacity={0.9}
                                        />
                                        {/* Peak marker - now using tracked maxHTime */}
                                        <circle
                                            cx={scaleX(result.maxHTime)}
                                            cy={scaleY(result.maxH)}
                                            r="4"
                                            fill={COLORS[i % COLORS.length]}
                                        />
                                    </g>
                                ))
                            ) : (
                                <g>
                                    <path
                                        d={pathD(simResult.trajectory)}
                                        fill="none"
                                        stroke="#00ffaa"
                                        strokeWidth="2.5"
                                        strokeLinecap="round"
                                    />
                                    {/* Peak marker - now using tracked maxHTime */}
                                    <circle
                                        cx={scaleX(simResult.maxHTime)}
                                        cy={scaleY(simResult.maxH)}
                                        r="5"
                                        fill="#00ffaa"
                                    />
                                    {/* Burnout marker */}
                                    {simResult.burnoutTime && (
                                        <g>
                                            <circle
                                                cx={scaleX(simResult.burnoutTime)}
                                                cy={scaleY(simResult.burnoutAlt)}
                                                r="4"
                                                fill="none"
                                                stroke="#ff6b6b"
                                                strokeWidth="2"
                                            />
                                            <text
                                                x={scaleX(simResult.burnoutTime) + 8}
                                                y={scaleY(simResult.burnoutAlt) - 8}
                                                fill="#ff6b6b"
                                                fontSize="9"
                                            >
                                                BURNOUT
                                            </text>
                                        </g>
                                    )}
                                </g>
                            )}

                            {/* Optimal line indicator */}
                            {showOptimal && !compareMode && (
                                <g>
                                    <line
                                        x1={padding.left}
                                        y1={scaleY(optimal.bestH)}
                                        x2={padding.left + plotW}
                                        y2={scaleY(optimal.bestH)}
                                        stroke="#00ffaa"
                                        strokeWidth="1"
                                        strokeDasharray="4,4"
                                        opacity="0.5"
                                    />
                                    <text
                                        x={padding.left + plotW - 5}
                                        y={scaleY(optimal.bestH) - 5}
                                        fill="#00ffaa"
                                        fontSize="9"
                                        textAnchor="end"
                                        opacity="0.7"
                                    >
                                        OPTIMAL: {optimal.bestH.toFixed(1)}m
                                    </text>
                                </g>
                            )}
                        </svg>

                        {/* Legend for compare mode */}
                        {compareMode && (
                            <div style={{
                                display: 'flex',
                                gap: '16px',
                                marginTop: '12px',
                                flexWrap: 'wrap'
                            }}>
                                {compareResults.map((result, i) => (
                                    <div key={i} style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: '6px',
                                        fontSize: '10px'
                                    }}>
                                        <div style={{
                                            width: '12px',
                                            height: '3px',
                                            background: COLORS[i % COLORS.length]
                                        }} />
                                        <span style={{ color: COLORS[i % COLORS.length] }}>
                                            {(result.ratio * 100).toFixed(0)}%
                                        </span>
                                        <span style={{ color: '#666' }}>
                                            ({result.maxH.toFixed(1)}m)
                                        </span>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>

                    {/* Stats Panel */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                        gap: '12px',
                    }}>
                        <StatBox
                            label="MAX ALTITUDE"
                            value={`${simResult.maxH.toFixed(1)} m`}
                            color="#00ffaa"
                        />
                        <StatBox
                            label="BURNOUT TIME"
                            value={simResult.burnoutTime ? `${(simResult.burnoutTime * 1000).toFixed(0)} ms` : '—'}
                            color="#ff6b6b"
                        />
                        <StatBox
                            label="BURNOUT VELOCITY"
                            value={simResult.burnoutVel ? `${simResult.burnoutVel.toFixed(1)} m/s` : '—'}
                            color="#4ecdc4"
                        />
                        <StatBox
                            label="WATER MASS"
                            value={`${(fillRatio * CONSTANTS.tank_volume * CONSTANTS.rho_water * 1000).toFixed(0)} g`}
                            color="#ffe66d"
                        />
                    </div>

                    {/* Optimization Curve */}
                    <div style={{
                        background: 'rgba(10, 10, 15, 0.9)',
                        border: '1px solid #2a2a3a',
                        borderRadius: '8px',
                        padding: '16px',
                        marginTop: '16px'
                    }}>
                        <div style={{
                            fontSize: '10px',
                            letterSpacing: '2px',
                            color: '#666',
                            marginBottom: '12px'
                        }}>
                            ALTITUDE vs FILL RATIO
                        </div>
                        <svg width={width} height={180}>
                            {/* Optimization curve */}
                            <path
                                d={optimal.data.map((p, i) =>
                                    `${i === 0 ? 'M' : 'L'} ${padding.left + (p.ratio - 0.05) / 0.9 * plotW} ${160 - (p.maxH / optimal.bestH) * 130}`
                                ).join(' ')}
                                fill="none"
                                stroke="#4ecdc4"
                                strokeWidth="2"
                            />
                            {/* Optimal marker */}
                            <circle
                                cx={padding.left + (optimal.bestRatio - 0.05) / 0.9 * plotW}
                                cy={160 - 130}
                                r="6"
                                fill="#00ffaa"
                            />
                            {/* Current position */}
                            <circle
                                cx={padding.left + (fillRatio - 0.05) / 0.9 * plotW}
                                cy={160 - (simResult.maxH / optimal.bestH) * 130}
                                r="5"
                                fill="none"
                                stroke="#fff"
                                strokeWidth="2"
                            />
                            {/* X axis labels */}
                            {[0.1, 0.3, 0.5, 0.7, 0.9].map(r => (
                                <text
                                    key={r}
                                    x={padding.left + (r - 0.05) / 0.9 * plotW}
                                    y={175}
                                    fill="#666"
                                    fontSize="9"
                                    textAnchor="middle"
                                >
                                    {(r * 100).toFixed(0)}%
                                </text>
                            ))}
                        </svg>
                    </div>
                </div>
            </div>

            {/* Physics Notes */}
            <div style={{
                marginTop: '24px',
                padding: '16px',
                background: 'rgba(20, 20, 30, 0.6)',
                border: '1px solid #2a2a3a',
                borderRadius: '8px',
                fontSize: '11px',
                color: '#888',
                lineHeight: 1.6
            }}>
                <span style={{ color: '#00ffaa' }}>MODEL:</span> Bernoulli exhaust velocity with diameter correction •
                Adiabatic air expansion (γ = 1.4) •
                Quadratic drag •
                RK4 integration (dt = 2ms) •
                <span style={{ color: '#ffe66d' }}>0.8L PET bottle</span>
            </div>
        </div>
    );
}

function StatBox({ label, value, color }) {
    return (
        <div style={{
            background: 'rgba(10, 10, 15, 0.9)',
            border: '1px solid #2a2a3a',
            borderRadius: '6px',
            padding: '12px',
        }}>
            <div style={{
                fontSize: '9px',
                color: '#555',
                letterSpacing: '1px',
                marginBottom: '6px'
            }}>
                {label}
            </div>
            <div style={{
                fontSize: '18px',
                color: color,
                fontWeight: 500
            }}>
                {value}
            </div>
        </div>
    );
}
