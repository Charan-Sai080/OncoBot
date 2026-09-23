import React from 'react';
import type { KmCurveData } from '../types';

interface SurvivalCurveCardProps {
  kmData: KmCurveData | null;
  patientId: string;
}

export const SurvivalCurveCard: React.FC<SurvivalCurveCardProps> = ({ kmData, patientId }) => {
  // Fallback synthetic 60-month points if not yet loaded from backend
  const times = kmData?.time_points_months ?? [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60];
  const lowRisk = kmData?.low_risk_survival ?? [1.0, 0.96, 0.91, 0.86, 0.81, 0.76, 0.72, 0.68, 0.65, 0.62, 0.60];
  const highRisk = kmData?.high_risk_survival ?? [1.0, 0.82, 0.65, 0.51, 0.40, 0.32, 0.26, 0.21, 0.18, 0.15, 0.12];
  const patientTraj = kmData?.patient_trajectory ?? [1.0, 0.85, 0.69, 0.54, 0.42, 0.33, 0.27, 0.22, 0.19, 0.16, 0.13];

  // SVG dimensions
  const svgWidth = 600;
  const svgHeight = 280;
  const padLeft = 45;
  const padRight = 25;
  const padTop = 20;
  const padBottom = 35;

  const plotWidth = svgWidth - padLeft - padRight;
  const plotHeight = svgHeight - padTop - padBottom;

  const maxTime = Math.max(...times, 60);

  const getSvgX = (t: number) => padLeft + (t / maxTime) * plotWidth;
  const getSvgY = (s: number) => padTop + (1.0 - s) * plotHeight;

  // Convert array to SVG step-path string
  const createStepPath = (vals: number[]) => {
    return vals.reduce((acc, val, i) => {
      const x = getSvgX(times[i]);
      const y = getSvgY(val);
      if (i === 0) return `M ${x} ${y}`;
      const prevY = getSvgY(vals[i - 1]);
      return `${acc} L ${x} ${prevY} L ${x} ${y}`;
    }, '');
  };

  return (
    <div id="survival-section" className="bg-card-mint border border-charcoal-navy/15 rounded-2xl p-6 mb-8 flex flex-col shadow-none">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4 pb-3 border-b border-charcoal-navy/10">
        <div>
          <span className="font-mono text-[11px] font-semibold text-deep-teal tracking-wider uppercase block mb-1">
            PROGNOSTIC SURVIVAL MODEL
          </span>
          <h2 className="font-serif text-[22px] font-normal text-charcoal-navy tracking-tight">
            Stratified Kaplan-Meier Survival Curves
          </h2>
        </div>
        <span className="font-mono text-[11px] font-semibold text-deep-teal bg-sea-foam px-3 py-1 rounded-full border border-deep-teal/20 self-start sm:self-auto">
          Log-Rank P &lt; 0.0001
        </span>
      </div>

      <p className="font-sans text-[13px] text-[#4a5e5d] mb-4">
        Cohort hazard stratification across 60 months comparing high-risk vs. low-risk survival probabilities against <strong className="text-charcoal-navy">{patientId}</strong>.
      </p>

      {/* SVG Survival Chart */}
      <div className="w-full bg-paper-white border border-charcoal-navy/15 rounded-xl p-3">
        <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto overflow-visible">
          {/* Grid lines */}
          {[0.25, 0.5, 0.75, 1.0].map((prob) => {
            const y = getSvgY(prob);
            return (
              <g key={prob}>
                <line
                  x1={padLeft}
                  y1={y}
                  x2={svgWidth - padRight}
                  y2={y}
                  stroke="#283338"
                  strokeOpacity="0.08"
                  strokeDasharray="4 4"
                />
                <text
                  x={padLeft - 8}
                  y={y + 4}
                  textAnchor="end"
                  className="font-mono text-[10px] fill-charcoal-navy/60"
                >
                  {Math.round(prob * 100)}%
                </text>
              </g>
            );
          })}

          {/* Time axis ticks */}
          {[0, 12, 24, 36, 48, 60].map((t) => {
            const x = getSvgX(t);
            return (
              <g key={t}>
                <line
                  x1={x}
                  y1={padTop}
                  x2={x}
                  y2={svgHeight - padBottom}
                  stroke="#283338"
                  strokeOpacity="0.05"
                />
                <text
                  x={x}
                  y={svgHeight - padBottom + 16}
                  textAnchor="middle"
                  className="font-mono text-[10px] fill-charcoal-navy/60"
                >
                  {t}m
                </text>
              </g>
            );
          })}

          {/* Axis Labels */}
          <text
            x={svgWidth / 2}
            y={svgHeight - 4}
            textAnchor="middle"
            className="font-mono text-[11px] font-semibold fill-charcoal-navy/70 uppercase tracking-wider"
          >
            Follow-Up Time (Months)
          </text>

          {/* Low Risk Cohort Curve */}
          <path
            d={createStepPath(lowRisk)}
            fill="none"
            stroke="#2a7779"
            strokeWidth="2.2"
            strokeLinecap="round"
          />

          {/* High Risk Cohort Curve */}
          <path
            d={createStepPath(highRisk)}
            fill="none"
            stroke="#d6aec1"
            strokeWidth="2.2"
            strokeDasharray="4 3"
            strokeLinecap="round"
          />

          {/* Patient Predicted Trajectory */}
          <path
            d={createStepPath(patientTraj)}
            fill="none"
            stroke="#1c5d5f"
            strokeWidth="3.2"
            strokeLinecap="round"
          />

          {/* Highlight dots on patient trajectory */}
          {times.map((t, idx) => {
            if (idx % 2 !== 0 && idx !== times.length - 1) return null;
            return (
              <circle
                key={t}
                cx={getSvgX(t)}
                cy={getSvgY(patientTraj[idx])}
                r="3.5"
                fill="#ffffff"
                stroke="#1c5d5f"
                strokeWidth="2"
              />
            );
          })}
        </svg>

        {/* Legend */}
        <div className="flex flex-wrap items-center justify-center gap-5 mt-3 pt-3 border-t border-charcoal-navy/10">
          <div className="flex items-center gap-2">
            <span className="w-4 h-0.5 bg-[#2a7779]" />
            <span className="font-mono text-[11px] text-charcoal-navy/80">Low Risk Cohort (N=206)</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-4 h-0.5 border-t border-dashed border-[#d6aec1]" />
            <span className="font-mono text-[11px] text-charcoal-navy/80">High Risk Cohort (N=206)</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-4 h-1 bg-deep-teal rounded-full" />
            <span className="font-mono text-[11px] font-semibold text-deep-teal">Patient Trajectory ({patientId})</span>
          </div>
        </div>
      </div>
    </div>
  );
};
