import React, { useState } from 'react';
import { ArrowLeft, ChevronDown, FlaskConical, History, RotateCcw, Upload } from 'lucide-react';
import type { CaseHistoryEntry } from '../types';

interface CaseHistoryPanelProps {
  entries: CaseHistoryEntry[];
  onBack: () => void;
  onRerun: (entry: CaseHistoryEntry) => void;
}

export const CaseHistoryPanel: React.FC<CaseHistoryPanelProps> = ({ entries, onBack, onRerun }) => {
  // Track open/collapsed state per case item (first entry expanded by default)
  const [expandedIds, setExpandedIds] = useState<Record<string, boolean>>(() => {
    if (entries.length > 0) {
      return { [entries[0].id]: true };
    }
    return {};
  });

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => ({
      ...prev,
      [id]: !prev[id],
    }));
  };

  return (
    <main className="w-full max-w-[1240px] mx-auto mb-8 min-h-[560px]">
      <section
        aria-labelledby="case-history-title"
        className="rounded-3xl border border-deep-teal/20 bg-card-mint p-5 shadow-sm lg:p-8"
      >
        <div className="flex items-start justify-between gap-4 border-b border-charcoal-navy/10 pb-4">
          <div className="flex items-center gap-3">
            <span className="flex h-10 w-10 items-center justify-center rounded-full border border-deep-teal/15 bg-sea-foam text-deep-teal">
              <History className="h-5 w-5" aria-hidden="true" />
            </span>
            <div>
              <span className="block font-mono text-[10px] font-semibold uppercase tracking-wider text-deep-teal">
                Patient tracking
              </span>
              <h1 id="case-history-title" className="font-serif text-[26px] font-normal tracking-tight text-charcoal-navy lg:text-[30px]">
                Case History
              </h1>
            </div>
          </div>
          <button
            type="button"
            onClick={onBack}
            className="flex items-center gap-2 rounded-full border border-charcoal-navy/15 bg-paper-white px-4 py-2 font-sans text-[12px] font-semibold text-charcoal-navy/70 transition-colors hover:border-deep-teal/30 hover:text-deep-teal cursor-pointer"
            aria-label="Back to home"
          >
            <ArrowLeft className="h-4 w-4" aria-hidden="true" />
            <span className="hidden sm:inline">Back to Home</span>
          </button>
        </div>

        {entries.length === 0 ? (
          <div className="flex flex-col items-center justify-center px-4 py-10 text-center">
            <History className="mb-3 h-7 w-7 text-deep-teal/45" aria-hidden="true" />
            <p className="font-sans text-[14px] font-semibold text-charcoal-navy">No completed cases yet</p>
            <p className="mt-1 font-sans text-[12px] text-[#4a5e5d]">
              Run a demo or uploaded-patient analysis and it will be recorded here.
            </p>
          </div>
        ) : (
          <div className="mt-5 flex flex-col gap-3 w-full">
            {entries.map((entry) => {
              const isExpanded = Boolean(expandedIds[entry.id]);
              return (
                <article
                  key={entry.id}
                  className={`w-full rounded-2xl border transition-all duration-200 bg-paper-white overflow-hidden ${
                    isExpanded
                      ? 'border-deep-teal/30 shadow-xs'
                      : 'border-charcoal-navy/10 hover:border-deep-teal/25'
                  }`}
                >
                  {/* Collapsible Header */}
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => toggleExpand(entry.id)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        toggleExpand(entry.id);
                      }
                    }}
                    className="w-full flex items-center justify-between gap-3 p-4 sm:px-5 sm:py-4 cursor-pointer select-none transition-colors hover:bg-sea-foam/30"
                    aria-expanded={isExpanded}
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-sea-foam text-deep-teal">
                        {entry.source === 'demo' ? <FlaskConical className="h-4 w-4" /> : <Upload className="h-4 w-4" />}
                      </span>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <h3 className="truncate font-sans text-[14px] sm:text-[15px] font-semibold text-charcoal-navy">
                            {entry.patient_id}
                          </h3>
                          <span className="shrink-0 rounded-full border border-deep-teal/15 bg-sea-foam px-2 py-0.5 font-mono text-[9px] font-semibold text-deep-teal uppercase">
                            {entry.source}
                          </span>
                        </div>
                        <p className="font-mono text-[10px] text-charcoal-navy/55">
                          {new Date(entry.analyzed_at).toLocaleString()}
                        </p>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 shrink-0">
                      <span className="shrink-0 rounded-full border border-deep-teal/15 bg-sea-foam px-2.5 py-1 font-mono text-[10px] font-semibold text-deep-teal">
                        {entry.risk_tier}
                      </span>
                      <span
                        className={`flex h-7 w-7 items-center justify-center rounded-full border border-charcoal-navy/10 text-charcoal-navy/60 transition-transform duration-200 ${
                          isExpanded ? 'rotate-180 bg-deep-teal/10 text-deep-teal border-deep-teal/20' : 'bg-canvas'
                        }`}
                        aria-hidden="true"
                      >
                        <ChevronDown className="h-4 w-4" />
                      </span>
                    </div>
                  </div>

                  {/* Expanded Body Panel */}
                  {isExpanded && (
                    <div className="border-t border-charcoal-navy/10 bg-canvas/40 px-4 py-4 sm:px-5 sm:py-4.5 animate-in fade-in duration-200">
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-4">
                        {/* 1. Risk Score: Warm Amber Indicator */}
                        <div className="rounded-xl border border-amber-500/20 bg-amber-50/75 p-3">
                          <span className="block font-mono text-[9px] uppercase tracking-wider text-amber-900/70">
                            Risk score (Cox-PH)
                          </span>
                          <strong className="font-serif text-[20px] font-normal text-amber-950 block mt-0.5">
                            {entry.risk_score.toFixed(4)}
                          </strong>
                        </div>

                        {/* 2. Median Survival: Clinical Sea-Foam Indicator */}
                        <div className="rounded-xl border border-deep-teal/20 bg-sea-foam/85 p-3">
                          <span className="block font-mono text-[9px] uppercase tracking-wider text-deep-teal/70">
                            Median survival
                          </span>
                          <strong className="font-serif text-[20px] font-normal text-deep-teal block mt-0.5">
                            {entry.median_survival_months} mo
                          </strong>
                        </div>

                        {/* 3. Risk Tier: Dynamic Tier Indicator (Rose for High, Emerald for Low, Orange for Moderate) */}
                        <div className={`rounded-xl border p-3 ${
                          entry.risk_tier.toLowerCase().includes('high')
                            ? 'border-rose-300/60 bg-rose-50/85 text-rose-950'
                            : entry.risk_tier.toLowerCase().includes('low')
                            ? 'border-emerald-300/60 bg-emerald-50/85 text-emerald-950'
                            : 'border-orange-300/60 bg-orange-50/85 text-orange-950'
                        }`}>
                          <span className="block font-mono text-[9px] uppercase tracking-wider opacity-70">
                            Risk Tier
                          </span>
                          <span className="font-sans text-[13px] font-semibold block mt-1">
                            {entry.risk_tier}
                          </span>
                        </div>

                        {/* 4. Cohort Source: Provenance Sky-Blue Indicator */}
                        <div className="rounded-xl border border-sky-400/25 bg-sky-50/75 p-3">
                          <span className="block font-mono text-[9px] uppercase tracking-wider text-sky-900/65">
                            Cohort Source
                          </span>
                          <span className="font-sans text-[13px] font-semibold text-sky-950 block mt-1">
                            {entry.source === 'demo' ? 'TCGA Bundled Demo' : 'Uploaded Patient Data'}
                          </span>
                        </div>
                      </div>

                      <div className="mt-4 pt-3 border-t border-charcoal-navy/10 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <span className="font-sans text-[11px] text-charcoal-navy/60">
                          Cross-attended Whole-Slide Morphology &amp; RNA-Seq pathway embeddings.
                        </span>
                        <button
                          type="button"
                          onClick={() => onRerun(entry)}
                          className="self-start sm:self-auto inline-flex items-center gap-1.5 rounded-full border border-deep-teal/20 bg-deep-teal px-3.5 py-1.5 font-sans text-[11px] font-semibold text-white transition-colors hover:bg-forest-floor cursor-pointer"
                        >
                          <RotateCcw className="h-3 w-3" aria-hidden="true" />
                          Run again
                        </button>
                      </div>
                    </div>
                  )}
                </article>
              );
            })}
          </div>
        )}
      </section>
    </main>
  );
};
