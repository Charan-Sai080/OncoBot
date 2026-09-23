import React from 'react';
import { ArrowLeft, CheckCircle2 } from 'lucide-react';
import { UnifiedDashboard } from './UnifiedDashboard';
import type { DashboardTab } from './UnifiedDashboard';
import type {
  ClinicalReport,
  HeatmapData,
  KmCurveData,
  PathwayItem,
  PatientProfile,
} from '../types';

interface AnalysisResultsPageProps {
  activeTab: DashboardTab;
  onTabChange: (tab: DashboardTab) => void;
  patientProfile: PatientProfile | null;
  patientId: string;
  heatmap: HeatmapData | null;
  kmData: KmCurveData | null;
  pathways: PathwayItem[];
  report: ClinicalReport | null;
  isAnalyzing: boolean;
  isReportLoading: boolean;
  onRegenerateReport: () => void;
  onBack: () => void;
}

export const AnalysisResultsPage: React.FC<AnalysisResultsPageProps> = ({
  activeTab,
  onTabChange,
  patientProfile,
  patientId,
  heatmap,
  kmData,
  pathways,
  report,
  isAnalyzing,
  isReportLoading,
  onRegenerateReport,
  onBack,
}) => (
  <main className="mx-auto mb-8 w-full max-w-[1240px]" aria-labelledby="analysis-results-title">
    <header className="mb-6 flex flex-col gap-4 rounded-3xl border border-deep-teal/20 bg-card-mint p-5 shadow-sm sm:flex-row sm:items-center sm:justify-between lg:p-6">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-deep-teal/15 bg-sea-foam text-deep-teal">
          <CheckCircle2 className="h-5 w-5" aria-hidden="true" />
        </span>
        <div>
          <span className="block font-mono text-[10px] font-semibold uppercase tracking-wider text-deep-teal">
            Analysis complete · {patientId}
          </span>
          <h1 id="analysis-results-title" className="font-serif text-[26px] font-normal tracking-tight text-charcoal-navy lg:text-[30px]">
            Patient Analysis Results
          </h1>
          <p className="mt-1 font-sans text-[12px] text-[#4a5e5d]">
            Review the multimodal pathology, genomics, survival, and clinical synthesis outputs.
          </p>
        </div>
      </div>
      <button
        type="button"
        onClick={onBack}
        className="inline-flex shrink-0 items-center justify-center gap-2 self-start rounded-full border border-deep-teal/20 bg-paper-white px-4 py-2 font-sans text-[12px] font-semibold text-deep-teal transition-colors hover:bg-sea-foam cursor-pointer sm:self-auto"
      >
        <ArrowLeft className="h-4 w-4" aria-hidden="true" />
        New Analysis
      </button>
    </header>

    {patientProfile && (
      <section className="mb-6 flex w-full flex-wrap items-center justify-between gap-3 rounded-xl border border-charcoal-navy/15 bg-paper-white px-4 py-2.5 text-[13px]" aria-label="Patient clinical profile">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] font-semibold uppercase text-deep-teal">Stage &amp; Grade:</span>
          <span className="font-medium text-charcoal-navy">{patientProfile.stage} • {patientProfile.grade}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] font-semibold uppercase text-deep-teal">Demographics:</span>
          <span className="font-medium text-charcoal-navy">{patientProfile.age} yrs • {patientProfile.sex}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] font-semibold uppercase text-deep-teal">TMB:</span>
          <span className="font-mono font-semibold text-deep-teal">{patientProfile.tmb}</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] font-semibold uppercase text-deep-teal">WSI Tiles:</span>
          <span className="font-mono font-semibold text-charcoal-navy">{patientProfile.num_wsi_tiles} patches</span>
        </div>
      </section>
    )}

    <div id="unified-workspace">
      <UnifiedDashboard
        activeTab={activeTab}
        onTabChange={onTabChange}
        heatmap={heatmap}
        kmData={kmData}
        pathways={pathways}
        report={report}
        patientId={patientId}
        isAnalyzing={isAnalyzing}
        isReportLoading={isReportLoading}
        onRegenerateReport={onRegenerateReport}
      />
    </div>
  </main>
);
