import React from 'react';
import { ArrowRight, Loader2 } from 'lucide-react';

interface HeroSectionProps {
  onRunAnalysis: () => void;
  isAnalyzing: boolean;
  children?: React.ReactNode;
}

export const HeroSection: React.FC<HeroSectionProps> = ({
  onRunAnalysis,
  isAnalyzing,
  children,
}) => {
  return (
    <section id="overview" className="w-full max-w-[1240px] mx-auto mb-8 flex flex-col gap-8 relative box-border">
      <div className="w-full flex items-center justify-between gap-4 relative box-border">
      {/* Left Hand-Drawn Illustration Flank */}
      <div className="hidden lg:flex flex-0 shrink-1 basis-[310px] max-w-[330px] min-w-[200px] items-center justify-start select-none">
        <img
          src="/assets/left_hero_illustration.png"
          alt="Pathology and Molecular Analysis Illustration"
          className="max-w-full w-auto h-auto max-h-[320px] block object-contain select-none pointer-events-none"
          width="295"
          height="320"
          loading="eager"
        />
      </div>

      {/* Center Editorial Typography & Controls */}
      <div className="flex-1 basis-[420px] max-w-[500px] min-w-[280px] mx-auto text-center px-2">
        <h1 className="font-serif text-[38px] md:text-[44px] leading-[1.15] font-normal text-charcoal-navy tracking-[-0.5px] mb-4 relative">
          We connect<br />
          <span className="font-serif italic font-medium text-deep-teal">genomic pathways</span><br />
          and tissue morphology
          <span className="block w-full max-w-[280px] mx-auto -mt-1">
            <svg viewBox="0 0 280 18" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-auto">
              <path d="M6 12C75 4 195 4 274 11" stroke="#1c5d5f" strokeWidth="2.8" strokeLinecap="round" />
            </svg>
          </span>
        </h1>

        <p className="font-sans text-[15px] leading-relaxed text-[#4a5e5d] max-w-[460px] mx-auto mb-6">
          Predict patient-specific cancer survival risk and generate interpretable clinical reports by cross-attending RNA-Seq transcriptomics with gigapixel Whole-Slide Images.
        </p>

        {/* Clean, Unified Call To Action Cluster */}
        <div className="flex items-center justify-center">
          <button
            id="hero-analyze-btn"
            onClick={onRunAnalysis}
            disabled={isAnalyzing}
            className="bg-deep-teal hover:bg-forest-floor disabled:opacity-75 text-white font-sans text-[15px] font-semibold px-8 py-3.5 rounded-full transition-colors inline-flex items-center gap-2 cursor-pointer border-none shadow-none active:scale-[0.99]"
          >
            {isAnalyzing ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Cross-Attending Pathways & WSI...
              </>
            ) : (
              <>
                Run Multimodal Analysis
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>

        </div>
      </div>

      {/* Right Hand-Drawn Illustration Flank */}
      <div className="hidden lg:flex flex-0 shrink-1 basis-[330px] max-w-[350px] min-w-[220px] items-center justify-end select-none">
        <img
          src="/assets/right_hero_illustration.png"
          alt="Clinical Synthesis and Prognosis Illustration"
          className="max-w-full w-auto h-auto max-h-[320px] block object-contain select-none pointer-events-none"
          width="329"
          height="320"
          loading="eager"
        />
      </div>
      </div>

      {children}
    </section>
  );
};
