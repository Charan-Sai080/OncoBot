import React from 'react';
import type { InferenceResult } from '../types';

interface StatBannerStripProps {
  inference: InferenceResult | null;
}

const pipelineFacts = [
  {
    iconSrc: '/assets/pipeline-icons/database-3d.png',
    value: '320',
    label: 'KEGG Cancer Pathways',
    iconBackground: '#def1ef',
  },
  {
    iconSrc: '/assets/pipeline-icons/document-3d.png',
    value: '1,024',
    label: 'WSI Patches / Sweet-Spot',
    iconBackground: '#e7f0fb',
  },
  {
    iconSrc: '/assets/pipeline-icons/dna-3d.png',
    value: 'DINO ViT-S/16',
    label: '384-Dim Embeddings',
    iconBackground: '#e0f2ef',
  },
  {
    iconSrc: '/assets/pipeline-icons/chart-3d.png',
    value: 'Cox-PH',
    label: 'Hazard Regression',
    iconBackground: '#f3f1ef',
  },
  {
    iconSrc: '/assets/pipeline-icons/people-3d.png',
    value: 'TCGA-BLCA',
    label: 'Bladder Cancer Cohort',
    iconBackground: '#ddf2ec',
  },
  {
    iconSrc: '/assets/pipeline-icons/microscope-3d.png',
    value: 'Gigapixel',
    label: 'Whole-Slide Images',
    iconBackground: '#f3f3f1',
  },
];

export const StatBannerStrip: React.FC<StatBannerStripProps> = ({ inference }) => {
  const isAnalysisReady = Boolean(inference?.success);

  return (
    <section
      aria-label="OncoBot analysis pipeline capabilities"
      className={`framework-marquee w-full overflow-hidden rounded-full border border-deep-teal/25 bg-white/90 px-3 py-3 transition-shadow duration-500 ${
        isAnalysisReady
          ? 'shadow-[0_10px_30px_rgba(28,93,95,0.12)]'
          : 'shadow-[0_8px_24px_rgba(28,93,95,0.07)]'
      }`}
    >
      <div className="framework-marquee-track">
        {[0, 1].map((groupIndex) => (
          <div
            key={groupIndex}
            className="flex shrink-0 items-center"
            aria-hidden={groupIndex === 1}
          >
            {pipelineFacts.map((fact, index) => (
              <div
                key={`${groupIndex}-${fact.value}`}
                className={`group flex min-w-[220px] items-center gap-3 px-5 py-1.5 ${
                  index > 0 || groupIndex > 0 ? 'border-l border-deep-teal/15' : ''
                }`}
              >
                <span
                  className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-full border border-deep-teal/10 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)] transition-transform duration-300 group-hover:scale-105"
                  style={{ backgroundColor: fact.iconBackground }}
                >
                  <img
                    src={fact.iconSrc}
                    alt=""
                    className="h-11 w-11 object-contain"
                    width="44"
                    height="44"
                    loading="lazy"
                    aria-hidden="true"
                  />
                </span>

                <span className="flex min-w-0 flex-col whitespace-nowrap">
                  <span className="font-sans text-[15px] font-bold leading-tight tracking-[-0.01em] text-deep-teal">
                    {fact.value}
                  </span>
                  <span className="mt-1 font-mono text-[9px] font-semibold uppercase leading-tight tracking-[0.06em] text-charcoal-navy/60">
                    {fact.label}
                  </span>
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
};
