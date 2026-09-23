import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="w-full max-w-[1240px] mx-auto mt-12 pt-8 pb-12 border-t border-charcoal-navy/15 text-center flex flex-col items-center gap-3">
      <div className="flex flex-wrap items-center justify-center gap-6 text-[13px] text-charcoal-navy/70">
        <span><strong>Model:</strong> PAMT (Pathway-Aware Multimodal Transformer)</span>
        <span>•</span>
        <span><strong>Cohort:</strong> TCGA-BLCA Muscle-Invasive Bladder Cancer (N=412)</span>
        <span>•</span>
        <span><strong>Pathways:</strong> 320 Curated KEGG Biological Networks</span>
        <span>•</span>
        <span><strong>Accuracy:</strong> C-Index 0.784 Validation</span>
      </div>

      <p className="font-sans text-[12px] text-charcoal-navy/50 max-w-[680px]">
        Onco_Bot is an explainable clinical decision support platform designed to bridge gigapixel digital pathology with genomic transcriptomics. Investigational research use only.
      </p>
    </footer>
  );
};
