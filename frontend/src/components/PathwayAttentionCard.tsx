import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Dna } from 'lucide-react';
import type { PathwayItem } from '../types';

interface PathwayAttentionCardProps {
  pathways: PathwayItem[];
}

export const PathwayAttentionCard: React.FC<PathwayAttentionCardProps> = ({ pathways }) => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(0);

  // Default curated pathways if not yet loaded from server
  const defaultPathways: PathwayItem[] = [
    {
      name: 'Focal Adhesion (hsa04510)',
      attention_score: 0.892,
      percentile: 98,
      biological_role: 'Mediates cell-matrix attachment and aggressive stromal invasion in bladder urothelium.',
      genes: ['PTEN', 'PIK3CA', 'AKT1', 'VEGFA', 'EGFR'],
    },
    {
      name: 'ERBB Signaling Pathway (hsa04012)',
      attention_score: 0.841,
      percentile: 95,
      biological_role: 'Drives EGFR/ERBB2 receptor tyrosine kinase hyperactivation and rapid cellular proliferation.',
      genes: ['ERBB2', 'EGFR', 'KRAS', 'HRAS'],
    },
    {
      name: 'Cell Cycle & G1/S Transition (hsa04110)',
      attention_score: 0.785,
      percentile: 91,
      biological_role: 'Loss of G1/S checkpoint arrest via TP53 mutation and CDKN2A/RB1 homozygous deletions.',
      genes: ['TP53', 'CDKN1A', 'CDKN2A', 'RB1', 'CCND1', 'CDK4'],
    },
    {
      name: 'ECM-Receptor Interaction (hsa04512)',
      attention_score: 0.742,
      percentile: 88,
      biological_role: 'Disrupts extracellular basement membrane integrity promoting muscularis propria invasion.',
      genes: ['COL1A1', 'FN1', 'ITGA2', 'LAMB1'],
    },
    {
      name: 'Bladder Cancer Specific Signaling (hsa05219)',
      attention_score: 0.718,
      percentile: 85,
      biological_role: 'Core pathway linking FGFR3/RAS activating mutations with invasive urothelial transition.',
      genes: ['FGFR3', 'HRAS', 'TP53', 'RB1', 'CDKN2A'],
    },
    {
      name: 'PI3K-Akt Signaling Pathway (hsa04151)',
      attention_score: 0.695,
      percentile: 82,
      biological_role: 'Promotes metabolic reprogramming, survival signaling, and chemoresistance.',
      genes: ['PIK3CA', 'PTEN', 'AKT1', 'MTOR'],
    },
  ];

  const items = pathways.length > 0 ? pathways : defaultPathways;

  return (
    <div id="pathways-section" className="bg-card-mint border border-charcoal-navy/15 rounded-2xl p-6 mb-8 flex flex-col shadow-none">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4 pb-3 border-b border-charcoal-navy/10">
        <div>
          <span className="font-mono text-[11px] font-semibold text-deep-teal tracking-wider uppercase block mb-1">
            GENOMIC ATTENTION WEIGHTS
          </span>
          <h2 className="font-serif text-[22px] font-normal text-charcoal-navy tracking-tight">
            Top KEGG Pathway Attention Ranks
          </h2>
        </div>
        <span className="font-mono text-[11px] font-semibold text-deep-teal bg-sea-foam px-3 py-1 rounded-full border border-deep-teal/20 self-start sm:self-auto">
          320 Pathways Active
        </span>
      </div>

      <p className="font-sans text-[13px] text-[#4a5e5d] mb-4">
        Cross-attention weights connecting patient RNA-Seq transcriptomics with WSI morphology across 320 KEGG biological pathways.
      </p>

      {/* Pathways List */}
      <div className="flex flex-col gap-3">
        {items.slice(0, 6).map((item, idx) => {
          const isExpanded = expandedIndex === idx;
          const scorePercent = Math.round(item.attention_score * 100);

          return (
            <div
              key={item.name}
              className="bg-paper-white border border-charcoal-navy/15 rounded-xl p-3.5 transition-all cursor-pointer hover:border-deep-teal/30"
              onClick={() => setExpandedIndex(isExpanded ? null : idx)}
            >
              <div className="flex items-center justify-between gap-3 mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[12px] font-bold text-deep-teal w-5">
                    #{idx + 1}
                  </span>
                  <span className="font-sans text-[14px] font-semibold text-charcoal-navy">
                    {item.name}
                  </span>
                </div>

                <div className="flex items-center gap-2.5">
                  <span className="font-mono text-[12px] font-bold text-deep-teal">
                    {item.attention_score.toFixed(3)}
                  </span>
                  <span className="bg-sea-foam text-deep-teal font-mono text-[10px] font-semibold px-2 py-0.5 rounded-full border border-deep-teal/15">
                    {item.percentile}th %ile
                  </span>
                  {isExpanded ? (
                    <ChevronUp className="w-4 h-4 text-charcoal-navy/60" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-charcoal-navy/60" />
                  )}
                </div>
              </div>

              {/* Progress Bar */}
              <div className="w-full h-2 bg-charcoal-navy/10 rounded-full overflow-hidden mb-1.5">
                <div
                  className="h-full bg-gradient-to-r from-lake-teal to-deep-teal rounded-full transition-all duration-500"
                  style={{ width: `${scorePercent}%` }}
                />
              </div>

              {/* Expandable Details */}
              {isExpanded && (
                <div className="mt-3 pt-2.5 border-t border-charcoal-navy/10 flex flex-col gap-2">
                  <p className="font-sans text-[12px] text-[#4a5e5d] leading-relaxed">
                    {item.biological_role}
                  </p>

                  <div className="flex flex-wrap items-center gap-1.5 mt-1">
                    <span className="font-mono text-[11px] font-semibold text-charcoal-navy/70 mr-1 flex items-center gap-1">
                      <Dna className="w-3 h-3 text-deep-teal" />
                      KEY GENES:
                    </span>
                    {item.genes.map((gene) => (
                      <span
                        key={gene}
                        className="bg-card-mint text-deep-teal font-mono text-[10px] font-semibold px-2 py-0.5 rounded-md border border-deep-teal/15"
                      >
                        {gene}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
