export interface SystemStatus {
  device: string;
  kegg_pathways_count: number;
  model_loaded: boolean;
  cache_ready: boolean;
}

export interface PatientSummary {
  patient_id: string;
  label: string;
}

export interface CaseHistoryEntry {
  id: string;
  patient_id: string;
  source: 'demo' | 'uploaded';
  analyzed_at: string;
  risk_score: number;
  risk_tier: string;
  median_survival_months: number;
}

export interface PatientProfile {
  patient_id: string;
  grade: string;
  stage: string;
  age: number;
  sex: string;
  mutation_count: number;
  tmb: string;
  num_wsi_tiles: number;
  description: string;
  key_mutations: string[];
}

export interface InferenceResult {
  success: boolean;
  patient_id: string;
  risk_score: number;
  risk_tier: string;
  risk_tier_class: string;
  median_survival_months: number;
  cohort_percentile: number;
  c_index: number;
  cross_modal_attention_summary: {
    top_attended_patches_count: number;
    peak_attention_weight: number;
    pathway_count: number;
    modality_interaction: string;
  };
}

export interface TileData {
  x: number;
  y: number;
  attention_weight: number;
  histology_type: string;
  cellular_density: string;
  rank: number;
}

export interface HeatmapData {
  patient_id: string;
  grid_dim: number;
  tile_size: number;
  tiles_count: number;
  min_attn: number;
  max_attn: number;
  mean_attn: number;
  raw_wsi_base64: string;
  heatmap_overlay_base64: string;
  hotspots_base64: string;
  tiles: TileData[];
}

export interface KmCurveData {
  time_points_months: number[];
  high_risk_survival: number[];
  low_risk_survival: number[];
  patient_trajectory: number[];
}

export interface PathwayItem {
  name: string;
  attention_score: number;
  percentile: number;
  biological_role: string;
  genes: string[];
}

export interface ClinicalReport {
  success: boolean;
  patient_id: string;
  risk_score: number;
  report_markdown: string;
  model_used: string;
  top_pathways: PathwayItem[];
  attention_summary?: any;
}

export interface UserProfile {
  name: string;
  title: string;
  email: string;
  institution: string;
  department: string;
  licenseNumber: string;
  role: string;
  avatarUrl?: string;
  casesReviewedCount: number;
  lastLogin: string;
}
