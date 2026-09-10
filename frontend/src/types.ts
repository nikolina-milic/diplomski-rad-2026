export interface FiredRule { name: string; weight: number; explanation: string }
export interface Shap { feature: string; value: number }
export type Level = 'LOW' | 'MEDIUM' | 'HIGH'

export interface EventInfo {
  first_name?: string
  last_name?: string
  amount: number
  currency: string
  merchant_category: string
  country: string
  city: string | null
  latitude: number
  longitude: number
  device_id: string
  channel: string
  timestamp: string
  home_country: string | null
  distance_from_home_km: number
}

export interface Decision {
  id?: number
  event_id: string
  user_id: string
  first_name?: string | null
  last_name?: string | null
  final_score: number
  level: Level
  action: string
  executed_action?: string
  mode?: string
  rules_score?: number
  ml_score?: number
  fired_rules?: FiredRule[]
  shap_top?: Shap[]
  features?: Record<string, number>
  model_version: string
  guardrail: string | null
  explanation: string
  event?: EventInfo | null
}

export interface ModelVersion { version: string; status: string; metrics: Record<string, number> }
export interface Stats { total: number; actions: Record<string, number>; levels: Record<string, number> }
export interface DecisionCount {
  total: number
  by_level: Partial<Record<Level, number>>
  by_action: Record<string, number>
}
export interface Confusion { tp: number; fp: number; tn: number; fn: number }
export interface Metrics {
  confusion: Confusion
  live_metrics: Record<string, number>
  versions: ModelVersion[]
  n: number
}
export interface ReliabilityBin {
  bin_start: number
  bin_end: number
  mean_predicted: number
  fraction_positive: number
  count: number
}
export interface CalibrationReport {
  brier: number
  ece: number
  reliability: ReliabilityBin[]
}
export interface ApproachResult {
  metrics: Record<string, number>
  pr_curve: { precision: number; recall: number }[]
  calibration?: CalibrationReport
}
export interface CostSection {
  low_max: number
  medium_max: number
  total_cost: number
  cost_per_event: number
  challenge_rate?: number
  review_rate?: number
  feasible?: boolean
}
export interface Evaluation {
  n_test: number
  n_fraud: number
  approaches: Record<string, ApproachResult>
  latency: { n: number; mean_ms: number; p50_ms: number; p95_ms: number; max_ms: number }
  cost?: {
    cost_matrix: Record<string, number>
    capacity: Record<string, number>
    current: CostSection
    closed_form: CostSection
    empirical: CostSection
    constrained: CostSection
    allow_all_baseline: { total_cost: number; note: string }
  }
}

export interface CostMatrix {
  avg_fraud_loss: number
  challenge_cost: number
  review_cost: number
  block_cost: number
  challenge_stop_rate: number
  review_stop_rate: number
}
export interface Capacity { max_challenge_rate: number; max_review_rate: number }
export interface CostConfig {
  cost_matrix: CostMatrix
  capacity: Capacity
  closed_form: { low_max: number; medium_max: number }
  current: { low_max: number; medium_max: number }
  applied?: { low_max: number; medium_max: number; challenge_rate: number; review_rate: number; feasible: boolean } | null
}

export interface Calibration {
  fitted: boolean
  method: string
  strategy: string
  meta_weights: { rules: number; ml: number; interaction: number; intercept: number } | null
}

export interface FeatureDrift {
  feature: string
  psi: number
  severity: string
  ks_statistic: number
  ks_pvalue: number
  reference_mean: number
  current_mean: number
}
export interface Drift {
  available: boolean
  reason?: string
  n_reference?: number
  n_current?: number
  features?: FeatureDrift[]
  n_drifting?: number
  max_psi?: number
  worst_feature?: string | null
  overall?: string
  thresholds?: { stable: number; significant: number }
  score?: { psi: number; severity: string; reference_mean: number; current_mean: number }
}

export interface Rule {
  name: string
  condition: string
  weight: number
  explanation: string
  enabled: boolean
  thresholds: number[]
  condition_parts: string[]
  threshold_features: string[]
}
export interface Fusion { strategy: string; rules_weight: number; ml_weight: number; cascade_threshold: number }
export interface Policy {
  low_max: number
  medium_max: number
  mode: string
  level_actions: Record<string, string>
}
