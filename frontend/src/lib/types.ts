export type Track = {
  name: string;
  validAuc: number;
  ootAuc: number;
  ootGini: number;
  ootKs: number;
  isChampion: boolean;
};

export type Summary = {
  champion: string;
  generatedAt: string;
  modelVersion: string;
  featureSpecFingerprint: string;
  nFeaturesBuilt: number;
  nFeaturesSelected: number;
  nMonotonicConstraints: number;
  headline: {
    auc: number; gini: number; ks: number; brier: number;
    prAuc: number; badRate: number; n: number; psi: number;
  };
  baseline: { auc: number; gini: number; ks: number; brier: number };
  calibration: Record<string, number>;
  tracks: Track[];
  latency: { onnxP99Ms: number; nativeP99Ms: number; apiP99Ms: number; apiUsers: number };
  policy: { approveAt: number; referAt: number; lgd: number };
};

export type Band = {
  decision: string; n: number; share: number;
  badRate: number; meanPd: number; expectedLoss: number;
};

export type Portfolio = {
  bands: Band[];
  scoreDistribution: { score: number; count: number }[];
  deciles: { decile: number; n: number; badRate: number; meanPd: number; lift: number; cumBadCapture: number }[];
  vintages: { vintage: string; n: number; badRate: number; meanPd: number; meanScore: number }[];
};

export type SimulatorData = {
  score: number[]; pd: number[]; y: number[]; exposure: number[];
  sampledFrom: number; lgd: number;
};

export type Monitoring = {
  scorePsi: number; verdict: string; isAlarm: boolean;
  thresholds: { investigate: number; alarm: number };
  psiBins: { bin: number; expected: number; actual: number; contribution: number }[];
  featureCsi: { feature: string; csi: number; verdict: string }[];
  vintagePsi: { vintage: string; n: number; psi: number; verdict: string }[];
  challengers: { name: string; ootAuc: number; ootGini: number; isChampion: boolean }[];
};

export type GroupRow = {
  group: string; n: number; selectionRate: number; observedBadRate: number;
  meanPredictedPd: number; calibrationGap: number; tprGoodApproved: number;
};

export type FairnessGroup = {
  attribute: string;
  disparate_impact: number;
  passes_four_fifths: boolean;
  equal_opportunity_difference: number;
  equalized_odds_difference: number;
  selection_rate_difference: number;
  reference_group: string;
  worst_group: string;
  byGroup: GroupRow[];
};

export type Fairness = {
  fourFifths: number;
  groups: Record<string, FairnessGroup>;
  cutoffCurve: {
    approvalRate: number; disparateImpact: number;
    equalOpportunityDifference: number; badRateAmongApproved: number;
    passesFourFifths: boolean;
  }[];
  thresholdOptimizer: {
    strategy: string; approvalRate: number; disparateImpact: number;
    equalOpportunityDifference: number; badRateAmongApproved: number; note: string;
  }[];
};

export type ReasonCode = {
  rank: number; family: string; label: string; phrase: string;
  actionable: boolean; contribution: number; driving_features: string[];
};

export type Applicant = {
  skIdCurr: number; pd: number; score: number; decision: string;
  exposure: number; expectedLoss: number;
  profile: {
    income: number; credit: number; annuity: number; ageYears: number;
    employedYears: number; education: string; occupation: string; contractType: string;
  };
  reasonCodes: ReasonCode[];
  shap: { feature: string; value: number | null; shap: number }[];
  baseValue: number;
};

export type Explainability = {
  globalShap: { feature: string; mean_abs_shap: number; mean_shap: number; share: number }[];
  additivityError: number;
  reasonCodes: {
    mapping_version: number;
    unmapped_features: string[];
    mean_distinct_families: number;
    share_with_four_reasons: number;
  };
  counterfactuals: {
    n_evaluated: number; flip_single_action: number; flip_stacked_three: number;
    median_score_gain: number; by_segment: Record<string, number>;
  };
  informationValues: { feature: string; iv: number; strength: string }[];
};

export type Genai = {
  credentialsConfigured: boolean;
  unitEconomics: {
    memo_model: string;
    memos_per_1000_decisions: number;
    cost_per_memo_usd: number;
    cost_per_1000_decisions_usd: number;
  };
  portfolioQueries: { name: string; description: string }[];
  memos: {
    skIdCurr: number; decision: string; model: string; offline: boolean;
    promptVersion: string; promptHash: string; costUsd: number;
    memo: { summary: string; detail: string; reason_families_cited: string[]; next_steps: string };
  }[];
  copilot: {
    question: string; answer: string; model: string;
    offline: boolean; toolsCalled: string[]; costUsd: number;
  }[];
};
