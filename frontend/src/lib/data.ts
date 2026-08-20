/**
 * Static demo data, read at build time by Server Components.
 *
 * Every page is driven by these snapshots so the deployed link works with no
 * backend running. `src/api/export_demo.py` writes them from the same
 * artifacts the model pipeline produces, so nothing here is hand-authored.
 */
import fs from "node:fs";
import path from "node:path";

import type {
  Applicant, Explainability, Fairness, Monitoring, Portfolio, SimulatorData, Summary,
} from "./types";

const DATA_DIR = path.join(process.cwd(), "public", "data");

function read<T>(name: string): T {
  const file = path.join(DATA_DIR, `${name}.json`);
  if (!fs.existsSync(file)) {
    throw new Error(
      `Missing ${name}.json. Run \`make demo-data\` to regenerate the frontend snapshots.`,
    );
  }
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

export const getSummary = () => read<Summary>("summary");
export const getPortfolio = () => read<Portfolio>("portfolio");
export const getSimulator = () => read<SimulatorData>("simulator");
export const getMonitoring = () => read<Monitoring>("monitoring");
export const getFairness = () => read<Fairness>("fairness");
export const getApplicants = () => read<Applicant[]>("applicants");
export const getExplainability = () => read<Explainability>("explainability");
