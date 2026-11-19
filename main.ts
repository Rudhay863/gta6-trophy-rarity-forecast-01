// gta6-trophy-rarity-forecast-01 — forecast each GTA VI PS5 trophy's Exophase earn
// rate at the case target, as five quantiles (5/25/50/75/95%), from the case file
// alone. Node 22, standard library only, no network, no model calls, deterministic.
//
// Model (designed for this contest, from the kit spec only):
//
// Exophase refreshes a game's counts in steps, not continuously: a read shows the
// rates as of the last refresh, and each refresh dilutes every rate because the
// denominator (players tracked) jumps faster than the numerator (earners) early on.
// So a rate is a ratio, and we forecast the ratio two ways and average them:
//
//   A) DIRECT — ordinary least squares of logit(rate) on x = ln(1 + hours since
//      the game's release), over the case's distinct reads (consecutive identical
//      reads collapse to one: a frozen page carries no new information).
//   B) DECOMPOSED — OLS of ln(earners) on x and of ln(players tracked) on x, then
//      rate = earners / tracked at the target. This is the view that sees dilution
//      for what it is: earners can only grow, while tracked can jump.
//
// Both fits are ANCHORED at the last read (the forecast is last-rate plus a
// fitted drift over the remaining log-time gap), so a level misfit cannot drag the
// median away from the freshest observation. Each fitted slope is shrunk toward a
// prior slope chosen from the trophy's own description (story-chapter trophies
// climb fast, counted grinds climb slowly, 100%-completion / platinum climb
// slowest), with the data's weight growing with the number of distinct reads.
// Fitted slopes are clamped to [-0.6, 2.0] logit units per log-time unit, and the
// median is never moved more than 3.5 logit units from the last read. When the
// history is a single frozen page and the horizon is short, the median instead
// drifts DOWN: the next event is a refresh, and the kit documents that a refresh
// dilutes (tracked 638 -> 3,868 at the first one, every rate but the 100% one
// lower). A trophy already at 100% keeps a tight band near 100% for the same
// reason — it did not fall through that refresh either.
//
// Spread (the honest part): the pinball loss on the logit scale punishes a narrow
// band that misses far more than a wide band that covers. The logit spread s is
//   s = sqrt( (0.45*sqrt(days))^2 + s_fit^2 + s_refresh^2 + s_rare^2 )
// where s_fit is the residual spread of the direct fit, s_refresh prices the
// unseen-refresh risk (large when the history is one frozen page, small once many
// distinct refreshes have been seen), and s_rare widens the band for very rare
// trophies whose earner counts are tiny. s is clamped to [0.35, 1.8]. Quantiles
// are the median logit shifted by the normal z-scores times s, mapped back
// through the logistic function. No randomness anywhere: two runs are identical.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Levels scored, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at those levels. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { release_date: string };
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const sigmoidPct = (l: number): number => 100 / (1 + Math.exp(-l));
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));

/** Least-squares line through (x, y); returns [slope, intercept, residual SD]. */
function ols(xs: number[], ys: number[]): [number, number, number] {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = my - slope * mx;
  let rss = 0;
  for (let i = 0; i < n; i++) rss += (ys[i] - (intercept + slope * xs[i])) ** 2;
  const residSd = n > 2 ? Math.sqrt(rss / (n - 2)) : 0;
  return [slope, intercept, residSd];
}

/**
 * Prior drift (logit units per unit of ln(1+hours)) from the trophy's own text.
 * Story beats are finished by a large share of early players, so they climb
 * fast; counted grinds ("Complete 250 Stunts", "Spend $100,000") need hours of
 * play and climb slowly; 100% completion and the platinum climb slowest.
 */
function priorSlope(t: Trophy): number {
  const d = `${t.name} ${t.description}`.toLowerCase();
  if (t.tier === 'platinum') return 0.25;
  if (/100% completion/.test(d)) return 0.25;
  if (/complete \d+|earn \$|spend \$|purchase all|discover \d+|win \d+|medal in all|at every|wa?ink it at \d+|complete 3 collectible|earn a gold/.test(d)) return 0.35;
  if (/complete chapter|complete the final mission/.test(d)) return 0.85;
  if (/try each type|complete an arcade game|enjoy a cocktail|follow-through|partnership with mass appeal/.test(d)) return 0.7;
  if (t.tier === 'gold') return 0.5;
  if (t.tier === 'silver') return 0.45;
  return 0.6;
}

/** Loose ceiling on a week-one median rate by tier (only binds on wild extrapolation). */
function tierCap(t: Trophy): number {
  if (t.tier === 'platinum') return 3;
  if (t.tier === 'gold') return 40;
  return 95;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  const releaseMs = Date.parse(c.game.release_date);
  const targetMs = Date.parse(c.target);
  const horizonDays = c.horizon_minutes / 1440;
  // Log-time of the target, measured from the release (never below 0).
  const xTarget = Math.log(1 + Math.max(0, (targetMs - releaseMs) / 3_600_000));

  // Collapse consecutive identical reads: a frozen page is one observation.
  const distinct: Read[] = [];
  for (const r of c.history) {
    const prev = distinct[distinct.length - 1];
    if (
      prev &&
      prev.players_tracked === r.players_tracked &&
      prev.rates.length === r.rates.length &&
      prev.rates.every((v, i) => v === r.rates[i])
    ) {
      continue;
    }
    distinct.push(r);
  }
  const last = c.history[c.history.length - 1];
  if (last === undefined) throw new Error(`${c.id} has no history`);
  const lastAtMs = Date.parse(last.at);
  const xLast = Math.log(1 + Math.max(0, (lastAtMs - releaseMs) / 3_600_000));
  const dx = Math.max(0, xTarget - xLast);
  const nDistinct = distinct.length;

  return c.trophies.map((t, i) => {
    // Last non-null rate for this trophy (fall back to 0.01% if never seen).
    let rLast = 0.01;
    for (let j = c.history.length - 1; j >= 0; j--) {
      const v = c.history[j].rates[i];
      if (v !== null && v !== undefined) {
        rLast = v;
        break;
      }
    }
    const lLast = logit(rLast);
    const prior = priorSlope(t);

    // Per-read series for this trophy over distinct reads (skip nulls).
    const xs: number[] = [];
    const logits: number[] = [];
    const logEarners: number[] = [];
    const logTracked: number[] = [];
    for (const r of distinct) {
      const v = r.rates[i];
      if (v === null || v === undefined) continue;
      const h = Math.max(0, (Date.parse(r.at) - releaseMs) / 3_600_000);
      xs.push(Math.log(1 + h));
      logits.push(logit(v));
      logEarners.push(Math.log(Math.max(0.5, (clampPct(v) / 100) * r.players_tracked)));
      logTracked.push(Math.log(Math.max(1, r.players_tracked)));
    }
    const xVar = xs.length > 1 ? Math.max(...xs) - Math.min(...xs) : 0;
    const fittable = xs.length >= 3 && xVar > 0.05;

    // Data weight grows with distinct reads; the prior always keeps a vote.
    const wData = Math.min(Math.max(xs.length - 1, 0), 6);
    const wPrior = 2;
    const shrink = (slope: number): number =>
      clamp((wData * slope + wPrior * prior) / (wData + wPrior), -0.6, 2.0);

    let medianLogit: number;
    let sFit = 0;
    if (fittable) {
      const [slopeA, , residA] = ols(xs, logits);
      sFit = clamp(residA, 0, 1.0);
      const medA = lLast + shrink(slopeA) * dx;
      const [slopeE] = ols(xs, logEarners);
      const [slopeN] = ols(xs, logTracked);
      // Earner growth minus tracked growth is the rate's drift in log space,
      // which for small rates is the drift of its logit too.
      const medB = lLast + shrink(slopeE - slopeN) * dx;
      medianLogit = (medA + medB) / 2;
    } else {
      // No usable trend (e.g. launch night's single frozen page): the page has
      // shown no movement at all, so the data earns no vote. Over a SHORT
      // horizon the next event is almost certainly a refresh, and the kit
      // documents what a refresh does: tracked jumps (638 -> 3,868 at the
      // first one) while earners grow only a little, so every rate but a 100%
      // one falls. Price that in as a downward drift. Over a LONG horizon
      // adoption has time to catch up, so drift at half the prior pace instead.
      medianLogit =
        horizonDays < 0.25 ? lLast - 0.7 : lLast + prior * 0.5 * dx;
    }
    medianLogit = clamp(medianLogit, lLast - 3.5, lLast + 3.5);
    // The tier ceiling only binds on wild upward extrapolation: it never
    // pulls the median below the last read (a trophy already at 100% stays
    // near 100% — the kit records that the 100% trophy did not fall even
    // through the first refresh, when every other rate did).
    medianLogit = Math.min(medianLogit, Math.max(lLast, logit(tierCap(t))));

    // Spread: horizon random walk + fit residual + unseen-refresh risk + rarity.
    const sRefresh = 0.25 + 0.75 * Math.max(0, 1 - nDistinct / 8);
    const sRare = rLast < 0.2 ? 0.3 : rLast < 1 ? 0.15 : 0;
    // A trophy every tracked player already holds cannot fall far: its band
    // stays tight even though the refresh spread would otherwise widen it.
    const s =
      rLast >= 99.99
        ? 0.35
        : clamp(
            Math.sqrt((0.45 * Math.sqrt(horizonDays)) ** 2 + sFit ** 2 + sRefresh ** 2 + sRare ** 2),
            0.35,
            1.8,
          );

    const q = Z.map((z) => round4(clampPct(sigmoidPct(medianLogit + z * s))));
    // Keep the row non-decreasing after rounding (rounding can only tie).
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR)
  .filter((f) => f.endsWith('.in'))
  .sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
