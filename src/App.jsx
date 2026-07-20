import React, { useState, useMemo } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";
import {
  Grid3x3, BookOpen, Wrench, ClipboardCheck, ShoppingBag, Search, Play,
  CircleCheck, CircleX, CircleAlert, ChevronRight, ChevronLeft, Plus, Trash2, RotateCcw, Calculator, Boxes,
  ListChecks, FileSpreadsheet, Copy, Check, Lightbulb, MousePointerClick,
} from "lucide-react";

/* ============================================================================
   PLATFORM (Windows / Mac)  — a header toggle makes every OS-specific
   instruction resolve to the right platform. Default: Mac.
   • P("win text", "mac text")  marks a platform-specific value in the data.
   • rp(value, platform)        resolves it (passes plain values through).
   ============================================================================ */
const PlatformCtx = React.createContext("mac");
const usePlatform = () => React.useContext(PlatformCtx);
const P = (win, mac) => ({ __p: true, win, mac });
const rp = (v, plat) => (v && typeof v === "object" && v.__p ? (plat === "mac" ? v.mac : v.win) : v);

/* ============================================================================
   OPTIMIZATION ENGINE  —  two-phase simplex + branch-and-bound (tested)
   ============================================================================ */
function solveLP(objType, c, constraints) {
  const n = c.length;
  const m = constraints.length;
  const maximize = objType === "max";
  const cc = c.map((v) => (maximize ? v : -v));
  let rows = constraints.map((con) => {
    let coeffs = con.coeffs.slice(), op = con.op, rhs = con.rhs;
    if (rhs < 0) { coeffs = coeffs.map((v) => -v); rhs = -rhs; op = op === "<=" ? ">=" : op === ">=" ? "<=" : "="; }
    return { coeffs, op, rhs };
  });
  const slackInfo = [], artInfo = [];
  rows.forEach((r, i) => {
    if (r.op === "<=") slackInfo.push({ row: i, sign: 1 });
    else if (r.op === ">=") { slackInfo.push({ row: i, sign: -1 }); artInfo.push(i); }
    else artInfo.push(i);
  });
  const nSlack = slackInfo.length, nArt = artInfo.length, total = n + nSlack + nArt;
  const T = Array.from({ length: m }, () => new Array(total + 1).fill(0));
  for (let i = 0; i < m; i++) { for (let j = 0; j < n; j++) T[i][j] = rows[i].coeffs[j]; T[i][total] = rows[i].rhs; }
  let sc = n; const basis = new Array(m).fill(-1);
  slackInfo.forEach((s) => { T[s.row][sc] = s.sign; if (s.sign === 1) basis[s.row] = sc; sc++; });
  let ac = n + nSlack; const artCols = [];
  artInfo.forEach((r) => { T[r][ac] = 1; basis[r] = ac; artCols.push(ac); ac++; });
  const EPS = 1e-9;
  const pivot = (pr, pc) => {
    const pv = T[pr][pc];
    for (let j = 0; j < T[pr].length; j++) T[pr][j] /= pv;
    for (let i = 0; i < m; i++) { if (i === pr) continue; const f = T[i][pc]; if (Math.abs(f) < EPS) continue; for (let j = 0; j < T[i].length; j++) T[i][j] -= f * T[pr][j]; }
    basis[pr] = pc;
  };
  const runSimplex = (reduced) => {
    let iter = 0;
    while (iter++ < 5000) {
      const cb = basis.map((b) => reduced[b]);
      let pcol = -1;
      for (let j = 0; j < total; j++) {
        let z = 0; for (let i = 0; i < m; i++) z += cb[i] * T[i][j];
        if (z - reduced[j] < -1e-7) { pcol = j; break; }
      }
      if (pcol === -1) break;
      let prow = -1, best = Infinity;
      for (let i = 0; i < m; i++) {
        if (T[i][pcol] > EPS) {
          const ratio = T[i][total] / T[i][pcol];
          if (ratio < best - EPS || (Math.abs(ratio - best) < EPS && (prow === -1 || basis[i] < basis[prow]))) { best = ratio; prow = i; }
        }
      }
      if (prow === -1) return "unbounded";
      pivot(prow, pcol);
    }
    return "ok";
  };
  if (nArt > 0) {
    const w = new Array(total).fill(0); artCols.forEach((a) => (w[a] = -1));
    runSimplex(w);
    let infeas = 0; artCols.forEach((a) => { const bi = basis.indexOf(a); if (bi !== -1) infeas += T[bi][total]; });
    if (infeas > 1e-6) return { status: "infeasible" };
    artCols.forEach((a) => {
      const bi = basis.indexOf(a);
      if (bi !== -1) for (let j = 0; j < total; j++) if (!artCols.includes(j) && Math.abs(T[bi][j]) > EPS) { pivot(bi, j); break; }
    });
  }
  const obj = new Array(total).fill(0);
  for (let j = 0; j < n; j++) obj[j] = cc[j];
  artCols.forEach((a) => { for (let i = 0; i < m; i++) T[i][a] = basis[i] === a ? T[i][a] : 0; });
  if (runSimplex(obj) === "unbounded") return { status: "unbounded" };
  const x = new Array(n).fill(0);
  for (let i = 0; i < m; i++) if (basis[i] < n) x[basis[i]] = T[i][total];
  let objective = 0; for (let j = 0; j < n; j++) objective += c[j] * x[j];
  return { status: "optimal", x: x.map((v) => (Math.abs(v) < 1e-7 ? 0 : v)), objective };
}
const unitVec = (i, n) => { const a = new Array(n).fill(0); a[i] = 1; return a; };
function solveMIP(objType, c, constraints, intVars) {
  if (!intVars || intVars.length === 0) return solveLP(objType, c, constraints);
  let best = null, bestObj = objType === "max" ? -Infinity : Infinity, nodes = 0;
  const branch = (extra) => {
    if (nodes++ > 3000) return;
    const r = solveLP(objType, c, constraints.concat(extra));
    if (r.status !== "optimal") return;
    if (objType === "max" && r.objective <= bestObj + 1e-9) return;
    if (objType === "min" && r.objective >= bestObj - 1e-9) return;
    let frac = -1;
    for (const idx of intVars) if (Math.abs(r.x[idx] - Math.round(r.x[idx])) > 1e-6) { frac = idx; break; }
    if (frac === -1) { if ((objType === "max" && r.objective > bestObj) || (objType === "min" && r.objective < bestObj)) { bestObj = r.objective; best = r; } return; }
    const f = r.x[frac];
    branch(extra.concat({ coeffs: unitVec(frac, c.length), op: "<=", rhs: Math.floor(f) }));
    branch(extra.concat({ coeffs: unitVec(frac, c.length), op: ">=", rhs: Math.ceil(f) }));
  };
  branch([]);
  return best ? { ...best, status: "optimal" } : { status: "infeasible" };
}

/* ============================================================================
   COURSE COLOR CONVENTION  (Cholette / Winston-Albright)
   Blue = given inputs · Amber = decisions · Green = uncertain · Indigo = key output
   ============================================================================ */
const ROLE = {
  input:    { name: "Given input",  chip: "bg-blue-50 text-blue-700 border-blue-200",     dot: "bg-blue-500",    ring: "border-blue-300",   note: "A parameter you don't control (e.g. unit cost)." },
  decision: { name: "Decision",     chip: "bg-amber-50 text-amber-700 border-amber-300",  dot: "bg-amber-500",   ring: "border-amber-300",  note: "A value you choose (e.g. order quantity)." },
  uncertain:{ name: "Uncertain",    chip: "bg-green-50 text-green-700 border-green-200",   dot: "bg-green-500",   ring: "border-green-300",  note: "A value you can't know for sure (e.g. demand)." },
  output:   { name: "Key output",   chip: "bg-indigo-50 text-indigo-700 border-indigo-300",dot: "bg-indigo-500", ring: "border-indigo-400", note: "The number you care about most (e.g. profit)." },
  calc:     { name: "Calculation",  chip: "bg-slate-100 text-slate-600 border-slate-300", dot: "bg-slate-400",   ring: "border-slate-300",  note: "A formula linking inputs and decisions to outputs." },
};

/* ============================================================================
   FORMULA REFERENCE  — the how & the why for every function in the tutorial
   ============================================================================ */
const CATS = ["All", "Refs", "Core", "Lookup", "Math", "Stats", "Finance", "Sim", "Count"];
const FORMULAS = [
  { name: "Absolute & relative refs ($)", cat: "Refs", syntax: "$A$1 · A$1 · $A1 · A1",
    how: "A $ locks a row and/or column so it doesn't shift when you copy a formula. $A$1 is fully locked; A$1 locks the row; $A1 locks the column.",
    why: "Copying one formula across a whole table is the single biggest time-saver in modeling. Lock the cells that should stay put (a rate, a total) and leave the ones that should move.",
    ex: { setup: "Rate in $AD$31, monthly units in AD34:AG34", formula: "=$AD$30+$AD$31*AD34", result: "copy right → the rate stays locked, units advance" },
    watch: P("Press F4 while editing a reference to cycle A1 → $A$1 → A$1 → $A1.",
             "Press ⌘ T (or fn + F4) while editing a reference to cycle A1 → $A$1 → A$1 → $A1. On a Mac the plain F4 key usually controls screen brightness.") },
  { name: "Range names", cat: "Refs", syntax: "=SUMPRODUCT(Cost,Ship)",
    how: "Assign a readable name to a cell or block (Formulas ▸ Define Name, or type in the Name Box). The name then works anywhere a reference does.",
    why: "Named ranges make Solver models and long formulas readable — reviewers see 'Capacity' not '$L$14'. Cholette leans on these in the LP labs.",
    ex: { setup: "Name J7:J12 'UnitsSold'", formula: "=SUM(UnitsSold)", result: "self-documenting total" },
    watch: "Names are workbook-wide by default; don't reuse the same name on two sheets." },
  { name: "SUMPRODUCT", cat: "Core", syntax: "=SUMPRODUCT(array1, array2, …)",
    how: "Multiplies arrays element-by-element, then sums the products. SUMPRODUCT(A,B)=A1·B1 + A2·B2 + … The arrays must be the same shape.",
    why: "This is THE workhorse of optimization. Total cost, total revenue, and every LP objective is a cost-array × quantity-array collapsed to one number. Learn this cold.",
    ex: { setup: "Unit costs L7:N9, units shipped L12:N14", formula: "=SUMPRODUCT(L7:N9,L12:N14)", result: "total shipping cost in one cell" },
    watch: "Cholette's #1 observed mistake: confusing SUM with SUMPRODUCT. SUM adds one block; SUMPRODUCT cross-multiplies two." },
  { name: "SUMIF", cat: "Core", syntax: "=SUMIF(range, criteria, [sum_range])",
    how: "Adds up the cells in sum_range whose matching cell in range meets the criteria (e.g. \">100\", \"West\").",
    why: "Rolls up data by category without a pivot table — revenue by region, hours by project.",
    ex: { setup: "Regions in A2:A20, sales in B2:B20", formula: "=SUMIF(A2:A20,\"West\",B2:B20)", result: "total West sales" },
    watch: "Text criteria go in quotes; wildcards * and ? work inside them." },
  { name: "SUMIFS", cat: "Core", syntax: "=SUMIFS(sum_range, range1, crit1, range2, crit2, …)",
    how: "Like SUMIF but with multiple conditions, all of which must be true. Note the argument order flips: sum_range comes first.",
    why: "Real questions have two filters — 'West region AND Q3'. SUMIFS handles them in one formula.",
    ex: { setup: "Sales B, region A, quarter C", formula: "=SUMIFS(B2:B20,A2:A20,\"West\",C2:C20,\"Q3\")", result: "West + Q3 sales" },
    watch: "The sum_range is the FIRST argument here — opposite of SUMIF. Easy to swap by accident." },
  { name: "IF", cat: "Core", syntax: "=IF(test, value_if_true, value_if_false)",
    how: "Evaluates a logical test and returns one of two results.",
    why: "Business rules — reorder if stock is low, apply a discount over a threshold.",
    ex: { setup: "End inventory in L11", formula: "=IF(L11<=50,200-L11,0)", result: "order back up to 200 only when stock ≤ 50" },
    watch: "Cholette's warning: avoid IF inside optimization models (it's non-smooth and slow, and even Evolutionary Solver struggles). Substitute MAX or MIN whenever you can." },
  { name: "Nested IF / IFS", cat: "Core", syntax: "=IF(t1,a,IF(t2,b,c)) · =IFS(t1,a,t2,b,TRUE,c)",
    how: "Chain tests to return more than two outcomes. IFS (Excel 2019+) reads cleaner than deeply nested IFs.",
    why: "Grade bands, tiered pricing, letter classifications.",
    ex: { setup: "Score in L27", formula: "=IF(L27>=90,\"A\",IF(L27>=60,\"S\",\"U\"))", result: "A / S / U by score" },
    watch: "Order matters — test the highest band first, or everything collapses to the first true branch." },
  { name: "AND / OR", cat: "Core", syntax: "=AND(t1,t2,…) · =OR(t1,t2,…)",
    how: "AND is TRUE only if every test passes; OR is TRUE if any test passes. Usually nested inside IF.",
    why: "Combine conditions — sell only if the price rose three days straight.",
    ex: { setup: "L50:L52 = Up/Down flags", formula: "=IF(AND(L50=\"Up\",L51=\"Up\",L52=\"Up\"),\"Yes\",\"No\")", result: "Yes only on a 3-day run" },
    watch: "AND/OR return a single TRUE/FALSE; they don't test arrays element-by-element." },
  { name: "MIN", cat: "Core", syntax: "=MIN(number1, number2, …)",
    how: "Returns the smallest value in the list or range.",
    why: "Units sold can't exceed demand OR stock: MIN(order, demand). A smooth alternative to IF inside Solver.",
    ex: { setup: "Order 1450, demand 1500", formula: "=MIN(Order,Demand)", result: "1450 sold at full price" },
    watch: "MIN ignores text and blanks — a blank cell won't force a zero." },
  { name: "MAX", cat: "Core", syntax: "=MAX(number1, number2, …)",
    how: "Returns the largest value in the list or range.",
    why: "Leftovers = MAX(order − demand, 0). Guards against negatives without an IF.",
    ex: { setup: "Order 1450, demand 1500", formula: "=MAX(Order-Demand,0)", result: "0 leftover (demand exceeded supply)" },
    watch: "MAX(x,0) and MIN(x,cap) are the smooth, Solver-friendly way to clamp a value." },
  { name: "ROUND", cat: "Math", syntax: "=ROUND(number, num_digits)",
    how: "Rounds to a set number of decimals. 0 = whole number, negative = round to tens/hundreds.",
    why: "Present money to cents; report whole units.",
    ex: { setup: "459.735…", formula: "=ROUND(459.735,2)", result: "459.74" },
    watch: "Rounding for display is fine; rounding mid-model can silently break totals. Keep full precision until the end." },
  { name: "INT", cat: "Math", syntax: "=INT(number)",
    how: "Drops the fractional part, rounding DOWN toward negative infinity.",
    why: "Whole cases, whole people. But for optimization, make the variable Integer in Solver instead of wrapping it in INT.",
    ex: { setup: "7.9 batches", formula: "=INT(7.9)", result: "7" },
    watch: "INT(-2.3) = −3, not −2. It floors, it doesn't truncate." },
  { name: "ABS", cat: "Math", syntax: "=ABS(number)",
    how: "Returns the magnitude without the sign.",
    why: "Forecast error, distance from a target — you care how far off, not which side.",
    ex: { setup: "Actual − forecast = −12", formula: "=ABS(-12)", result: "12" },
    watch: "ABS is non-smooth at zero; like IF, it can trip up gradient Solver." },
  { name: "SQRT / SUMSQ", cat: "Math", syntax: "=SQRT(n) · =SUMSQ(range)",
    how: "SQRT is the square root. SUMSQ squares every value then sums them.",
    why: "Together they build distances and least-squares error — the backbone of the nonlinear regression week.",
    ex: { setup: "Residuals in R2:R20", formula: "=SQRT(SUMSQ(R2:R20))", result: "root-sum-of-squares" },
    watch: "SQRT of a negative returns #NUM!. Guard inputs first." },
  { name: "LN / EXP", cat: "Math", syntax: "=LN(number) · =EXP(number)",
    how: "LN is the natural log (base e); EXP raises e to a power. They're inverses.",
    why: "Growth/decay models, log-transformed regressions, some utility curves in the nonlinear chapters.",
    ex: { setup: "e ≈ 2.718", formula: "=LN(EXP(3))", result: "3" },
    watch: "LN(0) and LN(negative) are errors. LN needs strictly positive inputs." },
  { name: "VLOOKUP", cat: "Lookup", syntax: "=VLOOKUP(value, table, col_index, [range_lookup])",
    how: "Searches the FIRST column of a table for value, then returns a cell from col_index in the same row. FALSE = exact match; TRUE (or omitted) = nearest-below on a SORTED table.",
    why: "Grade bands, price tiers, tax brackets — map a number to a category.",
    ex: { setup: "Score L43, band table $O$43:$P$47 sorted ascending", formula: "=VLOOKUP(L43,$O$43:$P$47,2)", result: "letter grade for that score" },
    watch: "Lock the table with $ so it doesn't drift when copied. For approximate match the table MUST be sorted ascending, or you get silent wrong answers." },
  { name: "INDEX", cat: "Lookup", syntax: "=INDEX(array, row_num, [col_num])",
    how: "Returns the value at a given row/column position inside a block.",
    why: "The retrieval half of the INDEX/MATCH pair — pull a value once you know where it sits.",
    ex: { setup: "Prices in D2:D50", formula: "=INDEX(D2:D50,7)", result: "the 7th price" },
    watch: "INDEX takes positions (1,2,3…), not labels. Pair it with MATCH to find the position." },
  { name: "MATCH", cat: "Lookup", syntax: "=MATCH(value, lookup_array, [match_type])",
    how: "Returns the POSITION of value within a one-row/one-column range. Use 0 for an exact match.",
    why: "Finds the row or column number to feed INDEX.",
    ex: { setup: "Product list A2:A50", formula: "=MATCH(\"Widget\",A2:A50,0)", result: "row number of Widget" },
    watch: "match_type 0 = exact (unsorted OK); 1 = largest ≤ value (needs ascending sort)." },
  { name: "INDEX + MATCH", cat: "Lookup", syntax: "=INDEX(return_col, MATCH(value, lookup_col, 0))",
    how: "MATCH finds the row; INDEX returns the value there. A two-way lookup with independent lookup and return columns.",
    why: "The pro alternative to VLOOKUP: it can look leftward, survives inserted columns, and is faster on big tables.",
    ex: { setup: "IDs A:A, names C:C", formula: "=INDEX(C2:C50,MATCH(\"X-14\",A2:A50,0))", result: "name for ID X-14" },
    watch: "Interviewers love asking why INDEX/MATCH beats VLOOKUP — the answer is the leftward-lookup and column-insertion robustness." },
  { name: "OFFSET", cat: "Lookup", syntax: "=OFFSET(anchor, rows, cols, [height], [width])",
    how: "Returns a reference a set number of rows/cols away from an anchor, optionally a whole block.",
    why: "Rolling windows and dynamic ranges — a 'last 12 months' block that moves as data grows.",
    ex: { setup: "Anchor A1", formula: "=OFFSET(A1,3,2)", result: "the value in C4" },
    watch: "OFFSET is volatile — it recalculates on every change and can slow big workbooks." },
  { name: "INDIRECT", cat: "Lookup", syntax: "=INDIRECT(ref_text)",
    how: "Turns a text string into a live cell reference, so \"A\"&5 becomes A5.",
    why: "Build references from other cells — pull from a sheet whose name a dropdown selects.",
    ex: { setup: "B1 holds \"Sheet2\"", formula: "=INDIRECT(B1&\"!A1\")", result: "A1 of the named sheet" },
    watch: "Also volatile, and it breaks quietly if a referenced sheet is renamed." },
  { name: "AVERAGE", cat: "Stats", syntax: "=AVERAGE(range) · =AVERAGEIF(range,crit,[avg_range])",
    how: "Arithmetic mean of the numbers. AVERAGEIF averages only rows meeting a condition.",
    why: "Baselines and expected values across simulation trials.",
    ex: { setup: "Trial profits P2:P1001", formula: "=AVERAGE(P2:P1001)", result: "expected profit" },
    watch: "AVERAGE skips blanks and text but counts zeros — a real difference in survey data." },
  { name: "MEDIAN / PERCENTILE / QUARTILE", cat: "Stats", syntax: "=MEDIAN(r) · =PERCENTILE.INC(r,0.95) · =QUARTILE.INC(r,3)",
    how: "MEDIAN is the middle value; PERCENTILE returns the value below which k% of data falls; QUARTILE returns the 0/25/50/75/100 cut points.",
    why: "Simulation risk lives in the tails. A 95th-percentile cost or a Value-at-Risk says far more than the average.",
    ex: { setup: "Simulated costs C2:C1001", formula: "=PERCENTILE.INC(C2:C1001,0.95)", result: "cost you'll stay under 95% of the time" },
    watch: "Report medians alongside means for skewed outputs — the mean alone hides the tail." },
  { name: "STDEV / VAR", cat: "Stats", syntax: "=STDEV.S(r) · =VAR.S(r) · =STDEV.P(r)",
    how: "STDEV measures spread around the mean; VAR is its square. .S for a sample, .P for a full population.",
    why: "Spread IS risk. Two options with the same mean profit but different STDEV are very different decisions.",
    ex: { setup: "Returns R2:R101", formula: "=STDEV.S(R2:R101)", result: "volatility of returns" },
    watch: "Use .S when your rows are a sample (the usual case in simulation), .P only for a complete population." },
  { name: "CORREL", cat: "Stats", syntax: "=CORREL(array1, array2)",
    how: "Returns the correlation coefficient between two series, from −1 (opposite) through 0 (none) to +1 (lockstep).",
    why: "Diversification and driver analysis — do two products' demands move together?",
    ex: { setup: "Demand A vs demand B", formula: "=CORREL(A2:A50,B2:B50)", result: "e.g. 0.82 = strong positive" },
    watch: "Correlation ≠ causation, and it only detects LINEAR co-movement. A U-shaped relationship can read near zero." },
  { name: "PMT", cat: "Finance", syntax: "=PMT(rate, nper, pv, [fv], [type])",
    how: "Returns the fixed periodic payment on a loan. Use a per-period rate (annual/12) and total periods.",
    why: "Loan and lease models — the payment cell a Data Table sweeps across interest rates.",
    ex: { setup: "6.5%/yr, 36 mo, finance $15,000", formula: "=PMT(0.065/12,36,-15000)", result: "≈ $459.74 / month" },
    watch: "Enter pv as a negative (money you receive) so the payment comes out positive. Match rate and nper to the SAME period." },
  { name: "NPV / XNPV", cat: "Finance", syntax: "=NPV(rate, cash1, cash2, …)+initial · =XNPV(rate,values,dates)",
    how: "NPV discounts a stream of equally-spaced future cash flows to today's value. XNPV handles irregular dates.",
    why: "The standard capital-budgeting yardstick: is a project worth more than it costs today?",
    ex: { setup: "Rate 10%, flows in C2:C6, outlay in C1", formula: "=NPV(0.1,C2:C6)+C1", result: "project value today" },
    watch: "NPV assumes the FIRST cash flow is one period out — add the time-zero outlay OUTSIDE the function, don't include it inside." },
  { name: "IRR", cat: "Finance", syntax: "=IRR(values, [guess])",
    how: "Finds the discount rate that makes NPV zero. values must include the initial negative outlay.",
    why: "A rate-of-return summary that's easy to compare against a hurdle rate.",
    ex: { setup: "Outlay + 5 yrs of inflow in C1:C6", formula: "=IRR(C1:C6)", result: "e.g. 14.2%" },
    watch: "IRR can return multiple answers when signs flip more than once, and it can mislead when comparing projects — cross-check with NPV." },
  { name: "RAND", cat: "Sim", syntax: "=RAND()",
    how: "Returns a fresh uniform random number in [0,1) on every recalculation.",
    why: "The seed of Monte Carlo. Wrap it to drive any distribution; a Data Table then replays thousands of trials.",
    ex: { setup: "—", formula: "=RAND()", result: "e.g. 0.4173 (changes each F9)" },
    watch: "Volatile by design. Paste-Special ▸ Values to freeze a draw you want to keep." },
  { name: "RANDBETWEEN", cat: "Sim", syntax: "=RANDBETWEEN(bottom, top)",
    how: "Returns a random WHOLE number between the two bounds, inclusive.",
    why: "Discrete uncertainty — a die, daily orders, arrivals per hour.",
    ex: { setup: "—", formula: "=RANDBETWEEN(1,6)", result: "a die roll" },
    watch: "Only whole numbers and only uniform. For skewed or continuous demand, build from RAND instead." },
  { name: "NORM.INV", cat: "Sim", syntax: "=NORM.INV(RAND(), mean, std_dev)",
    how: "Feeding RAND() into the inverse-normal converts a uniform draw into a normally distributed one (inverse-transform sampling).",
    why: "Demand, returns, and lead times are often modeled as normal. This is the standard way to simulate them in plain Excel.",
    ex: { setup: "Demand ~ N(1500,200)", formula: "=NORM.INV(RAND(),1500,200)", result: "one simulated demand" },
    watch: "The normal has infinite tails — it can return negatives. Wrap in MAX(...,0) for quantities that can't go below zero." },
  { name: "COUNTIF / COUNTIFS", cat: "Count", syntax: "=COUNTIF(range,crit) · =COUNTIFS(r1,c1,r2,c2,…)",
    how: "Counts cells meeting one (COUNTIF) or several (COUNTIFS) conditions.",
    why: "Turn simulation output into probabilities — how many of 1,000 trials lost money?",
    ex: { setup: "Trial profits P2:P1001", formula: "=COUNTIF(P2:P1001,\"<0\")/1000", result: "probability of a loss" },
    watch: "Dividing a COUNTIF by the trial count is the idiom for estimating a probability from a simulation." },
  { name: "XLOOKUP", cat: "Lookup", syntax: "=XLOOKUP(value, lookup_array, return_array, [if_not_found], [match_mode])",
    how: "Searches lookup_array for value and returns the matching item from return_array. No column index, no sorting needed, and it can look in any direction.",
    why: "The modern replacement for VLOOKUP/INDEX-MATCH (Excel 365 / 2021). Cleaner, safer, and survives inserted columns.",
    ex: { setup: "IDs A2:A50, names C2:C50", formula: "=XLOOKUP(\"X-14\", A2:A50, C2:C50, \"not found\")", result: "the name for ID X-14" },
    watch: "Only on Excel 365 / 2021+. For older Excel, fall back to INDEX/MATCH. Add the if_not_found argument to avoid #N/A." },
  { name: "IFERROR", cat: "Core", syntax: "=IFERROR(value, value_if_error)",
    how: "Returns value normally, but swaps in value_if_error whenever value evaluates to any error (#DIV/0!, #N/A, #VALUE!, …).",
    why: "Keeps a model clean when a lookup misses or a divisor is zero — no cascading errors, and Solver/data tables don't choke on stray #N/A.",
    ex: { setup: "Rate = margin / sales, sales may be 0", formula: "=IFERROR(margin/sales, 0)", result: "0 instead of #DIV/0! when sales is blank" },
    watch: "IFERROR hides ALL errors — including real bugs. During development, leave errors visible; only wrap once you know why they occur." },
  { name: "SWITCH", cat: "Core", syntax: "=SWITCH(expr, val1, res1, val2, res2, …, [default])",
    how: "Compares one expression against a list of values and returns the first match's result — a cleaner alternative to deeply nested IFs when testing the same cell.",
    why: "Grade letters, region codes, scenario names — reads top-to-bottom instead of nesting.",
    ex: { setup: "Scenario code in B2 (1/2/3)", formula: "=SWITCH(B2, 1, \"Base\", 2, \"Best\", 3, \"Worst\", \"?\")", result: "the scenario label" },
    watch: "SWITCH tests equality only. For range bands (≥90 → A) use IFS or nested IF instead." },
  { name: "FILTER / SORT / UNIQUE", cat: "Lookup", syntax: "=FILTER(array, include) · =SORT(array) · =UNIQUE(array)",
    how: "Dynamic-array functions that spill a whole result set into neighbouring cells: FILTER keeps rows meeting a condition, SORT orders them, UNIQUE removes duplicates.",
    why: "Build live sub-tables (e.g. all trials that lost money, sorted) without manual copy/paste or a pivot — they recalc automatically.",
    ex: { setup: "Data A2:B100, flag in B", formula: "=FILTER(A2:A100, B2:B100=\"West\")", result: "just the West rows, spilled" },
    watch: "Excel 365 / 2021+ only. The result 'spills' — leave the cells below/right empty or you get #SPILL!." },
];

/* ============================================================================
   BUILDER PRESETS  — classic PMS problems, ready to solve
   ============================================================================ */
const PRESETS = {
  productMix: {
    label: "Product mix (LP)",
    objType: "max",
    varNames: ["Desks", "Tables"],
    c: [60, 40],
    constraints: [
      { coeffs: [4, 2], op: "<=", rhs: 40, label: "Cutting hours" },
      { coeffs: [2, 4], op: "<=", rhs: 40, label: "Finishing hours" },
    ],
    intVars: [],
    story: "Maximize profit from two products sharing limited cutting and finishing hours.",
  },
  transportation: {
    label: "Transportation (network LP)",
    objType: "min",
    varNames: ["P1→A", "P1→B", "P2→A", "P2→B"],
    c: [8, 6, 10, 4],
    constraints: [
      { coeffs: [1, 1, 0, 0], op: "<=", rhs: 20, label: "Plant 1 supply" },
      { coeffs: [0, 0, 1, 1], op: "<=", rhs: 30, label: "Plant 2 supply" },
      { coeffs: [1, 0, 1, 0], op: ">=", rhs: 25, label: "City A demand" },
      { coeffs: [0, 1, 0, 1], op: ">=", rhs: 25, label: "City B demand" },
    ],
    intVars: [],
    story: "Ship from 2 plants to 2 cities at least cost, respecting supply and demand.",
  },
  staffing: {
    label: "Staffing (integer)",
    objType: "min",
    varNames: ["FullTime", "PartTime"],
    c: [800, 500],
    constraints: [
      { coeffs: [40, 20], op: ">=", rhs: 300, label: "Weekly hours needed" },
      { coeffs: [1, 0], op: ">=", rhs: 3, label: "Min full-timers" },
    ],
    intVars: [0, 1],
    story: "Cover the hours at least payroll cost — you can't hire a fraction of a person, so both are integer.",
  },
  bakeSale: {
    label: "Bake sale — cookies (LP)",
    objType: "max",
    varNames: ["ChocChip", "Sugar"],
    c: [30, 20],
    constraints: [
      { coeffs: [2, 2], op: "<=", rhs: 6, label: "Butter (sticks)" },
      { coeffs: [1, 2], op: "<=", rhs: 5, label: "Sugar (cups)" },
      { coeffs: [2, 3], op: "<=", rhs: 12, label: "Eggs" },
      { coeffs: [1, 0], op: "<=", rhs: 2, label: "Choc chips (cups)" },
      { coeffs: [2, 2], op: "<=", rhs: 10, label: "Flour (cups)" },
    ],
    intVars: [],
    story: "The classic bake-sale problem — batches of chocolate-chip vs sugar cookies ($30 / $20 revenue per batch of 20) sharing five limited ingredients. Optimum: 2 choc-chip + 1 sugar = $80. Butter and chips are binding.",
  },
  bakeSaleFudge: {
    label: "Bake sale + fudge (3-var LP)",
    objType: "max",
    varNames: ["ChocChip", "Sugar", "Fudge"],
    c: [30, 20, 45],
    constraints: [
      { coeffs: [2, 2, 2], op: "<=", rhs: 6, label: "Butter (sticks)" },
      { coeffs: [1, 2, 1], op: "<=", rhs: 5, label: "Sugar (cups)" },
      { coeffs: [2, 3, 0], op: "<=", rhs: 12, label: "Eggs" },
      { coeffs: [1, 0, 2], op: "<=", rhs: 2, label: "Choc chips (cups)" },
      { coeffs: [2, 2, 0], op: "<=", rhs: 10, label: "Flour (cups)" },
    ],
    intVars: [],
    story: "Add a dense fudge-squares recipe ($45/batch). With three products you can't graph it — Solver finds 2 sugar + 1 fudge = $85, a $5 gain over the two-cookie plan.",
  },
  shipCarpets: {
    label: "Shipping carpets (transportation)",
    objType: "min",
    varNames: [
      "F1→D1", "F1→D2", "F1→D3", "F1→D4", "F1→D5",
      "F2→D1", "F2→D2", "F2→D3", "F2→D4", "F2→D5",
      "F3→D1", "F3→D2", "F3→D3", "F3→D4", "F3→D5",
      "F4→D1", "F4→D2", "F4→D3", "F4→D4", "F4→D5",
    ],
    c: [
      11, 16, 18, 22, 15,
      12, 24, 20, 21, 18,
      18, 17, 15, 15, 20,
      17, 22, 14, 24, 21,
    ],
    constraints: [
      { coeffs: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], op: "<=", rhs: 40, label: "F1 capacity" },
      { coeffs: [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], op: "<=", rhs: 50, label: "F2 capacity" },
      { coeffs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], op: "<=", rhs: 50, label: "F3 capacity" },
      { coeffs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], op: "<=", rhs: 60, label: "F4 capacity" },
      { coeffs: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], op: ">=", rhs: 30, label: "D1 demand" },
      { coeffs: [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0], op: ">=", rhs: 24, label: "D2 demand" },
      { coeffs: [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0], op: ">=", rhs: 42, label: "D3 demand" },
      { coeffs: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], op: ">=", rhs: 36, label: "D4 demand" },
      { coeffs: [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], op: ">=", rhs: 48, label: "D5 demand" },
    ],
    intVars: [],
    story: "The full carpet template: ship rolls (000's) from 4 factories to 5 distributors at least cost. Each arc is a decision, each node a constraint — the classic transportation network. Sanity check first: total capacity 200 ≥ total demand 180, so it's feasible. Solver finds the cheapest schedule at $2,660 (F1→D2 10, F1→D5 30, F2→D1 30, F2→D5 18, F3→D2 14, F3→D4 36, F4→D3 42). Because every capacity and demand is integer, the answer comes out whole with no integer constraints. Sensitivity: the shadow price on D3's demand is +$14 (one more roll to D3 costs $14); on F1's capacity it's −$3 (one more unit of F1 capacity cuts total cost by $3).",
  },
  assignJobs: {
    label: "Assigning jobs to machines (assignment)",
    objType: "min",
    varNames: [
      "M1→J1", "M1→J2", "M1→J3", "M1→J4",
      "M2→J1", "M2→J2", "M2→J3", "M2→J4",
      "M3→J1", "M3→J2", "M3→J3", "M3→J4",
      "M4→J1", "M4→J2", "M4→J3", "M4→J4",
      "M5→J1", "M5→J2", "M5→J3", "M5→J4",
    ],
    c: [
      14, 5, 8, 7,
      2, 12, 6, 5,
      7, 8, 3, 9,
      2, 4, 6, 10,
      5, 5, 4, 8,
    ],
    constraints: [
      { coeffs: [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], op: "=", rhs: 1, label: "Job 1 done once" },
      { coeffs: [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0], op: "=", rhs: 1, label: "Job 2 done once" },
      { coeffs: [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0], op: "=", rhs: 1, label: "Job 3 done once" },
      { coeffs: [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], op: "=", rhs: 1, label: "Job 4 done once" },
      { coeffs: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], op: "<=", rhs: 1, label: "M1 capability" },
      { coeffs: [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], op: "<=", rhs: 2, label: "M2 capability" },
      { coeffs: [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], op: "<=", rhs: 1, label: "M3 capability" },
      { coeffs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], op: "<=", rhs: 2, label: "M4 capability" },
      { coeffs: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], op: "<=", rhs: 1, label: "M5 capability" },
    ],
    intVars: [],
    story: "The assignment template: place 4 jobs on 5 machines to minimize total processing time. Each job must be done exactly once (=1); each machine can take at most its capability (M2 and M4 can do 2 jobs, the rest 1). Assignment is a special case of transportation, so even though the decisions are really 0/1, integer inputs give a 0/1 answer for free — no integer constraints needed. Solver's optimum is 14 time units: M2→J1, M4→J2, M3→J3, M2→J4 (M2 handles two jobs). To forbid a pairing (say M1 can't do J2), either drop that arc, give it a huge cost, or add a constraint pinning it to 0.",
  },
};

/* ============================================================================
   QUIZ BANK
   ============================================================================ */
const QUIZ = [
  { cat: "Framework", q: "In the modeling framework, which element is something the decision maker controls?", a: ["Given input", "Decision variable", "Uncertain variable", "Calculation"], correct: 1,
    why: "Decisions are the values you choose (order quantity, product mix). Inputs and uncertain variables are outside your control; calculations are formulas linking them." },
  { cat: "Framework", q: "Randy's shirt demand is best classified as which element?", a: ["Decision", "Given input", "Uncertain variable", "Key output"], correct: 2,
    why: "Demand isn't known in advance and isn't chosen — it's uncertain. That's exactly why Randy builds a model: to experiment with it." },
  { cat: "Framework", q: "Per the course color convention, what color codes a DECISION cell?", a: ["Blue", "Amber / orange", "Green", "Indigo"], correct: 1,
    why: "Blue = given inputs, amber/orange = decisions, green = uncertain variables, and the key output gets a box." },
  { cat: "Framework", q: "Which is NOT one of the four common problems with end-user spreadsheets?", a: ["Frequent bugs", "End-user overconfidence", "Too much documentation", "No use of methods to gain insight"], correct: 2,
    why: "Over-documentation isn't the problem — the four are bugs, overconfidence, inefficiency, and failure to use productive methods for insight." },
  { cat: "Formulas", q: "Which function computes a total shipping cost from a cost table and a quantity table?", a: ["SUM", "SUMPRODUCT", "SUMIF", "VLOOKUP"], correct: 1,
    why: "SUMPRODUCT cross-multiplies the two blocks element-by-element and sums them. Cholette flags SUM-vs-SUMPRODUCT as the most common mistake." },
  { cat: "Formulas", q: "You copy =$AD$31*AD34 one column to the right. What does it become?", a: ["=$AD$31*AE34", "=$AE$31*AE34", "=$AD$31*AD34", "=$AE$32*AE35"], correct: 0,
    why: "$AD$31 is locked and stays; AD34 is relative and advances to AE34." },
  { cat: "Formulas", q: "Cholette advises AGAINST which function inside an optimization model?", a: ["MIN", "MAX", "IF", "SUMPRODUCT"], correct: 2,
    why: "IF is non-smooth and computationally heavy; substitute MAX or MIN where possible so Solver behaves." },
  { cat: "Formulas", q: "For VLOOKUP to safely use APPROXIMATE match, the lookup table must be…", a: ["Sorted ascending by the first column", "Named", "On another sheet", "Formatted as currency"], correct: 0,
    why: "Approximate match walks the sorted first column for the nearest value below. Unsorted data gives silent wrong answers." },
  { cat: "Formulas", q: "Units actually sold = MIN(order, demand). If order = 1450 and demand = 1200, units sold =", a: ["1450", "1200", "250", "2650"], correct: 1,
    why: "You can't sell more than the 1200 demanded, even though you ordered 1450. MIN clamps it to demand." },
  { cat: "Formulas", q: "Which converts a uniform RAND() draw into a normally distributed value?", a: ["RANDBETWEEN", "NORM.INV(RAND(), mean, sd)", "AVERAGE", "STDEV.S"], correct: 1,
    why: "Inverse-transform sampling: feed RAND() into NORM.INV with your mean and standard deviation." },
  { cat: "LP", q: "In an LP, the cell you set to Max or Min in Solver is the…", a: ["Constraint", "Objective (target) cell", "Changing cells", "Data table"], correct: 1,
    why: "The objective (target) cell holds the value to optimize; the changing cells are the decisions; constraints bound them." },
  { cat: "LP", q: "'Cutting hours used ≤ 40' is an example of a…", a: ["Objective", "Decision variable", "Constraint", "Parameter you maximize"], correct: 2,
    why: "It limits the decisions — a constraint. Solver enforces it as 'Subject to'." },
  { cat: "LP", q: "In Solver, the 'By Changing Variable Cells' box should point to the…", a: ["Objective cell", "Decision cells", "Constraint right-hand sides", "Cost table"], correct: 1,
    why: "Those are the decision cells Solver is allowed to adjust to optimize the objective." },
  { cat: "Integer", q: "Why make a staffing variable Integer in Solver rather than wrap it in INT()?", a: ["INT is banned in Excel", "Solver can't read INT", "INT() distorts the optimization; the Integer constraint keeps the model honest", "There is no difference"], correct: 2,
    why: "Wrapping a decision in INT() creates a non-smooth objective Solver can mis-handle. Declaring the variable Integer lets Solver branch-and-bound correctly." },
  { cat: "Simulation", q: "To estimate the probability of a loss from 1,000 simulated profits, you'd use…", a: ["AVERAGE(range)", "COUNTIF(range,\"<0\")/1000", "MAX(range)", "SUMPRODUCT(range,range)"], correct: 1,
    why: "Count the losing trials and divide by the number of trials — that fraction estimates the probability." },
  { cat: "Simulation", q: "Which Excel feature replays a model across many random trials or input values at once?", a: ["Goal Seek", "A Data Table", "Conditional Formatting", "Flash Fill"], correct: 1,
    why: "A Data Table sweeps an input (or re-rolls RAND) across a column and records the output each time — the simple Monte Carlo engine in plain Excel." },
  { cat: "Testing", q: "Best FIRST sanity check on a new profit model is…", a: ["Add more constraints", "Test an extreme case (e.g. order zero → revenue should be zero)", "Change the fonts", "Delete the inputs"], correct: 1,
    why: "Extreme cases expose logic errors instantly: order zero must give zero revenue and zero variable cost." },
  { cat: "Testing", q: "Which tool highlights every cell that FEEDS INTO a selected formula?", a: ["Trace Dependents", "Trace Precedents", "Goal Seek", "Solver"], correct: 1,
    why: "Trace Precedents shows the inputs to a formula; Trace Dependents shows where a cell is used downstream." },
  { cat: "Data", q: "You add 20 new rows to a table that a PivotTable summarizes. The pivot still shows the old totals. Why?", a: ["PivotTables are broken", "A pivot is a snapshot — you must Refresh it", "You need to re-open the file", "Excel caps pivots at 100 rows"], correct: 1,
    why: "A PivotTable is a cached snapshot of the source. It does not auto-update; PivotTable Analyze ▸ Refresh (or Refresh All) pulls in new/changed rows." },
  { cat: "Data", q: "To see TOTAL sales for each region in a PivotTable, where do the fields go?", a: ["Region → Values, Sales → Rows", "Region → Filters, Sales → Columns", "Region → Rows, Sales → Values", "Region → Columns, Sales → Filters"], correct: 2,
    why: "The category you group BY goes in Rows (Region); the number being summarized goes in Values (Sales), defaulting to Sum." },
  { cat: "Data", q: "When would you use SUMIFS instead of a PivotTable for a roll-up?", a: ["When you never want it to change", "When the result must update live inside a model", "PivotTables can't sum", "SUMIFS is always better"], correct: 1,
    why: "SUMIFS is a live formula that recalculates automatically — ideal inside a model. A PivotTable is better for fast, ad-hoc exploration but must be refreshed." },
  { cat: "LP", q: "At the optimum, a constraint that is 'binding' is one that…", a: ["Has leftover capacity (slack)", "Is used right up to its limit — zero slack", "Never touches the feasible region", "Can only be an equality"], correct: 1,
    why: "A binding constraint helps form the optimal corner point and has zero slack — you use all of that resource. In the cookie problem, butter and chocolate chips are binding." },
  { cat: "LP", q: "In the cookie LP you use only 6 of 12 eggs. The eggs constraint is therefore…", a: ["Binding, shadow price > 0", "Non-binding, with 6 units of slack", "Infeasible", "The objective"], correct: 1,
    why: "You have 6 eggs left over, so the constraint isn't binding — it has slack of 6. More eggs wouldn't raise revenue, so its shadow price is 0." },
  { cat: "LP", q: "A 'redundant' constraint is a special non-binding constraint that…", a: ["Is always an equality", "Never touches the feasible region at all", "Has the largest shadow price", "Must be deleted before solving"], correct: 1,
    why: "A redundant constraint lies entirely outside (or behind) the feasible region, so it never affects the solution. Outside a 2-variable graph it's hard to spot, and Excel won't flag it." },
  { cat: "LP", q: "The LP 'divisibility' assumption says…", a: ["Every variable must be a whole number", "Fractional decision values are allowed (e.g. 1.5 batches)", "Constraints must divide evenly", "The objective must be divisible by 10"], correct: 1,
    why: "Divisibility permits non-integer decisions — 1.5 batches of cookies is fine. When fractions make no sense (3.5 tables, 2.4 nurses), you force Integer variables instead." },
  { cat: "LP", q: "Why include xc ≥ 0 and xs ≥ 0 in the cookie model?", a: ["To maximize faster", "Non-negativity — you can't bake a negative number of batches", "To make it integer", "Solver requires exactly two constraints"], correct: 1,
    why: "Non-negativity: real decisions like batches baked can't go below zero (you can't un-bake cookies into ingredients). Solver's 'Make Unconstrained Variables Non-Negative' box enforces this." },
  { cat: "LP", q: "In a 2-variable LP, the optimal solution is always found…", a: ["Deep inside the feasible region", "At a corner (vertex) where constraints intersect", "At the origin", "On the objective's y-intercept"], correct: 1,
    why: "The optimum is never in the interior — it sits on a corner point of the feasible region. That's why you can solve it by testing vertices or the two binding constraint equations." },
  { cat: "LP", q: "The rule 'chocolate:banana must not exceed 5:2' (C/B ≤ 5/2) becomes which LINEAR constraint?", a: ["C ≤ 5B", "2C − 5B ≤ 0", "5C − 2B ≤ 0", "C/B ≤ 2.5 (leave as-is)"], correct: 1,
    why: "Clear the fraction: C/B ≤ 5/2 → 2C ≤ 5B → 2C − 5B ≤ 0. Ratios look nonlinear but rearrange into a straight-line constraint." },
  { cat: "LP", q: "'At least 40% of the B+C diet must be chocolate (C)' becomes…", a: ["C ≥ 0.4", "0.6C − 0.4B ≥ 0", "C ≥ 0.4B", "0.4C − 0.6B ≥ 0"], correct: 1,
    why: "C ≥ 0.4(B+C) → C − 0.4B − 0.4C ≥ 0 → 0.6C − 0.4B ≥ 0. A percentage-of-total rule is linear once you expand and collect terms." },
  { cat: "LP", q: "'Max z = 2x − 3y' can be rewritten as the equivalent minimization…", a: ["Min z = 2x − 3y", "Min z = −2x + 3y", "Min z = 3y − 2x, then negate x", "You cannot convert max to min"], correct: 1,
    why: "Flip every sign: maximizing 2x − 3y is the same as minimizing −2x + 3y. You get the same decisions either way — only the reported objective sign differs." },
  { cat: "LP", q: "You have 3+ decision variables. Which Solver method fits a linear model?", a: ["GRG Nonlinear", "Simplex LP", "Evolutionary", "Goal Seek"], correct: 1,
    why: "Simplex LP is exact for linear models (and handles integers by branch-and-bound). Solver defaults to GRG Nonlinear — switch it to Simplex LP. Graphing only works up to 2 variables." },
  { cat: "Sensitivity", q: "A binding constraint's SHADOW PRICE tells you…", a: ["Its leftover capacity", "How much the objective improves per one extra unit of that resource", "The cost of the resource you paid", "Whether the problem is feasible"], correct: 1,
    why: "The shadow price is the marginal value of relaxing a binding constraint by one unit. Chocolate chips have the highest shadow price ($15), so an extra cup is worth up to $15 — don't pay more." },
  { cat: "Sensitivity", q: "Which constraints have a NON-ZERO shadow price?", a: ["All constraints", "Only binding constraints", "Only non-binding constraints", "Only equality constraints"], correct: 1,
    why: "Only binding constraints (zero slack) have non-zero shadow prices. A resource you don't fully use is worth $0 more — its shadow price is zero." },
  { cat: "Sensitivity", q: "The 'range of optimality' for an objective coefficient is the range over which…", a: ["The problem stays feasible", "The coefficient can change without changing the optimal decision values", "The shadow price stays constant", "Solver runs faster"], correct: 1,
    why: "Within the range of optimality the objective VALUE (z) changes but the optimal mix (xc, xs, xf) does not. Sugar cookies can range $15–$30 a batch before you'd re-plan." },
  { cat: "Sensitivity", q: "The 'allowable RHS increase/decrease' of a constraint defines its…", a: ["Range of optimality", "Range of feasibility — how far the RHS can move while the shadow price stays valid", "Slack", "Reduced cost"], correct: 1,
    why: "The range of feasibility is how much a right-hand-side can change before the shadow price (the marginal value) would change. Beyond it, you re-solve." },
  { cat: "Sensitivity", q: "In a Solver sensitivity report, the entry 1E+30 means…", a: ["A rounding error", "Infinity — effectively no limit", "The shadow price", "Thirty units"], correct: 1,
    why: "1E+30 is Excel's stand-in for infinity: the coefficient or RHS can increase without bound in that direction and the current solution structure still holds." },
  { cat: "Sensitivity", q: "'X + Y ≤ 2 and X + Y ≥ 4' together describe which solution state?", a: ["Unbounded", "Infeasible — no point satisfies both", "Multiple optima", "A unique optimum"], correct: 1,
    why: "No X, Y can be both ≤ 2 and ≥ 4, so the feasible region is empty — infeasible. Solver reports it can't find a feasible solution. Usually a sign error or over-tight limit." },
  { cat: "Sensitivity", q: "Maximizing with only 'X + Y ≥ 4, X ≥ 0, Y ≥ 0' and no upper limit gives…", a: ["Infeasible", "Unbounded — the objective grows to infinity", "Multiple optima", "z = 4"], correct: 1,
    why: "Nothing caps how large X and Y can grow, so a maximize objective runs off to infinity — unbounded. It means a real limiting constraint was left out." },
  { cat: "Network", q: "In a network model drawn as nodes and arcs, each ARC corresponds to a…", a: ["Constraint", "Decision variable", "Shadow price", "Objective"], correct: 1,
    why: "Each arc (a possible flow) is a decision variable; each node (a location) is a constraint. That clean structure is what makes network models special cases of LP." },
  { cat: "Network", q: "Why do network LPs often return whole-number answers WITHOUT integer constraints?", a: ["Solver rounds them", "The constraint coefficients are all 0s and 1s, so integer supplies/demands give integer solutions", "They use branch-and-bound automatically", "Network problems can't have fractions"], correct: 1,
    why: "The 0/1 coefficient structure means that when every capacity and demand is integer, the optimal solution is naturally integer — you skip the integer constraints and the model still solves cleanly." },
  { cat: "Network", q: "Before solving a transportation model, the key sanity check is…", a: ["Total demand must not exceed total capacity, or it's infeasible", "Every cost must be positive", "There must be equal plants and customers", "Demand must be an integer"], correct: 0,
    why: "If total demand exceeds total supply you can't meet everyone — the problem is infeasible. Check total capacity ≥ total demand before you even open Solver." },
  { cat: "Network", q: "The shadow price on a DEMAND constraint at distributor D3 is $14. That means…", a: ["Shipping to D3 costs $14 per roll", "Increasing D3's requirement by one unit adds $14 to total shipping cost", "D3 has $14 of slack", "You should stop shipping to D3"], correct: 1,
    why: "A demand-constraint shadow price is the marginal cost of one more unit of demand there — one extra roll required at D3 raises the optimal total cost by $14." },
  { cat: "Network", q: "The shadow price on factory F1's CAPACITY constraint is −$3. One more unit of F1 capacity…", a: ["Raises total cost by $3", "Reduces total shipping cost by $3", "Has no effect", "Makes the model infeasible"], correct: 1,
    why: "A capacity-constraint shadow price tells you how total cost moves as you add capacity. −$3 means an extra unit of F1 capacity lets you re-route and cut $3 from the optimal cost." },
  { cat: "Network", q: "The classic ASSIGNMENT model is best described as…", a: ["A brand-new model type", "A special case of the transportation problem with 0/1 decisions", "A nonlinear model", "A simulation"], correct: 1,
    why: "Assigning people/machines to tasks is transportation with supplies and demands of 1, so the decisions come out 0/1 — and integer inputs keep them integer for free." },
  { cat: "Network", q: "What makes a TRANSSHIPMENT model more general than a plain transportation model?", a: ["It allows negative costs", "It has intermediate nodes you can ship THROUGH (e.g. warehouses)", "It ignores capacity", "It must be nonlinear"], correct: 1,
    why: "Transshipment adds pass-through nodes — plants can ship via warehouses, and flow can move between same-tier nodes — which is exactly how real supply chains route product." },
  { cat: "Network", q: "In RedBrand's transshipment sheet, =SUMIF(Origin,H6,Flow) − SUMIF(Destination,H6,Flow) is safe for Solver because…", a: ["SUMIF is always allowed", "The range and criteria don't change while Solver runs — only Flow (the sum_range) does", "It's really a SUMPRODUCT", "IF statements are fine in Solver"], correct: 1,
    why: "Solver dislikes conditionals applied to the CHANGING cells. Here the range (Origin) and criteria (H6) are fixed; only Flow, the sum_range, varies — so the logic is smooth as far as Solver is concerned. =SUMIF(Flow,100,Flow) would NOT be okay." },
  { cat: "Network", q: "You can solve a SHORTEST-PATH problem as a transshipment model by…", a: ["Maximizing total distance", "Putting a supply of 1 at the start node and a demand of 1 at the end node", "Removing all intermediate nodes", "Setting every cost to 1"], correct: 1,
    why: "Push one unit of flow from start to finish: supply 1 at the origin, demand 1 at the destination, arc 'costs' = distances. The min-cost flow traces the shortest path." },
];

/* ============================================================================
   HOW-TO LIBRARY  — step-by-step Excel procedures
   step = { do, path?, formula?, why }
   ============================================================================ */
const HOWTOS = [
  { id: "solver", title: "Set up & run Solver", tag: "optimization",
    goal: "Turn a spreadsheet into an optimization that finds the best decisions for you.",
    steps: [
      { do: "Turn Solver on (one-time)", path: P("File ▸ Options ▸ Add-ins ▸ Manage: Excel Add-ins ▸ Go… ▸ ✓ Solver Add-in ▸ OK", "Tools ▸ Excel Add-ins… ▸ ✓ Solver Add-In ▸ OK"), why: P("Solver ships with Excel but is off by default. On Windows it lives under File ▸ Options. Once enabled it appears in the Data ribbon's Analyze group.", "Solver ships with Excel but is off by default. Mac Excel has no “File ▸ Options” — add-ins live under the Tools menu. Once ticked, Solver appears in the Data tab, same as Windows.") },
      { do: "Lay out the model first", why: "You need input cells, decision cells (start them at 0 or any guess), formula cells, and one objective cell in place before you open Solver." },
      { do: "Write the objective in one cell", formula: "=SUMPRODUCT(profit_per_unit, decisions)", why: "This single number is what Solver pushes up (Max) or down (Min)." },
      { do: "Open Solver and set the target", path: "Data ▸ Solver → Set Objective = your objective cell → choose Max or Min", why: "Tells Solver what 'best' means." },
      { do: "Point 'By Changing Variable Cells' at the decisions", path: "select your amber decision cells", why: "These are the only cells Solver is allowed to alter." },
      { do: "Add each business limit as a constraint", path: "Add ▸ Cell Reference (the 'used' cell) ▸ operator ▸ Constraint (the limit)", formula: "e.g.  $E$6  <=  $G$6", why: "Every capacity, demand, or budget line becomes one constraint row." },
      { do: "Pick the method and solve", path: "Select 'Simplex LP' ▸ ✓ Make Unconstrained Variables Non-Negative ▸ Solve ▸ Keep Solver Solution", why: "Simplex LP is exact for linear models and handles integers via branch-and-bound. The Solver dialog is identical on Windows and Mac." },
    ] },
  { id: "sumproduct", title: "Write a SUMPRODUCT objective", tag: "core",
    goal: "Collapse a whole cost or profit table into one number — the heart of every LP.",
    steps: [
      { do: "Put per-unit values on one line", why: "Costs or profits per unit — the 'rate' array." },
      { do: "Put decisions on a parallel line, same shape", why: "SUMPRODUCT needs the two ranges to have identical dimensions." },
      { do: "Multiply-and-sum in one formula", formula: "=SUMPRODUCT(profits, decisions)", why: "Excel multiplies each rate by its quantity and adds them — total profit in a single cell." },
      { do: "For a grid (transportation), use 2-D blocks", formula: "=SUMPRODUCT(cost_block, ship_block)", why: "Works on rectangles too, as long as both blocks are the same size." },
      { do: "Never substitute SUM", why: "SUM just adds the quantities and throws away the per-unit values — Cholette's most-flagged mistake." },
    ] },
  { id: "datatable1", title: "Build a one-way data table", tag: "sensitivity",
    goal: "Replay your model across many values of one input at once — the simple way to see sensitivity.",
    steps: [
      { do: "Build the model with the output in a cell", why: "e.g. a Profit cell that depends on an Order-quantity input." },
      { do: "List the input values to test down a column", why: "e.g. order quantities 0, 100, 200 … in a vertical list." },
      { do: "One row up and one column RIGHT of that list, link the output", formula: "=ProfitCell", why: "This corner cell tells the table which result to record." },
      { do: "Select the whole block", why: "From the top of your input column across to the linked output cell, down to the last input." },
      { do: "Run the data table", path: "Data ▸ What-If Analysis ▸ Data Table → leave Row input blank → Column input cell = the model's input cell (e.g. the order cell)", why: "Excel substitutes each listed value into that input and writes the output beside it." },
    ] },
  { id: "datatable2", title: "Build a two-way data table", tag: "sensitivity",
    goal: "See how the output responds to two inputs at once — a full sensitivity grid.",
    steps: [
      { do: "Link the output in the TOP-LEFT corner cell", formula: "=ProfitCell", why: "The corner anchors both axes of the grid." },
      { do: "List the first input's values down the left column", why: "e.g. unit prices going down." },
      { do: "List the second input's values across the top row", why: "e.g. quantities going across — they meet at the corner." },
      { do: "Select the entire grid, corner included", why: "Excel needs the full rectangle to fill it." },
      { do: "Run it with BOTH input cells set", path: "Data ▸ What-If ▸ Data Table → Row input cell = top-row input → Column input cell = left-column input", why: "Each interior cell is the output for that (row, column) pair." },
    ] },
  { id: "goalseek", title: "Use Goal Seek", tag: "what-if",
    goal: "Back-solve a single input to hit a target output — like break-even.",
    steps: [
      { do: "Have one formula output and one changeable input", why: "Goal Seek adjusts exactly one cell." },
      { do: "Open it", path: "Data ▸ What-If Analysis ▸ Goal Seek" },
      { do: "Fill the three boxes", path: "Set cell = the output · To value = your target · By changing cell = the input", why: "Excel iterates the input until the output equals the target." },
      { do: "Know the limit", why: "One input, one target only. For several inputs or constraints, use Solver instead." },
    ] },
  { id: "montecarlo", title: "Run a Monte Carlo simulation", tag: "simulation",
    goal: "Turn one uncertain input into a distribution of thousands of outcomes — in plain Excel.",
    steps: [
      { do: "Replace the uncertain input with a random draw", formula: "=MAX(NORM.INV(RAND(),1500,200),0)", why: "Inverse-transform sampling generates a normal demand; MAX(…,0) stops negatives. Use RANDBETWEEN for discrete uncertainty." },
      { do: "Let the output flow from it as usual", why: "Your profit formula now produces a random result each recalculation." },
      { do: "Number the trials 1…1000 down a column", why: "One row per simulated scenario." },
      { do: "One column right of the trial list, link the output", formula: "=ProfitCell", why: "Same corner-cell idea as a data table." },
      { do: "Run a data table with an EMPTY column input", path: "Data ▸ What-If ▸ Data Table → Column input cell = any blank cell", why: "The blank input forces a fresh recalculation — new RAND draws — for every trial row." },
      { do: "Summarize the distribution", formula: "=AVERAGE(results) · =STDEV.S(results) · =PERCENTILE.INC(results,0.05) · =COUNTIF(results,\"<0\")/1000", why: "Mean profit, its risk, a worst-5% value, and the probability of a loss." },
    ] },
  { id: "rangenames", title: "Name a range", tag: "core",
    goal: "Make formulas and Solver models read in English instead of $L$14.",
    steps: [
      { do: "Select the cell or block", why: "One cell or a whole array can take a name." },
      { do: "Type a name in the Name Box (left of the formula bar) and press Enter", why: "Fastest way; no spaces allowed in the name." },
      { do: "Use it anywhere a reference works", formula: "=SUMPRODUCT(Cost, Ship)", why: "Solver's Set Objective and constraints show the names too." },
      { do: "Create many at once from labels", path: "Select the table incl. headers ▸ Formulas ▸ Create from Selection ▸ ✓ Top row / Left column", why: "Names every column/row from its label in one move." },
    ] },
  { id: "audit", title: "Audit a formula", tag: "testing",
    goal: "Find the broken link fast instead of squinting at results.",
    steps: [
      { do: "Trace what feeds a cell", path: "click the suspect cell ▸ Formulas ▸ Trace Precedents", why: "Blue arrows point to every input the formula depends on." },
      { do: "Trace what a cell feeds", path: "Formulas ▸ Trace Dependents", why: "Shows the downstream blast radius before you change anything." },
      { do: "Flip the whole sheet to formulas", path: P("Ctrl + ~ (top-left key)", "⌃ ` (Control + the top-left key)"), why: "Eyeball the logic instead of the numbers." },
      { do: P("Pin key outputs while you work", "Audit with the trace arrows (no Watch Window on Mac)"),
        path: P("Formulas ▸ Watch Window ▸ Add Watch", "Formulas ▸ Trace Precedents / Trace Dependents"),
        why: P("See critical cells update as you edit elsewhere.", "Excel for Mac has no Watch Window — the trace arrows and Show Formulas are the Mac way to inspect a cell's logic.") },
      { do: "Clear the arrows when done", path: "Formulas ▸ Remove Arrows" },
    ] },
  { id: "pivot", title: "Build a PivotTable", tag: "core",
    goal: "Summarize a big table — totals by category, counts, averages — in seconds, no formulas.",
    steps: [
      { do: "Clean the source data first", why: "Every column needs a single header row, no blank rows/columns, and one fact per row (a 'tidy' table). PivotTables choke on merged cells and gaps." },
      { do: "Put the cursor in the data, then insert", path: P("Insert ▸ PivotTable ▸ (confirm range) ▸ New Worksheet ▸ OK", "Insert ▸ PivotTable ▸ (confirm range) ▸ OK"), why: "Excel auto-detects the whole table; a new sheet keeps the pivot away from your raw data." },
      { do: "Drag fields into the four zones", why: "In the PivotTable Fields pane: a category into ROWS (e.g. Region), another into COLUMNS (e.g. Quarter), the number you want summarized into VALUES (e.g. Sales), and anything you want to slice by into FILTERS." },
      { do: "Set how Values are summarized", path: P("click the value ▸ Value Field Settings ▸ Sum / Count / Average", "click the value ▸ i ▸ Summarize by ▸ Sum / Count / Average"), why: "Defaults to Sum for numbers and Count for text. Switch to Average, Max, % of total, etc. as needed." },
      { do: "Refresh after the data changes", path: P("PivotTable Analyze ▸ Refresh (Alt+F5)", "PivotTable ▸ Refresh"), why: "A pivot is a snapshot — it does NOT auto-update when you edit the source. Refresh (or Refresh All) to pull new rows/values." },
      { do: "Drill into any number", why: "Double-click a value cell and Excel drops the underlying rows onto a new sheet — perfect for auditing 'where did this total come from?'" },
      { do: "PivotTable or SUMIFS?", why: "SUMIFS/COUNTIFS do the same roll-ups with live formulas. Reach for a PivotTable for fast, ad-hoc exploration; use formulas when the result must update automatically inside a model." },
    ] },
  { id: "sensitivity", title: "Read Solver's sensitivity report", tag: "optimization",
    goal: "Go beyond the answer — learn how much each constraint and price actually matters.",
    steps: [
      { do: "Solve the model first", path: "Data ▸ Solver ▸ Solve", why: "The report is only offered once Solver finds an optimal (Simplex LP) solution." },
      { do: "Pick the report in the results dialog", path: "Solver Results ▸ Reports ▸ Sensitivity ▸ OK", why: "Excel adds a new 'Sensitivity Report' sheet. (Only Simplex LP produces it — not the nonlinear or evolutionary engines.)" },
      { do: "Read the shadow price", why: "For each binding constraint, the shadow price is how much the objective improves if you relax that limit by one unit — i.e. what one more hour of press time is worth." },
      { do: "Read the allowable ranges", why: "'Allowable increase/decrease' tells you how far a coefficient or a right-hand-side can move before the optimal mix changes. Wide ranges = a stable plan." },
      { do: "Spot the slack constraints", why: "A constraint with a zero shadow price isn't binding — you have spare capacity there, so buying more of it is wasted money." },
    ] },
  { id: "scenario", title: "Save & compare cases with Scenario Manager", tag: "what-if",
    goal: "Store named what-if cases (Base / Best / Worst) and flip between them or compare side by side.",
    steps: [
      { do: "Open it", path: "Data ▸ What-If Analysis ▸ Scenario Manager" },
      { do: "Add a scenario", path: "Add… ▸ name it (e.g. 'Best case') ▸ Changing cells = your input cells", why: "Each scenario stores one set of values for the cells you nominate — prices, demand, rates." },
      { do: "Enter the values for that case", why: "Type the inputs for this scenario; repeat Add for Base and Worst." },
      { do: "Flip between them", path: "select a scenario ▸ Show", why: "Excel drops that scenario's inputs into the sheet and everything recalculates — instant case switching." },
      { do: "Build a comparison", path: "Summary… ▸ choose your result cells", why: "Produces a table of every scenario's inputs and outputs on one sheet — the deliverable a manager actually wants." },
    ] },
  { id: "freeze", title: "Freeze a random draw", tag: "simulation",
    goal: "Capture a specific set of RAND() results so they stop changing on every edit.",
    steps: [
      { do: "Select the volatile cells", why: "Anything built on RAND()/RANDBETWEEN/NORM.INV re-rolls on every recalculation (F9) — fine while simulating, a problem when you want to keep a draw." },
      { do: "Copy them", path: P("Ctrl + C", "⌘ C") },
      { do: "Paste back as values", path: P("Paste Special ▸ Values (Ctrl+Alt+V, V)", "Edit ▸ Paste Special ▸ Values (⌘ ⌥ V)"), why: "Replaces the live formulas with the numbers they currently show, so the draw is frozen." },
      { do: "Turn calc to manual for big sims", path: P("Formulas ▸ Calculation Options ▸ Manual", "Excel ▸ Settings ▸ Calculation ▸ Manual"), why: "Stops thousand-row simulations from re-rolling every keystroke; press F9 to recalc on demand." },
    ] },
  { id: "formulate", title: "Formulate an LP model", tag: "optimization",
    goal: "Turn a word problem into decision variables, an objective, and constraints — before you touch Solver.",
    steps: [
      { do: "Identify the decision variables", why: "What choices are actually yours to make? In the bake sale it's the number of BATCHES of each cookie (xc, xs). Pick convenient units — batches, not individual cookies, keeps the numbers small." },
      { do: "Decide the direction: maximize or minimize", why: "Benefits (profit, revenue, units made) → maximize. Costs (dollars, hours, waste) → minimize. A mix of both usually maximizes — just get every sign right." },
      { do: "Write the objective in terms of the decisions", formula: "Max z = 30·xc + 20·xs", why: "One linear expression: each decision times its per-unit benefit or cost. This becomes the SUMPRODUCT objective cell." },
      { do: "Translate each limit into one linear constraint", formula: "num1·var1 + num2·var2  (≤ or ≥ or =)  num3", why: "Every resource, demand, or budget line has the same form. Butter: 2·xc + 2·xs ≤ 6. Only three relationship types exist: ≤, ≥, =." },
      { do: "Linearize tricky rules by clearing fractions", formula: "C/B ≤ 5/2  →  2C − 5B ≤ 0", why: "Ratios and percentages look nonlinear but rearrange to straight lines. '40% must be chocolate' → C ≥ 0.4(B+C) → 0.6C − 0.4B ≥ 0. A range like '20–30 oz' is two constraints (≥20 and ≤30)." },
      { do: "Add non-negativity", formula: "xc ≥ 0,  xs ≥ 0", why: "Real quantities can't go negative. In Solver this is the 'Make Unconstrained Variables Non-Negative' checkbox — leave it on unless a variable genuinely can be negative (e.g. shorting a stock)." },
    ] },
  { id: "graphical", title: "Solve a 2-variable LP graphically", tag: "optimization",
    goal: "See the feasible region and slide the objective line to the optimal corner — the intuition behind every solver.",
    steps: [
      { do: "Put one decision on each axis", why: "Only works with exactly two variables — e.g. xc on the horizontal axis, xs on the vertical. Three or more variables need Solver." },
      { do: "Plot each constraint as a line via its two intercepts", formula: "2xc + 2xs = 6  →  (xc=3, xs=0) and (xc=0, xs=3)", why: "Set one variable to 0 to get each axis crossing, then connect the two points. Fast and exact." },
      { do: "Shade the side each inequality allows", why: "For a ≤ constraint the feasible side is toward the origin; for ≥ it's away. Test the point (0,0): if it satisfies the inequality, the origin's side is allowed." },
      { do: "The overlap of every shaded side is the feasible region", why: "Only points inside ALL constraints at once are feasible. Its corners are the candidate optima." },
      { do: "Plot the objective at two easy z values", formula: "z = 60 and z = 120 for 30xc + 20xs", why: "Pick z values divisible by the coefficients so intercepts are clean. These lines are parallel — their common slope is the ratio of the objective coefficients." },
      { do: "Slide the objective line to the last feasible corner", why: "To maximize, push the line away from the origin until it just leaves the feasible region — the last corner it touches is optimal. To minimize, slide toward the origin." },
      { do: "Read or solve the optimal vertex", formula: "2xc + 2xs = 6  and  xc = 2  →  xc = 2, xs = 1", why: "Identify the two constraints that cross at that corner and solve them as two equations in two unknowns. For the cookies that's butter and chips → (2, 1), z = $80." },
    ] },
  { id: "transport", title: "Build a transportation model", tag: "optimization",
    goal: "Set up a supply→demand shipping network as an LP — a cost grid, a shipment grid, row/column roll-ups, and one SUMPRODUCT cost.",
    steps: [
      { do: "Lay the unit-cost grid as a rectangle", why: "Rows = sources (factories), columns = destinations (distributors). This is the blue given data — one cell per possible route. On a network diagram each of those cells is an arc." },
      { do: "Build a same-shaped shipment grid", why: "Directly below (or beside) the costs, make a matching amber block for units shipped — start every cell at 0. These are Solver's changing cells; each is a decision variable." },
      { do: "Roll up each source row and each destination column", formula: "=SUM(ship_row)  ·  =SUM(ship_col)", why: "A row sum is units shipped out of a factory (limited by its capacity); a column sum is units received by a distributor (must cover its demand). Each node becomes one constraint." },
      { do: "Write the objective as one SUMPRODUCT over both grids", formula: "=SUMPRODUCT(cost_grid, ship_grid)", why: "Cross-multiplies every route's cost by its shipment and totals them — the whole shipping bill in a single cell. SUMPRODUCT works on rectangles, not just rows." },
      { do: "Sanity-check feasibility BEFORE solving", formula: "=SUM(capacities) >= SUM(demands)", why: "If total demand exceeds total supply you can't satisfy everyone and Solver returns infeasible. Check total capacity ≥ total demand first." },
      { do: "Run Solver: minimize cost, supply ≤, demand ≥", path: "Data ▸ Solver → Set Objective = cost cell, To: Min → By Changing = the ship grid → Add row sums ≤ capacities and column sums ≥ demands → Simplex LP ▸ ✓ Non-Negative ▸ Solve", why: "Because every coefficient is a 0 or a 1, integer capacities and demands hand you a whole-number shipping plan with no integer constraints. (An assignment problem is the same recipe with supplies and demands of 1.)" },
    ] },
];

/* ============================================================================
   PROJECT MODE  — guided builds in the student's own Excel.
   Answer keys verified against the tested engine.
   ============================================================================ */
const PROJECTS = [
  {
    id: "printmix", title: "Print-shop product mix", level: "Core LP · Solver",
    brief: "Your shop profits $5 per poster and $4 per banner. Ink allows 2·posters + 1·banner ≤ 60 units. Press time allows 1·poster + 2·banners ≤ 48 hours. Finishing handles at most 25 posters. Find the mix that maximizes profit — build it in Excel and let Solver solve it.",
    stages: [
      { title: "Lay out the given inputs", role: "input",
        instr: ["Label and enter the profit per unit: $5 (poster), $4 (banner).", "Enter the three limits: Ink 60, Press 48, Poster cap 25.", "Colour these blue — you don't control them."], formulas: [] },
      { title: "Add the decision cells", role: "decision",
        instr: ["Make two cells for Posters and Banners — colour them amber.", "Start both at 0. Solver will change these."], formulas: [] },
      { title: "Compute what each limit uses", role: "calc",
        instr: ["Next to each limit, compute the amount used by the current decisions.", "Put the 'used' cell beside its ≤ limit so the constraint reads left-to-right."],
        formulas: [{ l: "Ink used", c: "=SUMPRODUCT({2;1}, decisions)" }, { l: "Press used", c: "=SUMPRODUCT({1;2}, decisions)" }, { l: "Posters used", c: "=Posters" }],
        note: "Tip: =2*Posters+1*Banners works too — SUMPRODUCT just scales better." },
      { title: "Build the objective", role: "output",
        instr: ["In one boxed cell, total the profit across both products.", "This is the cell Solver will maximize."],
        formulas: [{ l: "Total profit", c: "=SUMPRODUCT(profits, decisions)" }] },
      { title: "Run Solver", role: "calc",
        instr: ["Data ▸ Solver. Set Objective = the profit cell, To: Max.", "By Changing Variable Cells = your two decision cells.", "Add three constraints: Ink used ≤ 60, Press used ≤ 48, Posters ≤ 25.", "Method: Simplex LP · ✓ Non-Negative · Solve · Keep Solution."], formulas: [],
        checkpoint: { prompt: "What maximum profit does Solver report?", answer: 168, tol: 0.5, unit: "$",
          hints: [
            "Try the worksheet below — nudge posters and banners until ink and press are both fully used. That's usually where the optimum sits.",
            "At the optimum both ink (≤60) and press (≤48) are used to the limit; the poster cap (25) has room to spare.",
            "Solve 2P+B=60 and P+2B=48 together → P=24, B=12.",
          ],
          why: "Ink and press are the binding constraints. Solving them simultaneously gives Posters=24, Banners=12, so profit = 5·24 + 4·12 = 168. The poster cap (25) never bites.",
          solution: ["Binding limits: 2P+B=60 (ink) and P+2B=48 (press).", "Subtract: (2P+B)−(P+2B)=60−48 → P−B=12.", "With P+2B=48 → P=24, B=12.", "Profit = 5·24 + 4·12 = 120 + 48 = 168."],
          worksheet: {
            title: "Try a mix — watch the limits and profit",
            inputs: [{ key: "P", label: "Posters", def: 24 }, { key: "B", label: "Banners", def: 12 }],
            rows: (v) => [
              { label: "Ink used = 2·P + 1·B", value: 2 * v.P + v.B, limit: 60 },
              { label: "Press used = 1·P + 2·B", value: v.P + 2 * v.B, limit: 48 },
              { label: "Posters", value: v.P, limit: 25 },
            ],
            result: (v) => ({ label: "Profit = 5·P + 4·B", value: 5 * v.P + 4 * v.B, unit: "$" }),
          } } },
      { title: "Test the model", role: "calc",
        instr: ["Extreme case: set both decisions to 0 — profit must read exactly 0.", "Which constraints are binding at the optimum? (Ink and Press are both fully used; the poster cap is slack.)", "Bump a profit coefficient up by $1 and confirm total profit rises."], formulas: [] },
    ],
  },
  {
    id: "newsvendor", title: "Randy's ordering decision", level: "What-if · Data table",
    brief: "Randy pays $750 fixed + $8 per shirt, sells at $18, and salvages leftovers at $6. Build his profit model, verify one scenario, then use a one-way data table to find the best order quantity when demand turns out to be exactly 1,500.",
    stages: [
      { title: "Enter the inputs", role: "input",
        instr: ["Blue cells: Fixed $750, Variable $8, Price $18, Salvage $6."], formulas: [] },
      { title: "Add the decision and the uncertainty", role: "decision",
        instr: ["Amber: Order quantity — start at 1450.", "Green: Demand — start at 1500."], formulas: [] },
      { title: "Write the calculations", role: "calc",
        instr: ["Build the chain from decision + demand to profit.", "Notice there's no IF — MIN and MAX keep it smooth."],
        formulas: [{ l: "Units sold", c: "=MIN(Order, Demand)" }, { l: "Leftover", c: "=MAX(Order-Demand, 0)" }, { l: "Revenue", c: "=18*Sold + 6*Leftover" }, { l: "Total cost", c: "=750 + 8*Order" }, { l: "Profit", c: "=Revenue - TotalCost" }],
        checkpoint: { prompt: "With Order 1450 and Demand 1500, what profit does your model show?", answer: 13750, tol: 1, unit: "$",
          hints: [
            "Use the worksheet below with Order 1450 and Demand 1500 — watch each line compute.",
            "You order 1450 but demand is 1500, so you sell all 1450 and have 0 leftovers.",
            "Revenue = 18·1450 = 26,100. Cost = 750 + 8·1450 = 12,350. Profit = revenue − cost.",
          ],
          why: "Order (1450) is below demand (1500), so every shirt sells at full price and nothing is salvaged. Profit = 26,100 − 12,350 = 13,750.",
          solution: ["Units sold = MIN(1450, 1500) = 1450.", "Leftover = MAX(1450−1500, 0) = 0.", "Revenue = 18·1450 + 6·0 = 26,100.", "Cost = 750 + 8·1450 = 12,350.", "Profit = 26,100 − 12,350 = 13,750."],
          worksheet: {
            title: "Randy's profit — plug in your numbers",
            inputs: [{ key: "order", label: "Order", def: 1450 }, { key: "demand", label: "Demand", def: 1500 }],
            rows: (v) => [
              { label: "Units sold = MIN(order, demand)", value: Math.min(v.order, v.demand) },
              { label: "Leftover = MAX(order − demand, 0)", value: Math.max(v.order - v.demand, 0) },
              { label: "Revenue = 18·sold + 6·leftover", value: 18 * Math.min(v.order, v.demand) + 6 * Math.max(v.order - v.demand, 0), unit: "$" },
              { label: "Cost = 750 + 8·order", value: 750 + 8 * v.order, unit: "$" },
            ],
            result: (v) => ({ label: "Profit = Revenue − Cost", value: (18 * Math.min(v.order, v.demand) + 6 * Math.max(v.order - v.demand, 0)) - (750 + 8 * v.order), unit: "$" }),
          } } },
      { title: "Sweep the order with a data table", role: "calc",
        instr: ["List order quantities 0, 100, 200 … 2500 down a column.", "One cell up-right of the list, link the output: =Profit.", "Select the block ▸ Data ▸ What-If ▸ Data Table ▸ Column input cell = the Order cell.", "Keep Demand fixed at 1500."], formulas: [{ l: "Table link cell", c: "=Profit" }],
        checkpoint: { prompt: "Reading the table, which order gives the highest profit at demand 1500?", answer: 1500, tol: 50, unit: "shirts",
          hints: [
            "Drag the Order in the worksheet below and watch profit rise, peak, then fall.",
            "Below demand, each extra shirt earns 18 − 8 = $10. Above demand, an extra shirt earns only 6 − 8 = −$2 (a loss).",
            "So profit climbs right up to order = demand, then declines. The peak is exactly at demand.",
          ],
          why: "Every shirt up to demand nets +$10; every shirt past demand nets −$2. Profit therefore peaks exactly where order meets demand — 1,500 — at $14,250.",
          solution: ["Marginal shirt below demand: +18 − 8 = +$10 → keep ordering.", "Marginal shirt above demand: +6 − 8 = −$2 → stop.", "Peak is at order = demand = 1,500.", "Peak profit = 10·1,500 − 750 = 14,250."],
          worksheet: {
            title: "Sweep the order (demand fixed at 1500)",
            inputs: [{ key: "order", label: "Order", def: 1500, min: 0, max: 2500, step: 50 }],
            rows: (v) => [
              { label: "Units sold", value: Math.min(v.order, 1500) },
              { label: "Leftover (salvaged at $6)", value: Math.max(v.order - 1500, 0) },
            ],
            result: (v) => ({ label: "Profit at demand 1500", value: (18 * Math.min(v.order, 1500) + 6 * Math.max(v.order - 1500, 0)) - (750 + 8 * v.order), unit: "$" }),
          } } },
      { title: "Read the trade-off", role: "output",
        instr: ["The profit curve kinks at order = demand — that bend is the newsvendor trade-off.", "The peak profit at order 1500 is $14,250 (=10·1500 − 750).", "Since salvage ($6) < cost ($8), never order past what you expect to sell."], formulas: [] },
    ],
  },
  {
    id: "shipping", title: "Two-plant shipping network", level: "Network LP · Solver",
    brief: "Plant 1 supplies 20 units, Plant 2 supplies 30. City A needs 25, City B needs 25. Unit costs: P1→A $8, P1→B $6, P2→A $10, P2→B $4. Build the shipment grid and minimize total cost.",
    stages: [
      { title: "Enter the cost grid", role: "input",
        instr: ["Blue 2×2 table of unit costs (rows = plants, columns = cities)."], formulas: [] },
      { title: "Build the shipment grid", role: "decision",
        instr: ["Amber 2×2 block for units shipped — start every cell at 0.", "These four cells are Solver's changing cells."], formulas: [] },
      { title: "Add supply and demand roll-ups", role: "calc",
        instr: ["Sum each plant's row (units shipped out) beside its supply limit.", "Sum each city's column (units received) above its demand."],
        formulas: [{ l: "Shipped from a plant", c: "=SUM(ship_row)" }, { l: "Received by a city", c: "=SUM(ship_col)" }] },
      { title: "Build the objective", role: "output",
        instr: ["Total the cost across the whole grid in one boxed cell."],
        formulas: [{ l: "Total cost", c: "=SUMPRODUCT(cost_grid, ship_grid)" }] },
      { title: "Run Solver", role: "calc",
        instr: ["Set Objective = total cost, To: Min. By Changing = the four ship cells.", "Constraints: each plant's Shipped ≤ its Supply; each city's Received ≥ its Demand.", "Simplex LP · ✓ Non-Negative · Solve."], formulas: [],
        checkpoint: { prompt: "What minimum total shipping cost does Solver report?", answer: 310, tol: 0.5, unit: "$",
          hints: [
            "Use the worksheet — try to satisfy every city's demand while keeping each plant within supply, favoring the cheapest lanes.",
            "P2→B is the cheapest lane at $4 — send as much there as you can (City B needs 25, Plant 2 has 30).",
            "Fill P2→B=25, then cover City A (25) with P2→A=5 (Plant 2's remainder) and P1→A=20. That's the optimum: 20, 0, 5, 25.",
          ],
          why: "Greedily using the cheapest lanes while respecting supply/demand gives P1→A=20, P1→B=0, P2→A=5, P2→B=25. Cost = 8·20 + 6·0 + 10·5 + 4·25 = 310.",
          solution: ["Cheapest lane P2→B ($4): ship 25 (all of City B).", "Plant 2 has 5 left → P2→A = 5.", "City A still needs 20 → P1→A = 20 (Plant 1's full supply).", "Cost = 8·20 + 10·5 + 4·25 = 160 + 50 + 100 = 310."],
          worksheet: {
            title: "Try a shipping plan — watch cost & feasibility",
            inputs: [
              { key: "p1a", label: "P1→A ($8)", def: 20 }, { key: "p1b", label: "P1→B ($6)", def: 0 },
              { key: "p2a", label: "P2→A ($10)", def: 5 }, { key: "p2b", label: "P2→B ($4)", def: 25 },
            ],
            rows: (v) => [
              { label: "Plant 1 shipped = P1→A + P1→B", value: v.p1a + v.p1b, limit: 20 },
              { label: "Plant 2 shipped = P2→A + P2→B", value: v.p2a + v.p2b, limit: 30 },
              { label: "City A received = P1→A + P2→A  (need 25)", value: v.p1a + v.p2a, need: 25 },
              { label: "City B received = P1→B + P2→B  (need 25)", value: v.p1b + v.p2b, need: 25 },
            ],
            result: (v) => ({ label: "Total cost = 8·P1A + 6·P1B + 10·P2A + 4·P2B", value: 8 * v.p1a + 6 * v.p1b + 10 * v.p2a + 4 * v.p2b, unit: "$" }),
          } } },
      { title: "Sanity-check the flows", role: "calc",
        instr: ["Confirm each column received exactly its demand (25 and 25).", "Confirm no plant ships more than its supply (20 and 30).", "The cheapest lane ($4, P2→B) should be used to the hilt — it is."], formulas: [] },
    ],
  },
  {
    id: "pivotsales", title: "Roll up sales with a PivotTable", level: "Data · PivotTable",
    brief: "You have 8 sales records (Region, Quarter, Sales): West/Q1 120, East/Q1 90, West/Q2 150, East/Q2 110, West/Q1 80, East/Q2 60, West/Q3 200, East/Q3 130. Build a PivotTable that totals Sales by Region, then answer a couple of questions from it.",
    stages: [
      { title: "Enter the raw data", role: "input",
        instr: ["Type the 8 rows into a clean table with headers Region | Quarter | Sales (one header row, no blank rows).", "This is your source — blue, given data you don't control."], formulas: [] },
      { title: "Insert the PivotTable", role: "calc",
        instr: ["Click anywhere in the data, then Insert ▸ PivotTable ▸ New Worksheet ▸ OK.", "Excel auto-selects the whole table."], formulas: [] },
      { title: "Summarize Sales by Region", role: "calc",
        instr: ["Drag Region into ROWS.", "Drag Sales into VALUES — it should read 'Sum of Sales'.", "You now have a total for West and a total for East."],
        checkpoint: { prompt: "What is the total Sales for the West region?", answer: 550, tol: 0.5, unit: "$",
          hints: [
            "Use the worksheet below — enter the four West-region Sales values and watch them total.",
            "The West rows are: 120, 150, 80, 200.",
            "Sum them: 120 + 150 + 80 + 200.",
          ],
          why: "There are four West records — 120, 150, 80, 200 — and the pivot's 'Sum of Sales' for the West row adds them to 550.",
          solution: ["Find every West row: Q1 120, Q2 150, Q1 80, Q3 200.", "Add them: 120 + 150 + 80 + 200 = 550.", "That's the number in the pivot's West row."],
          worksheet: {
            title: "Add the West-region rows",
            inputs: [{ key: "a", label: "West #1", def: 120 }, { key: "b", label: "West #2", def: 150 }, { key: "c", label: "West #3", def: 80 }, { key: "d", label: "West #4", def: 200 }],
            rows: (v) => [
              { label: "West rows entered", value: 4 },
            ],
            result: (v) => ({ label: "Sum of West Sales", value: v.a + v.b + v.c + v.d, unit: "$" }),
          } } },
      { title: "Check the grand total & drill in", role: "output",
        instr: ["The PivotTable's grand total is West + East.", "Double-click the grand-total cell — Excel drops all 8 source rows onto a new sheet so you can audit them.", "Remember: if you add rows to the source, you must Refresh the pivot."],
        checkpoint: { prompt: "What is the grand total of all Sales?", answer: 940, tol: 0.5, unit: "$",
          hints: [
            "Grand total = West total + East total.",
            "West = 550 (from the last step). Now total the East rows: 90, 110, 60, 130.",
            "East = 390, so grand total = 550 + 390.",
          ],
          why: "West sums to 550 and East (90 + 110 + 60 + 130) sums to 390, so every record together totals 940 — the pivot's grand total.",
          solution: ["West total = 550.", "East rows: 90 + 110 + 60 + 130 = 390.", "Grand total = 550 + 390 = 940."],
          worksheet: {
            title: "Region subtotals → grand total",
            inputs: [{ key: "west", label: "West total", def: 550 }, { key: "east", label: "East total", def: 390 }],
            rows: (v) => [
              { label: "West region", value: v.west, unit: "$" },
              { label: "East region", value: v.east, unit: "$" },
            ],
            result: (v) => ({ label: "Grand total = West + East", value: v.west + v.east, unit: "$" }),
          } } },
    ],
  },
  {
    id: "bakesale", title: "Bake-sale cookie mix", level: "Core LP · formulation → Solver",
    brief: "Tomorrow's charity bake sale. Chocolate-chip cookies earn $30 a batch (20 × $1.50), sugar cookies $20 a batch (20 × $1.00). You have 6 sticks butter, 5 cups sugar, 12 eggs, 2 cups chocolate chips, 10 cups flour. Recipes per batch — Choc-chip: 2 butter, 1 sugar, 2 eggs, 1 chips, 2 flour · Sugar: 2 butter, 2 sugar, 3 eggs, 0 chips, 2 flour. Formulate the LP and let Solver find the money-maximizing mix — then add a fudge recipe and watch the plan change.",
    stages: [
      { title: "Choose the decision variables", role: "decision",
        instr: ["Two amber cells: xc = batches of chocolate-chip, xs = batches of sugar cookies. Start both at 0.", "Work in BATCHES, not individual cookies — the recipes are per batch, so the numbers stay small and clean.", "Divisibility is fine here: 1.5 batches is allowed (you can make a partial batch)."], formulas: [] },
      { title: "Enter the givens", role: "input",
        instr: ["Blue cells — revenue per batch: $30 (choc-chip), $20 (sugar).", "Blue cells — the five ingredient limits: butter 6, sugar 5, eggs 12, chips 2, flour 10.", "These are parameters: fixed, not yours to change."], formulas: [] },
      { title: "Build the objective", role: "output",
        instr: ["One boxed cell totals bake-sale revenue across both products.", "This is the cell Solver maximizes."],
        formulas: [{ l: "Max revenue", c: "=SUMPRODUCT({30;20}, decisions)" }],
        note: "Direction check: both coefficients are benefits (revenue), so keep them positive and MAXIMIZE." },
      { title: "Add the five ingredient constraints", role: "calc",
        instr: ["Beside each limit, compute how much the current mix uses, then constrain used ≤ limit.", "Every constraint has the same shape: coef·xc + coef·xs ≤ limit."],
        formulas: [
          { l: "Butter used ≤ 6", c: "=2*xc + 2*xs" },
          { l: "Sugar used ≤ 5", c: "=1*xc + 2*xs" },
          { l: "Eggs used ≤ 12", c: "=2*xc + 3*xs" },
          { l: "Chips used ≤ 2", c: "=1*xc + 0*xs" },
          { l: "Flour used ≤ 10", c: "=2*xc + 2*xs" },
        ],
        note: "Plus non-negativity: xc ≥ 0, xs ≥ 0 — you can't bake negative batches." },
      { title: "Run Solver (two cookies)", role: "calc",
        instr: ["Data ▸ Solver. Set Objective = revenue cell, To: Max.", "By Changing Variable Cells = xc and xs.", "Add the five constraints: Butter ≤ 6, Sugar ≤ 5, Eggs ≤ 12, Chips ≤ 2, Flour ≤ 10.", "Method: Simplex LP · ✓ Non-Negative · Solve · Keep Solution."], formulas: [],
        checkpoint: { prompt: "What maximum revenue does Solver report for the two-cookie problem?", answer: 80, tol: 0.5, unit: "$",
          hints: [
            "Use the worksheet — raise the batches until butter (≤6) and chips (≤2) are both fully used. That corner is usually the optimum.",
            "At the optimum, butter and chocolate chips are the binding constraints; sugar, eggs and flour all have slack to spare.",
            "Chips forces xc = 2; then butter 2·xc + 2·xs = 6 gives xs = 1.",
          ],
          why: "Butter and chocolate chips are the binding constraints. Chips caps choc-chip at xc = 2; butter 2(2) + 2xs = 6 gives xs = 1. Revenue = 30·2 + 20·1 = $80. Sugar, eggs and flour are non-binding (leftovers).",
          solution: ["Binding constraints: chips (xc ≤ 2) and butter (2xc + 2xs ≤ 6).", "Chips binding → xc = 2.", "Butter binding → 2(2) + 2xs = 6 → xs = 1.", "Revenue = 30·2 + 20·1 = $80 (between the $60 and $120 objective lines)."],
          worksheet: {
            title: "Try a batch plan — watch each ingredient and the revenue",
            inputs: [{ key: "xc", label: "Choc-chip batches", def: 2 }, { key: "xs", label: "Sugar batches", def: 1 }],
            rows: (v) => [
              { label: "Butter = 2·xc + 2·xs", value: 2 * v.xc + 2 * v.xs, limit: 6 },
              { label: "Sugar = 1·xc + 2·xs", value: v.xc + 2 * v.xs, limit: 5 },
              { label: "Eggs = 2·xc + 3·xs", value: 2 * v.xc + 3 * v.xs, limit: 12 },
              { label: "Chips = 1·xc", value: v.xc, limit: 2 },
              { label: "Flour = 2·xc + 2·xs", value: 2 * v.xc + 2 * v.xs, limit: 10 },
            ],
            result: (v) => ({ label: "Revenue = 30·xc + 20·xs", value: 30 * v.xc + 20 * v.xs, unit: "$" }),
          } } },
      { title: "Extend the model: add fudge squares", role: "decision",
        instr: ["A third amber cell: xf = batches of fudge squares, at $45 a batch.", "Fudge recipe per batch: 2 butter, 1 sugar, 0 eggs, 2 chips, 0 flour.", "New objective: Max 30·xc + 20·xs + 45·xf. Update every constraint to include the fudge column, then re-run Solver.", "Three variables can't be graphed — this is exactly why we hand it to Solver."],
        formulas: [{ l: "New objective", c: "=SUMPRODUCT({30;20;45}, decisions)" }, { l: "Chips used ≤ 2", c: "=1*xc + 0*xs + 2*xf" }],
        checkpoint: { prompt: "After adding fudge squares, what maximum revenue does Solver report?", answer: 85, tol: 0.5, unit: "$",
          hints: [
            "Fudge is dense and pays $45 — try dropping choc-chip cookies entirely and spending the chocolate chips on fudge instead.",
            "Solver picks 0 choc-chip, 2 sugar, 1 fudge. Butter, sugar and chips all become binding.",
            "Check the revenue: 30·0 + 20·2 + 45·1 = 40 + 45.",
          ],
          why: "Fudge's $45/batch beats blending in choc-chip cookies. The optimum shifts to 2 sugar + 1 fudge (xc = 0), using all the butter, sugar and chips — revenue $85, a $5 gain over the two-cookie plan. The optimum is still a corner point, just harder to see with three variables.",
          solution: ["New objective: Max 30xc + 20xs + 45xf.", "Solver returns xc = 0, xs = 2, xf = 1.", "Binding now: butter (6), sugar (5) and chips (2) all fully used.", "Revenue = 20·2 + 45·1 = $85 — up $5 from the $80 two-cookie plan."],
          worksheet: {
            title: "Add fudge — find the better mix",
            inputs: [{ key: "xc", label: "Choc-chip", def: 0 }, { key: "xs", label: "Sugar", def: 2 }, { key: "xf", label: "Fudge", def: 1 }],
            rows: (v) => [
              { label: "Butter = 2·xc + 2·xs + 2·xf", value: 2 * v.xc + 2 * v.xs + 2 * v.xf, limit: 6 },
              { label: "Sugar = xc + 2·xs + xf", value: v.xc + 2 * v.xs + v.xf, limit: 5 },
              { label: "Chips = xc + 2·xf", value: v.xc + 2 * v.xf, limit: 2 },
              { label: "Eggs = 2·xc + 3·xs", value: 2 * v.xc + 3 * v.xs, limit: 12 },
              { label: "Flour = 2·xc + 2·xs", value: 2 * v.xc + 2 * v.xs, limit: 10 },
            ],
            result: (v) => ({ label: "Revenue = 30·xc + 20·xs + 45·xf", value: 30 * v.xc + 20 * v.xs + 45 * v.xf, unit: "$" }),
          } } },
      { title: "Read binding vs non-binding", role: "output",
        instr: ["Binding constraints (zero slack) form the optimal corner — in the two-cookie plan that's butter and chips; you use every stick and every cup of chips.", "Non-binding constraints have slack — you finish with spare sugar, eggs and flour, so more of them wouldn't earn a cent (shadow price $0).", "Extreme-case test: set every decision to 0 — revenue must read exactly $0.", "The prized resource is whatever is binding with the highest shadow price — buy more of that first, up to what an extra unit is actually worth."], formulas: [] },
    ],
  },
  {
    id: "carpets", title: "Ship carpets across the network", level: "Network LP · transportation · Solver",
    brief: "Four factories make rolls of carpet (capacities F1 40, F2 50, F3 50, F4 60, in 000's) for five distributors (demands D1 30, D2 24, D3 42, D4 36, D5 48). Unit shipping costs fill a 4×5 grid. Build the transportation model and let Solver find the least-cost shipping schedule.",
    stages: [
      { title: "Enter the cost grid", role: "input",
        instr: ["Blue 4×5 table of unit shipping costs — rows are factories F1–F4, columns are distributors D1–D5.", "Costs ($/roll): F1 = 11,16,18,22,15 · F2 = 12,24,20,21,18 · F3 = 18,17,15,15,20 · F4 = 17,22,14,24,21.", "Put the capacities (40, 50, 50, 60) down the right edge and the demands (30, 24, 42, 36, 48) along the bottom."], formulas: [] },
      { title: "Build the shipment grid", role: "decision",
        instr: ["A matching amber 4×5 block for rolls shipped on each route — start every cell at 0.", "All 20 cells are Solver's changing cells; each one is a decision variable (an arc on the network)."], formulas: [] },
      { title: "Add capacity and demand roll-ups", role: "calc",
        instr: ["Sum each factory's row (rolls shipped out) next to its capacity.", "Sum each distributor's column (rolls received) above its demand."],
        formulas: [{ l: "Shipped from a factory", c: "=SUM(ship_row)" }, { l: "Received by a distributor", c: "=SUM(ship_col)" }] },
      { title: "Build the objective", role: "output",
        instr: ["Total the cost across the whole grid in one boxed cell — this is what Solver minimizes."],
        formulas: [{ l: "Total shipping cost", c: "=SUMPRODUCT(cost_grid, ship_grid)" }] },
      { title: "Sanity-check, then run Solver", role: "calc",
        instr: ["First check feasibility: total capacity 40+50+50+60 = 200 ≥ total demand 30+24+42+36+48 = 180. Good.", "Set Objective = total cost, To: Min. By Changing = the 20 ship cells.", "Constraints: each factory's Shipped ≤ its Capacity; each distributor's Received ≥ its Demand.", "Simplex LP · ✓ Non-Negative · Solve."], formulas: [],
        checkpoint: { prompt: "What minimum total shipping cost does Solver report?", answer: 2660, tol: 1, unit: "$",
          hints: [
            "Use the worksheet — it starts on the optimal plan. Try nudging a shipment onto a cheaper-looking lane and watch either the cost climb or a demand/capacity break.",
            "Lean on the cheap lanes: F4→D3 is $14 (cover all of D3's 42 there), F2→D1 is $12 (cover D1's 30), F1→D5 is $15.",
            "The optimal routes are F1→D2 10, F1→D5 30, F2→D1 30, F2→D5 18, F3→D2 14, F3→D4 36, F4→D3 42.",
          ],
          why: "Routing on the cheapest lanes that still satisfy every demand within each factory's capacity gives total cost 16·10 + 15·30 + 12·30 + 18·18 + 17·14 + 15·36 + 14·42 = $2,660. F1 and F3 ship to capacity (binding); F2 and F4 have slack. Because all capacities and demands are integer, the plan comes out whole with no integer constraints.",
          solution: ["Cover D3 (42) on the cheapest lane F4→D3 ($14) and D1 (30) on F2→D1 ($12).", "Send F1's 40 as D5 30 + D2 10; F3's 50 as D4 36 + D2 14; F2's remaining 18 to D5.", "Every demand is met and no factory exceeds capacity.", "Cost = 160 + 450 + 360 + 324 + 238 + 540 + 588 = $2,660."],
          worksheet: {
            title: "Nudge the shipping plan — watch cost & feasibility",
            inputs: [
              { key: "f1d2", label: "F1→D2 ($16)", def: 10 }, { key: "f1d5", label: "F1→D5 ($15)", def: 30 },
              { key: "f2d1", label: "F2→D1 ($12)", def: 30 }, { key: "f2d5", label: "F2→D5 ($18)", def: 18 },
              { key: "f3d2", label: "F3→D2 ($17)", def: 14 }, { key: "f3d4", label: "F3→D4 ($15)", def: 36 },
              { key: "f4d3", label: "F4→D3 ($14)", def: 42 },
            ],
            rows: (v) => [
              { label: "F1 shipped = F1→D2 + F1→D5", value: v.f1d2 + v.f1d5, limit: 40 },
              { label: "F2 shipped = F2→D1 + F2→D5", value: v.f2d1 + v.f2d5, limit: 50 },
              { label: "F3 shipped = F3→D2 + F3→D4", value: v.f3d2 + v.f3d4, limit: 50 },
              { label: "F4 shipped = F4→D3", value: v.f4d3, limit: 60 },
              { label: "D1 received = F2→D1  (need 30)", value: v.f2d1, need: 30 },
              { label: "D2 received = F1→D2 + F3→D2  (need 24)", value: v.f1d2 + v.f3d2, need: 24 },
              { label: "D3 received = F4→D3  (need 42)", value: v.f4d3, need: 42 },
              { label: "D4 received = F3→D4  (need 36)", value: v.f3d4, need: 36 },
              { label: "D5 received = F1→D5 + F2→D5  (need 48)", value: v.f1d5 + v.f2d5, need: 48 },
            ],
            result: (v) => ({ label: "Total cost = 16·F1D2 + 15·F1D5 + 12·F2D1 + 18·F2D5 + 17·F3D2 + 15·F3D4 + 14·F4D3", value: 16 * v.f1d2 + 15 * v.f1d5 + 12 * v.f2d1 + 18 * v.f2d5 + 17 * v.f3d2 + 15 * v.f3d4 + 14 * v.f4d3, unit: "$" }),
          } } },
      { title: "Read the sensitivity", role: "output",
        instr: ["The shadow price on D3's demand is +$14 — requiring one more roll at D3 adds $14 to the optimal cost.", "The shadow price on F1's capacity is −$3 — one more unit of F1 capacity lets you re-route and cut $3 from the total.", "Only binding constraints (F1 and F3 capacity, all demands) carry a non-zero shadow price; F2 and F4 have slack, so theirs are $0.", "This whole model is a special case of LP with 0/1 coefficients — and the assignment problem is the same setup with every supply and demand equal to 1."], formulas: [] },
    ],
  },
];

/* ============================================================================
   SMALL UI PRIMITIVES
   ============================================================================ */
function RoleChip({ role }) {
  const r = ROLE[role];
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[11px] ds-mono ${r.chip}`}>
      <span className={`w-2 h-2 rounded-full ${r.dot}`} />{r.name}
    </span>
  );
}
function ModelCell({ role, label, value, formula }) {
  const r = ROLE[role];
  return (
    <div className={`rounded-lg border-2 ${r.ring} bg-white px-3 py-2 shadow-sm`}>
      <div className="flex items-center justify-between">
        <span className="text-[11px] uppercase tracking-wide text-slate-500 ds-mono">{label}</span>
        <span className={`w-2.5 h-2.5 rounded-full ${r.dot}`} />
      </div>
      <div className="text-xl ds-mono text-slate-900 leading-tight mt-0.5">{value}</div>
      {formula && <div className="text-[11px] ds-mono text-slate-400 mt-1 truncate" title={formula}>{formula}</div>}
    </div>
  );
}
function Fx({ children }) {
  return <code className="ds-mono text-[13px] bg-slate-900 text-green-300 px-2 py-1 rounded inline-block break-all">{children}</code>;
}
function CopyBtn({ text }) {
  const [ok, setOk] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(text); setOk(true); setTimeout(() => setOk(false), 1200); } catch (e) { /* clipboard blocked; ignore */ }
  };
  return (
    <button onClick={copy} title="Copy formula" className="shrink-0 inline-flex items-center gap-1 text-[11px] ds-mono px-1.5 py-1 rounded border border-slate-600 text-slate-300 hover:bg-slate-800">
      {ok ? <Check className="w-3 h-3 text-green-400" /> : <Copy className="w-3 h-3" />}{ok ? "copied" : "copy"}
    </button>
  );
}
function FormulaLine({ label, code }) {
  return (
    <div className="flex items-center gap-2 bg-slate-900 rounded px-2 py-1.5">
      {label && <span className="text-[11px] ds-mono text-slate-400 shrink-0 w-28 truncate">{label}</span>}
      <code className="ds-mono text-[12.5px] text-green-300 flex-1 break-all">{code}</code>
      <CopyBtn text={code} />
    </div>
  );
}
function SectionTitle({ eyebrow, title, sub }) {
  return (
    <div className="mb-5">
      <div className="text-[11px] ds-mono uppercase tracking-[0.2em] text-indigo-500">{eyebrow}</div>
      <h2 className="ds-display text-2xl sm:text-3xl text-slate-900 font-semibold mt-1">{title}</h2>
      {sub && <p className="text-slate-500 mt-1 max-w-2xl">{sub}</p>}
    </div>
  );
}

/* ============================================================================
   TAB 1 — FRAMEWORK
   ============================================================================ */
function Framework() {
  return (
    <div>
      <SectionTitle eyebrow="Section 1 · Spreadsheet fundamentals" title="Every model is made of four kinds of cells"
        sub="Before you touch Excel, sort every quantity into one of these roles. The whole course — and the color convention you'll grade on — is built on this." />

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-8">
        {["input", "decision", "uncertain", "output"].map((k) => (
          <div key={k} className={`rounded-xl border-2 ${ROLE[k].ring} bg-white p-4`}>
            <RoleChip role={k} />
            <p className="text-sm text-slate-600 mt-2">{ROLE[k].note}</p>
          </div>
        ))}
      </div>

      {/* Influence diagram */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 mb-8">
        <div className="text-[11px] ds-mono uppercase tracking-widest text-slate-400 mb-3">Influence diagram · how the cells connect</div>
        <svg viewBox="0 0 720 220" className="w-full max-w-3xl mx-auto">
          {/* nodes */}
          <g className="ds-mono" fontSize="12">
            <rect x="20" y="30" width="150" height="40" rx="8" fill="#eff6ff" stroke="#93c5fd" strokeWidth="2" />
            <text x="95" y="55" textAnchor="middle" fill="#1d4ed8">Inputs (cost, price)</text>
            <rect x="20" y="150" width="150" height="40" rx="8" fill="#fffbeb" stroke="#fcd34d" strokeWidth="2" />
            <text x="95" y="175" textAnchor="middle" fill="#b45309">Decision (order qty)</text>
            <rect x="285" y="90" width="150" height="40" rx="8" fill="#f1f5f9" stroke="#cbd5e1" strokeWidth="2" />
            <text x="360" y="115" textAnchor="middle" fill="#475569">Calculations</text>
            <rect x="285" y="10" width="150" height="40" rx="8" fill="#f0fdf4" stroke="#86efac" strokeWidth="2" />
            <text x="360" y="35" textAnchor="middle" fill="#15803d">Uncertain (demand)</text>
            <rect x="545" y="90" width="155" height="40" rx="10" fill="#eef2ff" stroke="#6366f1" strokeWidth="3" />
            <text x="622" y="115" textAnchor="middle" fill="#4338ca" fontWeight="bold">Profit (output)</text>
          </g>
          {/* arrows */}
          <g stroke="#94a3b8" strokeWidth="1.6" fill="none" markerEnd="url(#ah)">
            <path d="M170 50 C 230 60, 240 95, 285 105" />
            <path d="M170 170 C 230 160, 240 125, 285 115" />
            <path d="M360 50 L 360 90" />
            <path d="M435 110 L 545 110" />
          </g>
          <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#94a3b8" /></marker></defs>
        </svg>
        <p className="text-sm text-slate-500 mt-2">Inputs and your decision flow through calculations to the output you care about; uncertainty enters along the way. Box the output — it's what the model exists to reveal.</p>
      </div>

      <div className="grid lg:grid-cols-2 gap-5 mb-8">
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h3 className="ds-display text-lg font-semibold text-slate-900 mb-3">The build process</h3>
          <ol className="space-y-2 text-sm text-slate-600">
            {[
              "Frame the decision — what are you actually choosing, and what does 'good' mean?",
              "List the four cell types: parameters, decisions, uncertain variables, outputs.",
              "Sketch an influence diagram so the logic is clear before any formulas.",
              "Build with the color convention — blue inputs, amber decisions, green uncertain, boxed output.",
              "Test it (next tab) before you trust a single number it produces.",
            ].map((s, i) => (
              <li key={i} className="flex gap-3"><span className="ds-mono text-indigo-500 shrink-0">{String(i + 1).padStart(2, "0")}</span><span>{s}</span></li>
            ))}
          </ol>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-5">
          <h3 className="ds-display text-lg font-semibold text-slate-900 mb-3">Why bother modeling?</h3>
          <ul className="space-y-2 text-sm text-slate-600">
            <li className="flex gap-2"><ChevronRight className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />Mistakes on a spreadsheet are cheap; mistakes in the warehouse aren't.</li>
            <li className="flex gap-2"><ChevronRight className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />You can explore options — even ones that seemed impossible — in seconds.</li>
            <li className="flex gap-2"><ChevronRight className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />Good models generate insight, and insight drives better decisions.</li>
            <li className="flex gap-2"><ChevronRight className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />A one-off model can grow into a decision-support tool others reuse.</li>
          </ul>
          <div className="mt-4 rounded-lg bg-amber-50 border border-amber-200 p-3 text-sm text-amber-800">
            <span className="font-semibold">Spreadsheet engineering matters.</span> End-user sheets are riddled with bugs, overconfidence, inefficiency, and unused insight-methods. Discipline is the fix.
          </div>
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   TAB 2 — FORMULAS
   ============================================================================ */
function Formulas() {
  const platform = usePlatform();
  const [cat, setCat] = useState("All");
  const [q, setQ] = useState("");
  const list = useMemo(() => FORMULAS.filter((f) =>
    (cat === "All" || f.cat === cat) &&
    (q === "" || (f.name + f.syntax + f.why + f.how).toLowerCase().includes(q.toLowerCase()))
  ), [cat, q]);

  return (
    <div>
      <SectionTitle eyebrow="Reference" title="Formulas — the how and the why"
        sub="Every function from the DS852 Excel tutorial, with what it does mechanically, why you'd reach for it in a model, a worked example, and the mistake to avoid." />

      <div className="flex flex-col sm:flex-row gap-3 mb-4">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search SUMPRODUCT, VLOOKUP, NPV…"
            className="w-full pl-9 pr-3 py-2 rounded-lg border border-slate-300 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300" />
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5 mb-5">
        {CATS.map((c) => (
          <button key={c} onClick={() => setCat(c)}
            className={`px-3 py-1 rounded-full text-xs ds-mono border transition ${cat === c ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-slate-600 border-slate-300 hover:border-indigo-300"}`}>{c}</button>
        ))}
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        {list.map((f) => (
          <div key={f.name} className="rounded-xl border border-slate-200 bg-white overflow-hidden">
            <div className="flex items-center justify-between px-4 pt-3">
              <h3 className="ds-display font-semibold text-slate-900">{f.name}</h3>
              <span className="text-[10px] ds-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 border border-slate-200">{f.cat}</span>
            </div>
            <div className="px-4 pb-1"><Fx>{f.syntax}</Fx></div>
            <div className="px-4 py-3 space-y-2 text-sm">
              <p className="text-slate-700"><span className="ds-mono text-[11px] text-blue-600 uppercase tracking-wide">How&nbsp;</span>{f.how}</p>
              <p className="text-slate-700"><span className="ds-mono text-[11px] text-indigo-600 uppercase tracking-wide">Why&nbsp;</span>{f.why}</p>
              <div className="rounded-lg bg-slate-50 border border-slate-200 p-2.5">
                <div className="text-[11px] text-slate-500 mb-1">{f.ex.setup}</div>
                <Fx>{f.ex.formula}</Fx>
                <div className="text-[12px] text-green-700 mt-1 ds-mono">→ {f.ex.result}</div>
              </div>
              <p className="text-amber-800 text-[13px] flex gap-1.5"><CircleAlert className="w-4 h-4 shrink-0 mt-0.5 text-amber-500" />{rp(f.watch, platform)}</p>
            </div>
          </div>
        ))}
      </div>
      {list.length === 0 && <div className="text-center text-slate-400 py-12 ds-mono">No functions match that search.</div>}
    </div>
  );
}

/* ============================================================================
   TAB 3 — BUILD & SOLVE
   ============================================================================ */
function Builder() {
  const [objType, setObjType] = useState("max");
  const [varNames, setVarNames] = useState(["Desks", "Tables"]);
  const [c, setC] = useState([60, 40]);
  const [intVars, setIntVars] = useState([]);
  const [cons, setCons] = useState([
    { coeffs: [4, 2], op: "<=", rhs: 40, label: "Cutting hours" },
    { coeffs: [2, 4], op: "<=", rhs: 40, label: "Finishing hours" },
  ]);
  const [result, setResult] = useState(null);
  const n = varNames.length;

  const loadPreset = (key) => {
    const p = PRESETS[key];
    setObjType(p.objType); setVarNames(p.varNames.slice()); setC(p.c.slice());
    setCons(p.constraints.map((x) => ({ ...x, coeffs: x.coeffs.slice() }))); setIntVars(p.intVars.slice()); setResult(null);
  };
  const setVarCount = (delta) => {
    let nn = Math.max(1, Math.min(6, n + delta));
    if (nn === n) return;
    const names = Array.from({ length: nn }, (_, i) => varNames[i] || `x${i + 1}`);
    setVarNames(names);
    setC(Array.from({ length: nn }, (_, i) => c[i] ?? 0));
    setCons(cons.map((r) => ({ ...r, coeffs: Array.from({ length: nn }, (_, i) => r.coeffs[i] ?? 0) })));
    setIntVars(intVars.filter((i) => i < nn)); setResult(null);
  };
  const addCon = () => { setCons([...cons, { coeffs: new Array(n).fill(0), op: "<=", rhs: 0, label: `Constraint ${cons.length + 1}` }]); setResult(null); };
  const rmCon = (i) => { setCons(cons.filter((_, k) => k !== i)); setResult(null); };
  const toggleInt = (i) => { setIntVars(intVars.includes(i) ? intVars.filter((k) => k !== i) : [...intVars, i]); setResult(null); };

  const solve = () => setResult(solveMIP(objType, c.map(Number), cons.map((r) => ({ coeffs: r.coeffs.map(Number), op: r.op, rhs: Number(r.rhs) })), intVars));

  const num = "w-16 px-2 py-1 rounded border border-slate-300 text-sm ds-mono text-center focus:outline-none focus:ring-2 focus:ring-amber-300";

  return (
    <div>
      <SectionTitle eyebrow="Section 2 · Optimization" title="Build a model, then solve it"
        sub="Lay out decisions, an objective, and constraints. The engine runs a two-phase simplex (with branch-and-bound for integer variables) and hands you the exact Excel Solver setup to reproduce it." />

      <div className="flex flex-wrap gap-2 mb-5">
        <span className="text-xs ds-mono text-slate-400 self-center">Load a classic:</span>
        {Object.entries(PRESETS).map(([k, p]) => (
          <button key={k} onClick={() => loadPreset(k)} className="px-3 py-1 rounded-full text-xs ds-mono border border-slate-300 bg-white hover:border-amber-400 text-slate-700">{p.label}</button>
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-5">
        {/* LEFT: builder */}
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="ds-mono text-[11px] uppercase tracking-widest text-slate-400">Objective</span>
              <div className="flex gap-1">
                {["max", "min"].map((t) => (
                  <button key={t} onClick={() => { setObjType(t); setResult(null); }} className={`px-3 py-1 rounded-md text-xs ds-mono border ${objType === t ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-slate-600 border-slate-300"}`}>{t === "max" ? "Maximize" : "Minimize"}</button>
                ))}
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              {varNames.map((v, i) => (
                <span key={i} className="flex items-center gap-1">
                  {i > 0 && <span className="text-slate-400">+</span>}
                  <input value={c[i]} onChange={(e) => { const cc = c.slice(); cc[i] = e.target.value; setC(cc); setResult(null); }} className={num} />
                  <input value={v} onChange={(e) => { const nn = varNames.slice(); nn[i] = e.target.value; setVarNames(nn); }} className="w-20 px-2 py-1 rounded border border-amber-300 bg-amber-50 text-sm ds-mono text-amber-800 focus:outline-none focus:ring-2 focus:ring-amber-300" />
                </span>
              ))}
            </div>
            <div className="flex items-center gap-2 mt-3">
              <span className="text-xs text-slate-400 ds-mono">Decisions:</span>
              <button onClick={() => setVarCount(-1)} className="w-6 h-6 rounded border border-slate-300 text-slate-600">−</button>
              <span className="ds-mono text-sm w-4 text-center">{n}</span>
              <button onClick={() => setVarCount(1)} className="w-6 h-6 rounded border border-slate-300 text-slate-600">+</button>
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="ds-mono text-[11px] uppercase tracking-widest text-slate-400">Constraints</span>
              <button onClick={addCon} className="flex items-center gap-1 text-xs ds-mono text-indigo-600"><Plus className="w-3.5 h-3.5" />Add</button>
            </div>
            <div className="space-y-2">
              {cons.map((r, ci) => (
                <div key={ci} className="flex flex-wrap items-center gap-1.5">
                  {r.coeffs.map((v, vi) => (
                    <span key={vi} className="flex items-center gap-1">
                      {vi > 0 && <span className="text-slate-300 text-xs">+</span>}
                      <input value={v} onChange={(e) => { const cc = cons.slice(); cc[ci].coeffs[vi] = e.target.value; setCons(cc); setResult(null); }} className={num.replace("w-16", "w-14")} />
                      <span className="text-[11px] ds-mono text-amber-600">{varNames[vi]}</span>
                    </span>
                  ))}
                  <select value={r.op} onChange={(e) => { const cc = cons.slice(); cc[ci].op = e.target.value; setCons(cc); setResult(null); }} className="px-1.5 py-1 rounded border border-slate-300 text-sm ds-mono">
                    <option value="<=">≤</option><option value=">=">≥</option><option value="=">=</option>
                  </select>
                  <input value={r.rhs} onChange={(e) => { const cc = cons.slice(); cc[ci].rhs = e.target.value; setCons(cc); setResult(null); }} className={num} />
                  <button onClick={() => rmCon(ci)} className="text-slate-300 hover:text-red-500"><Trash2 className="w-4 h-4" /></button>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <span className="ds-mono text-[11px] uppercase tracking-widest text-slate-400">Integer decisions (Ch. 6)</span>
            <div className="flex flex-wrap gap-2 mt-2">
              {varNames.map((v, i) => (
                <button key={i} onClick={() => toggleInt(i)} className={`px-2.5 py-1 rounded-md text-xs ds-mono border ${intVars.includes(i) ? "bg-green-600 text-white border-green-600" : "bg-white text-slate-500 border-slate-300"}`}>{v} {intVars.includes(i) ? "= int" : ""}</button>
              ))}
            </div>
            <p className="text-[11px] text-slate-400 mt-2">Toggle a decision to force whole-number values — the model becomes a MIP solved by branch-and-bound.</p>
          </div>

          <button onClick={solve} className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white py-2.5 rounded-lg ds-mono text-sm font-semibold shadow">
            <Play className="w-4 h-4" />Solve
          </button>
        </div>

        {/* RIGHT: result + Excel translation */}
        <div className="space-y-4">
          {result ? (
            result.status === "optimal" ? (
              <>
                <div className="rounded-xl border-2 border-indigo-300 bg-indigo-50 p-4">
                  <div className="ds-mono text-[11px] uppercase tracking-widest text-indigo-500 mb-2">Optimal solution</div>
                  <div className="grid grid-cols-2 gap-2">
                    {varNames.map((v, i) => (
                      <ModelCell key={i} role="decision" label={v} value={Number(result.x[i]).toLocaleString(undefined, { maximumFractionDigits: 2 })} />
                    ))}
                  </div>
                  <div className="mt-3">
                    <ModelCell role="output" label={`${objType === "max" ? "Maximized" : "Minimized"} objective`} value={Number(result.objective).toLocaleString(undefined, { maximumFractionDigits: 2 })} />
                  </div>
                </div>

                <div className="rounded-xl border border-slate-200 bg-white p-4">
                  <div className="ds-mono text-[11px] uppercase tracking-widest text-slate-400 mb-3">Reproduce it in Excel Solver</div>
                  <ol className="space-y-2 text-sm text-slate-700">
                    <li className="flex gap-2"><span className="ds-mono text-indigo-500">1</span><div>Enter decisions in a row of cells (e.g. <span className="ds-mono text-amber-700">B2:{String.fromCharCode(65 + n)}2</span>) — these are the <span className="text-amber-700">changing cells</span>.</div></li>
                    <li className="flex gap-2"><span className="ds-mono text-indigo-500">2</span><div>Objective cell: <Fx>=SUMPRODUCT(costs, decisions)</Fx></div></li>
                    <li className="flex gap-2"><span className="ds-mono text-indigo-500">3</span><div>Data ▸ Solver → Set Objective to that cell, To: <span className="ds-mono">{objType === "max" ? "Max" : "Min"}</span>, By Changing: the decision cells.</div></li>
                    <li className="flex gap-2"><span className="ds-mono text-indigo-500">4</span><div>Add each constraint as a row using SUMPRODUCT vs. its limit:</div></li>
                  </ol>
                  <div className="mt-2 space-y-1">
                    {cons.map((r, i) => (
                      <div key={i} className="ds-mono text-[12px] bg-slate-50 border border-slate-200 rounded px-2 py-1 text-slate-600">
                        {r.label || `C${i + 1}`}: {r.coeffs.map((v, vi) => `${v}·${varNames[vi]}`).join(" + ")} {r.op} {r.rhs}
                      </div>
                    ))}
                    {intVars.length > 0 && <div className="ds-mono text-[12px] bg-green-50 border border-green-200 rounded px-2 py-1 text-green-700">{intVars.map((i) => varNames[i]).join(", ")} = integer</div>}
                  </div>
                  <p className="text-[12px] text-slate-500 mt-3">Choose <span className="ds-mono">Simplex LP</span> as the solving method{intVars.length ? " (it handles the integer constraints via branch-and-bound)" : ""}, and check <span className="ds-mono">Make Unconstrained Variables Non-Negative</span>.</p>
                </div>
              </>
            ) : (
              <div className="rounded-xl border-2 border-red-200 bg-red-50 p-5">
                <div className="flex items-center gap-2 text-red-700 font-semibold ds-display"><CircleX className="w-5 h-5" />{result.status === "unbounded" ? "Unbounded" : "Infeasible"}</div>
                <p className="text-sm text-red-700 mt-2">
                  {result.status === "unbounded"
                    ? "The objective can grow without limit — a real constraint is missing. In business terms, you've told the model it can make infinite profit. Add the limit you forgot."
                    : "No solution satisfies every constraint at once — they contradict each other. Loosen a limit or check for a sign error in a ≥ / ≤."}
                </p>
              </div>
            )
          ) : (
            <div className="rounded-xl border border-dashed border-slate-300 bg-white p-8 text-center text-slate-400">
              <Calculator className="w-8 h-8 mx-auto mb-2 text-slate-300" />
              <p className="ds-mono text-sm">Set up your model and press Solve.</p>
              <p className="text-xs mt-1">The solution and its Excel Solver translation appear here.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   TAB 4 — TEST & PRACTICE
   ============================================================================ */
function TestPractice() {
  const [mode, setMode] = useState("validator");
  return (
    <div>
      <SectionTitle eyebrow="Section 3 · Validation" title="Test a model — and test yourself"
        sub="Cholette grades on whether your numbers hold up, not whether the sheet looks busy. Practice the sanity checks she teaches, then quiz the concepts." />
      <div className="flex gap-2 mb-6">
        <button onClick={() => setMode("validator")} className={`px-4 py-2 rounded-lg text-sm ds-mono border ${mode === "validator" ? "bg-slate-900 text-white border-slate-900" : "bg-white text-slate-600 border-slate-300"}`}>Model validator</button>
        <button onClick={() => setMode("quiz")} className={`px-4 py-2 rounded-lg text-sm ds-mono border ${mode === "quiz" ? "bg-slate-900 text-white border-slate-900" : "bg-white text-slate-600 border-slate-300"}`}>Concept quiz</button>
      </div>
      {mode === "validator" ? <Validator /> : <Quiz />}
    </div>
  );
}

// --- Randy profit model used by both Validator and the Randy tab ---
function randyProfit({ fixed, varCost, price, salvage }, order, demand) {
  const sold = Math.min(order, demand);
  const leftover = Math.max(order - demand, 0);
  const revenue = price * sold + salvage * leftover;
  const cost = fixed + varCost * order;
  return { sold, leftover, revenue, cost, profit: revenue - cost };
}

function Validator() {
  const platform = usePlatform();
  const base = { fixed: 750, varCost: 8, price: 18, salvage: 6 };
  const [fixed, setFixed] = useState(750);
  const [order, setOrder] = useState(1450);
  const p = { fixed, varCost: 8, price: 18, salvage: 6 };

  // Run Cholette's sanity checks automatically against the live model
  const checks = useMemo(() => {
    const atZeroOrder = randyProfit(p, 0, 1500);
    const atZeroDemand = randyProfit(p, order, 0);
    const lowFixed = randyProfit({ ...p, fixed: fixed }, order, 1500).profit;
    const highFixed = randyProfit({ ...p, fixed: fixed + 500 }, order, 1500).profit;
    const roughVar = 8 * order; // rough estimate of variable cost
    return [
      { name: "Extreme case: order 0", pass: atZeroOrder.revenue === 0, detail: `Order zero → revenue ${atZeroOrder.revenue}, variable cost 0. Revenue should be exactly 0.` },
      { name: "Extreme case: demand 0", pass: atZeroDemand.sold === 0 && atZeroDemand.leftover === order, detail: `Demand zero → 0 sold at full price, all ${order} become salvage leftovers. Logic holds.` },
      { name: "Monotonicity: ↑ fixed cost", pass: highFixed < lowFixed, detail: `Raising fixed cost by $500 moved profit from ${lowFixed.toFixed(0)} to ${highFixed.toFixed(0)}. Profit must fall — it does.` },
      { name: "Rough estimate: variable cost", pass: Math.abs(randyProfit(p, order, 1500).cost - (fixed + roughVar)) < 1, detail: `Head-math: ${order} shirts × $8 = $${roughVar.toLocaleString()}, plus $${fixed} fixed = $${(fixed + roughVar).toLocaleString()}. Matches the model's total cost.` },
    ];
  }, [fixed, order]);

  return (
    <div className="grid lg:grid-cols-2 gap-5">
      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <h3 className="ds-display text-lg font-semibold text-slate-900 mb-1">Automated sanity checks</h3>
        <p className="text-sm text-slate-500 mb-4">These are the exact tests from the "Testing a Spreadsheet" slides — run live on Randy's profit model. Move the sliders and watch them re-evaluate.</p>
        <div className="space-y-2 mb-4">
          <label className="text-xs ds-mono text-slate-500 flex justify-between">Fixed cost <span className="text-blue-600">${fixed}</span></label>
          <input type="range" min="0" max="3000" step="50" value={fixed} onChange={(e) => setFixed(+e.target.value)} className="w-full accent-blue-600" />
          <label className="text-xs ds-mono text-slate-500 flex justify-between">Order quantity <span className="text-amber-600">{order}</span></label>
          <input type="range" min="0" max="2500" step="50" value={order} onChange={(e) => setOrder(+e.target.value)} className="w-full accent-amber-500" />
        </div>
        <div className="space-y-2">
          {checks.map((c) => (
            <div key={c.name} className={`rounded-lg border p-3 ${c.pass ? "border-green-200 bg-green-50" : "border-red-200 bg-red-50"}`}>
              <div className="flex items-center gap-2 text-sm font-semibold">
                {c.pass ? <CircleCheck className="w-4 h-4 text-green-600" /> : <CircleX className="w-4 h-4 text-red-600" />}
                <span className={c.pass ? "text-green-800" : "text-red-800"}>{c.name}</span>
              </div>
              <p className="text-[13px] text-slate-600 mt-1">{c.detail}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <h3 className="ds-display text-lg font-semibold text-slate-900 mb-3">The auditing toolkit</h3>
        <p className="text-sm text-slate-500 mb-4">When a number looks wrong, these built-in tools find the break. Know them before the lab.</p>
        <ul className="space-y-3 text-sm">
          {[
            ["Trace Precedents", "Formulas ▸ shows every cell that FEEDS a formula. Follow the arrows back to a bad input."],
            ["Trace Dependents", "Shows every cell that USES the selected one — gauge the blast radius before you edit."],
            [P("Show Formulas (Ctrl+~)", "Show Formulas (⌃ `)"), "Flip the whole sheet to formula view to eyeball logic instead of results."],
            [P("Watch Window", "Trace arrows (no Watch Window on Mac)"), P("Pin key output cells so you see them update while you edit elsewhere.", "Excel for Mac has no Watch Window — use Trace Precedents/Dependents and Show Formulas to inspect logic instead.")],
            ["Error checking", P("File ▸ Options ▸ Formulas — flags inconsistent formulas and stray references.", "Excel ▸ Settings… ▸ Error Checking — flags inconsistent formulas and stray references.")],
            ["Give it to an outsider", "If a colleague can't follow it, the model isn't done. Usability is part of correctness."],
          ].map(([t, d], i) => (
            <li key={i} className="flex gap-3">
              <ClipboardCheck className="w-4 h-4 text-indigo-500 shrink-0 mt-0.5" />
              <div><span className="font-semibold text-slate-800 ds-mono text-[13px]">{rp(t, platform)}</span><p className="text-slate-600">{rp(d, platform)}</p></div>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function Quiz() {
  const [cat, setCat] = useState("All");
  const cats = ["All", ...Array.from(new Set(QUIZ.map((q) => q.cat)))];
  const pool = useMemo(() => QUIZ.filter((q) => cat === "All" || q.cat === cat), [cat]);
  const [i, setI] = useState(0);
  const [picked, setPicked] = useState(null);
  const [score, setScore] = useState({ right: 0, total: 0 });
  const item = pool[i % pool.length];

  const choose = (idx) => { if (picked !== null) return; setPicked(idx); setScore((s) => ({ right: s.right + (idx === item.correct ? 1 : 0), total: s.total + 1 })); };
  const next = () => { setPicked(null); setI((v) => (v + 1) % pool.length); };
  const reset = (nc) => { setCat(nc); setI(0); setPicked(null); setScore({ right: 0, total: 0 }); };

  return (
    <div className="max-w-2xl">
      <div className="flex flex-wrap gap-1.5 mb-4">
        {cats.map((c) => (
          <button key={c} onClick={() => reset(c)} className={`px-3 py-1 rounded-full text-xs ds-mono border ${cat === c ? "bg-slate-900 text-white border-slate-900" : "bg-white text-slate-600 border-slate-300"}`}>{c}</button>
        ))}
        <span className="ml-auto text-xs ds-mono text-slate-400 self-center">Score {score.right}/{score.total}</span>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-5">
        <div className="flex items-center justify-between mb-3">
          <span className="text-[10px] ds-mono px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-600 border border-indigo-200">{item.cat}</span>
          <span className="text-xs ds-mono text-slate-400">Q{(i % pool.length) + 1} / {pool.length}</span>
        </div>
        <p className="text-slate-900 font-medium mb-4">{item.q}</p>
        <div className="space-y-2">
          {item.a.map((opt, idx) => {
            const isCorrect = idx === item.correct;
            const show = picked !== null;
            let cls = "border-slate-300 hover:border-indigo-300 bg-white";
            if (show && isCorrect) cls = "border-green-400 bg-green-50";
            else if (show && idx === picked) cls = "border-red-400 bg-red-50";
            return (
              <button key={idx} onClick={() => choose(idx)} disabled={show}
                className={`w-full text-left px-3 py-2 rounded-lg border text-sm transition flex items-center justify-between ${cls}`}>
                <span className="text-slate-700">{opt}</span>
                {show && isCorrect && <CircleCheck className="w-4 h-4 text-green-600" />}
                {show && idx === picked && !isCorrect && <CircleX className="w-4 h-4 text-red-600" />}
              </button>
            );
          })}
        </div>
        {picked !== null && (
          <div className="mt-4 rounded-lg bg-slate-50 border border-slate-200 p-3">
            <p className="text-sm text-slate-700"><span className="ds-mono text-[11px] uppercase text-indigo-600">Why&nbsp;</span>{item.why}</p>
            <button onClick={next} className="mt-3 flex items-center gap-1 text-sm ds-mono text-indigo-600 font-semibold">Next question <ChevronRight className="w-4 h-4" /></button>
          </div>
        )}
      </div>
    </div>
  );
}

/* ============================================================================
   TAB — EXAMPLES  (a worked answer for every assignment, solved live)
   Replaces the single Randy tab. Randy stays as the featured interactive
   example (RandyInteractive); every Builder preset is then solved by the
   in-app solver so the numbers on screen are always exactly right.
   ============================================================================ */

// The interactive newsvendor — Randy's model, minus its own section title so
// it can sit inside the Examples gallery as the featured example.
function RandyInteractive() {
  const inp = { fixed: 750, varCost: 8, price: 18, salvage: 6 };
  const [order, setOrder] = useState(1450);
  const [demand, setDemand] = useState(1500);
  const [sweep, setSweep] = useState("order");
  const r = randyProfit(inp, order, demand);

  const chartData = useMemo(() => {
    const pts = [];
    for (let v = 0; v <= 2500; v += 50) {
      const res = sweep === "order" ? randyProfit(inp, v, demand) : randyProfit(inp, order, v);
      pts.push({ x: v, profit: Math.round(res.profit) });
    }
    return pts;
  }, [order, demand, sweep]);

  return (
    <div>
      <div className="grid lg:grid-cols-5 gap-4 mb-6">
        <div className="lg:col-span-2 space-y-3">
          <div className="rounded-xl border border-blue-200 bg-blue-50/40 p-4">
            <div className="ds-mono text-[11px] uppercase tracking-widest text-blue-500 mb-2">Given inputs</div>
            <div className="grid grid-cols-2 gap-2">
              <ModelCell role="input" label="Fixed cost" value="$750" formula="order fee" />
              <ModelCell role="input" label="Variable cost" value="$8" formula="per shirt" />
              <ModelCell role="input" label="Full price" value="$18" formula="per shirt" />
              <ModelCell role="input" label="Salvage" value="$6" formula="leftover" />
            </div>
          </div>

          <div className="rounded-xl border border-amber-200 bg-amber-50/40 p-4 space-y-3">
            <div className="ds-mono text-[11px] uppercase tracking-widest text-amber-600">Decision & uncertainty — drag to explore</div>
            <div>
              <label className="text-xs ds-mono text-slate-500 flex justify-between mb-1">Order quantity <span className="text-amber-600 font-semibold">{order}</span></label>
              <input type="range" min="0" max="2500" step="50" value={order} onChange={(e) => setOrder(+e.target.value)} className="w-full accent-amber-500" />
            </div>
            <div>
              <label className="text-xs ds-mono text-slate-500 flex justify-between mb-1">Demand (uncertain) <span className="text-green-600 font-semibold">{demand}</span></label>
              <input type="range" min="0" max="2500" step="50" value={demand} onChange={(e) => setDemand(+e.target.value)} className="w-full accent-green-500" />
            </div>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-3">
          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="ds-mono text-[11px] uppercase tracking-widest text-slate-400 mb-3">Calculations (the formulas you'd type)</div>
            <div className="grid grid-cols-2 gap-2">
              <ModelCell role="calc" label="Units sold" value={r.sold.toLocaleString()} formula="=MIN(Order,Demand)" />
              <ModelCell role="calc" label="Leftover" value={r.leftover.toLocaleString()} formula="=MAX(Order-Demand,0)" />
              <ModelCell role="calc" label="Revenue" value={"$" + r.revenue.toLocaleString()} formula="=18*Sold+6*Leftover" />
              <ModelCell role="calc" label="Total cost" value={"$" + r.cost.toLocaleString()} formula="=750+8*Order" />
            </div>
            <div className="mt-3">
              <ModelCell role="output" label="Profit — the number Randy cares about" value={(r.profit < 0 ? "−$" : "$") + Math.abs(r.profit).toLocaleString()} formula="=Revenue-Total cost" />
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-4">
            <div className="flex items-center justify-between mb-2">
              <div className="ds-mono text-[11px] uppercase tracking-widest text-slate-400">One-way data table · profit sensitivity</div>
              <div className="flex gap-1">
                <button onClick={() => setSweep("order")} className={`px-2 py-0.5 rounded text-[11px] ds-mono border ${sweep === "order" ? "bg-amber-500 text-white border-amber-500" : "bg-white text-slate-500 border-slate-300"}`}>vary order</button>
                <button onClick={() => setSweep("demand")} className={`px-2 py-0.5 rounded text-[11px] ds-mono border ${sweep === "demand" ? "bg-green-500 text-white border-green-500" : "bg-white text-slate-500 border-slate-300"}`}>vary demand</button>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="x" tick={{ fontSize: 11, fontFamily: "monospace" }} stroke="#94a3b8" label={{ value: sweep === "order" ? "order qty" : "demand", position: "insideBottom", offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11, fontFamily: "monospace" }} stroke="#94a3b8" width={48} />
                <Tooltip formatter={(v) => ["$" + v.toLocaleString(), "profit"]} labelFormatter={(l) => (sweep === "order" ? "order " : "demand ") + l} />
                <ReferenceLine y={0} stroke="#cbd5e1" />
                <ReferenceLine x={sweep === "order" ? order : demand} stroke="#6366f1" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="profit" stroke="#4f46e5" strokeWidth={2.5} dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-[12px] text-slate-500 mt-1">
              The dashed indigo line marks your current {sweep === "order" ? "order" : "demand"}. Notice the kink: once {sweep === "order" ? "you order past demand, extra shirts only earn salvage" : "demand passes your order, you sell out and profit flattens"} — that bend is the whole point of the newsvendor trade-off.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

// Each entry ties a Builder preset (one of the assignments we've built) to a
// short takeaway and the name/units of its objective. The optimum itself is
// NOT hard-coded — it's computed live by the same tested solver the Builder
// uses, so the number on the card can never drift out of sync with the model.
const EXAMPLES = [
  { key: "productMix", objName: "profit", unit: "$",
    note: "The classic max-profit LP: two products competing for limited cutting and finishing hours. Every assignment after this is a variation on this shape." },
  { key: "bakeSale", objName: "revenue", unit: "$",
    note: "Two cookie types share five limited ingredients. A 2-variable LP you can still check by hand or graph." },
  { key: "bakeSaleFudge", objName: "revenue", unit: "$",
    note: "Add a third recipe and you can no longer graph it — Solver earns its keep, finding a plan the two-cookie mix can't beat." },
  { key: "transportation", objName: "shipping cost", unit: "$",
    note: "The smallest network model: two plants supply two cities at least cost. Each route is a decision (an arc); each supply/demand line is a constraint (a node)." },
  { key: "shipCarpets", objName: "shipping cost", unit: "$",
    note: "The full transportation assignment — 4 factories → 5 distributors. All 20 routes are decisions; because every coefficient is 0/1, integer supplies and demands give a whole-number plan for free." },
  { key: "assignJobs", objName: "total time", unit: "",
    note: "Assignment is transportation with supplies and demands of 1: place jobs on machines so total processing time is least. The 0/1 answer falls out without any integer constraints." },
  { key: "staffing", objName: "payroll cost", unit: "$",
    note: "An integer program — you can't hire a fraction of a person, so both variables are whole numbers and Solver uses branch-and-bound." },
];

/* ============================================================================
   TAB — EXAMPLES  (gallery: every assignment, solved by the in-app solver)
   ============================================================================ */
function ExampleCard({ ex }) {
  const preset = PRESETS[ex.key];
  const res = useMemo(
    () => solveMIP(preset.objType, preset.c, preset.constraints, preset.intVars || []),
    [ex.key],
  );
  const optimal = res.status === "optimal";
  const fmtNum = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
  // Only the routes/quantities the solver actually uses — matches how the
  // worked solutions in the stories read, and keeps 20-arc models legible.
  const active = optimal
    ? preset.varNames.map((name, i) => ({ name, val: res.x[i] })).filter((d) => Math.abs(d.val) > 1e-6)
    : [];
  const objLabel = `Optimal ${ex.objName}`;
  const objValue = optimal ? `${ex.unit}${fmtNum(res.objective)}` : "—";

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 flex flex-col">
      <div className="flex items-center gap-2 mb-2 flex-wrap">
        <span className={`text-[10px] ds-mono px-2 py-0.5 rounded-full border ${preset.objType === "max" ? "bg-amber-50 text-amber-700 border-amber-300" : "bg-blue-50 text-blue-700 border-blue-200"}`}>
          {preset.objType === "max" ? "maximize" : "minimize"}
        </span>
        <span className="text-[10px] ds-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 border border-slate-200">
          {preset.varNames.length} decisions · {preset.constraints.length} constraints
        </span>
      </div>
      <h3 className="ds-display font-semibold text-slate-900">{preset.label}</h3>
      <p className="text-[13px] text-slate-500 mt-1 mb-3 flex-1">{ex.note}</p>

      <div className="flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-indigo-50 border border-indigo-200 mb-3">
        <span className="text-[12px] ds-mono text-indigo-700">{objLabel}</span>
        <span className="ds-mono text-lg font-semibold text-indigo-800 shrink-0">{objValue}</span>
      </div>

      {optimal ? (
        <div>
          <div className="text-[11px] ds-mono uppercase tracking-widest text-slate-400 mb-1.5">
            {preset.objType === "max" ? "Make this many" : "Use these routes"}
          </div>
          <div className="flex flex-wrap gap-1.5">
            {active.map((d, i) => (
              <span key={i} className="text-[12px] ds-mono px-2 py-1 rounded bg-slate-50 border border-slate-200 text-slate-700">
                {d.name} <span className="text-slate-900 font-semibold">{fmtNum(d.val)}</span>
              </span>
            ))}
          </div>
        </div>
      ) : (
        <p className="text-[12px] ds-mono text-red-600">Model status: {res.status}</p>
      )}
    </div>
  );
}

function Examples() {
  return (
    <div>
      <SectionTitle eyebrow="Worked examples" title="Every assignment, solved"
        sub="One card per model we've built across the labs — product mix, the bake sale, the shipping networks, assignment, and staffing. Each optimum is computed live by the same tested solver the Builder uses, so what you see is exactly what Excel's Solver returns. Open Build & Solve to load any of these and change the numbers yourself." />

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {EXAMPLES.map((ex) => (
          <ExampleCard key={ex.key} ex={ex} />
        ))}
      </div>

      <div className="mb-3">
        <div className="ds-mono text-[11px] uppercase tracking-widest text-indigo-500 mb-1">Featured · Lab 1 · interactive</div>
        <h3 className="ds-display text-xl font-semibold text-slate-900">Randy the shirt vendor</h3>
        <p className="text-sm text-slate-500 mt-0.5 max-w-3xl">
          The newsvendor problem the course opens with, built the way you'll build it in Excel — color-coded cells, MIN/MAX logic, and a one-way data table. Drag the sliders to explore the order-vs-demand trade-off.
        </p>
      </div>
      <RandyInteractive />
    </div>
  );
}

/* ============================================================================
   TAB — SHOW ME HOW TO
   ============================================================================ */
function HowTo() {
  const platform = usePlatform();
  const [id, setId] = useState(HOWTOS[0].id);
  const ht = HOWTOS.find((h) => h.id === id);
  return (
    <div>
      <SectionTitle eyebrow="Show me how to…" title="Step-by-step in Excel"
        sub="Pick a task and follow the exact clicks and formulas. Every step says what to do, where to click, and why it matters — so you're building understanding, not just following a recipe." />

      <div className="flex flex-wrap gap-2 mb-6">
        {HOWTOS.map((h) => (
          <button key={h.id} onClick={() => setId(h.id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm ds-mono border transition ${id === h.id ? "bg-slate-900 text-white border-slate-900" : "bg-white text-slate-600 border-slate-300 hover:border-slate-400"}`}>
            <MousePointerClick className="w-3.5 h-3.5" />{h.title}
          </button>
        ))}
      </div>

      <div className="rounded-xl border border-slate-200 bg-white overflow-hidden max-w-3xl">
        <div className="px-5 py-4 border-b border-slate-100 bg-slate-50">
          <h3 className="ds-display text-lg font-semibold text-slate-900">{ht.title}</h3>
          <p className="text-sm text-slate-500 mt-0.5">{ht.goal}</p>
        </div>
        <ol className="divide-y divide-slate-100">
          {ht.steps.map((s, i) => (
            <li key={i} className="px-5 py-4 flex gap-4">
              <div className="shrink-0 w-7 h-7 rounded-full bg-indigo-100 text-indigo-700 ds-mono text-sm flex items-center justify-center font-semibold">{i + 1}</div>
              <div className="flex-1 min-w-0 space-y-1.5">
                <div className="font-medium text-slate-900">{rp(s.do, platform)}</div>
                {s.path && <div className="ds-mono text-[12px] text-slate-500 bg-slate-50 border border-slate-200 rounded px-2 py-1 inline-block">{rp(s.path, platform)}</div>}
                {s.formula && <FormulaLine code={s.formula} />}
                {s.why && <p className="text-[13px] text-slate-600 flex gap-1.5"><Lightbulb className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />{rp(s.why, platform)}</p>}
              </div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

/* ============================================================================
   TAB — PROJECT MODE  (guided build in the student's own Excel)
   ============================================================================ */
function ProjectMode() {
  const [pid, setPid] = useState(null);
  const proj = PROJECTS.find((p) => p.id === pid);
  const [stage, setStage] = useState(0);
  const [done, setDone] = useState({}); // stageIdx -> true when checkpoint passed

  const pick = (id) => { setPid(id); setStage(0); setDone({}); };

  if (!proj) {
    return (
      <div>
        <SectionTitle eyebrow="Project mode" title="Build it yourself"
          sub="Open Excel and build a real model, stage by stage. Each project gives you the layout, the exact formulas, and checkpoints that verify your result against the tested solver. Nothing is done for you — the app coaches, you build." />
        <div className="grid md:grid-cols-3 gap-4">
          {PROJECTS.map((p) => (
            <button key={p.id} onClick={() => pick(p.id)}
              className="text-left rounded-xl border border-slate-200 bg-white p-5 hover:border-indigo-300 hover:shadow-sm transition">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-9 h-9 rounded-lg bg-indigo-50 flex items-center justify-center"><FileSpreadsheet className="w-5 h-5 text-indigo-600" /></div>
                <span className="text-[10px] ds-mono px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 border border-slate-200">{p.level}</span>
              </div>
              <h3 className="ds-display font-semibold text-slate-900">{p.title}</h3>
              <p className="text-[13px] text-slate-500 mt-1 line-clamp-3">{p.brief}</p>
              <div className="mt-3 flex items-center gap-1 text-indigo-600 text-sm ds-mono">Start build <ChevronRight className="w-4 h-4" /></div>
            </button>
          ))}
        </div>
        <div className="mt-5 rounded-lg bg-amber-50 border border-amber-200 p-3 text-sm text-amber-800 max-w-3xl">
          <span className="font-semibold">How to use it:</span> keep Excel open beside this. Work through a stage, type the formulas into your own sheet, then enter your result at the checkpoint to confirm you're on track before moving on.
        </div>
      </div>
    );
  }

  const st = proj.stages[stage];
  const total = proj.stages.length;
  const pct = Math.round(((stage + (done[stage] || !st.checkpoint ? 1 : 0)) / total) * 100);

  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <button onClick={() => setPid(null)} className="text-sm ds-mono text-slate-500 hover:text-slate-800 flex items-center gap-1"><ChevronLeft className="w-4 h-4" />All projects</button>
        <span className="text-xs ds-mono text-slate-400">{proj.level}</span>
      </div>
      <h2 className="ds-display text-2xl font-semibold text-slate-900">{proj.title}</h2>

      {/* progress */}
      <div className="mt-3 mb-5">
        <div className="flex items-center gap-1">
          {proj.stages.map((s, i) => (
            <button key={i} onClick={() => setStage(i)} className="flex-1 group">
              <div className={`h-1.5 rounded-full ${i < stage || done[i] ? "bg-green-500" : i === stage ? "bg-indigo-500" : "bg-slate-200"}`} />
            </button>
          ))}
        </div>
        <div className="flex justify-between mt-1.5 text-[11px] ds-mono text-slate-400">
          <span>Stage {stage + 1} of {total}</span><span>{pct}% complete</span>
        </div>
      </div>

      {stage === 0 && (
        <div className="rounded-lg bg-slate-900 text-slate-100 p-4 mb-5 max-w-3xl">
          <div className="text-[11px] ds-mono uppercase tracking-widest text-slate-400 mb-1">The brief</div>
          <p className="text-sm leading-relaxed">{proj.brief}</p>
        </div>
      )}

      <div className="rounded-xl border border-slate-200 bg-white p-5 max-w-3xl">
        <div className="flex items-center gap-2 mb-1 flex-wrap">
          <span className="text-[11px] ds-mono text-slate-400">Stage {stage + 1}</span>
          <RoleChip role={st.role} />
          <h3 className="ds-display text-lg font-semibold text-slate-900">{st.title}</h3>
        </div>
        <p className="text-[12px] text-slate-400 mb-3">{ROLE[st.role].note}</p>
        <ul className="space-y-2 mb-4">
          {st.instr.map((line, i) => (
            <li key={i} className="flex gap-2 text-sm text-slate-700"><ChevronRight className="w-4 h-4 text-slate-300 shrink-0 mt-0.5" />{line}</li>
          ))}
        </ul>
        {st.formulas && st.formulas.length > 0 && (
          <div className="mb-3">
            <div className="text-[11px] ds-mono uppercase tracking-widest text-slate-400 mb-1.5">Formulas to type — tap copy</div>
            <div className="space-y-1.5">
              {st.formulas.map((f, i) => <FormulaLine key={i} label={f.l} code={f.c} />)}
            </div>
          </div>
        )}
        {st.note && <p className="text-[13px] text-slate-500 flex gap-1.5 mb-3"><Lightbulb className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />{st.note}</p>}
        {st.checkpoint && <Checkpoint key={proj.id + stage} cp={st.checkpoint} onPass={() => {
          setDone((d) => ({ ...d, [stage]: true }));
          // Auto-advance to the next stage after a beat so the learner can read
          // the "why" first. Guard prevents skipping if they've already moved.
          if (stage < total - 1) setTimeout(() => setStage((s) => (s === stage ? s + 1 : s)), 2800);
        }} />}
      </div>

      {/* nav */}
      <div className="flex items-center justify-between mt-4 max-w-3xl">
        <button onClick={() => setStage((s) => Math.max(0, s - 1))} disabled={stage === 0}
          className="flex items-center gap-1 px-3 py-2 rounded-lg text-sm ds-mono border border-slate-300 text-slate-600 disabled:opacity-40"><ChevronLeft className="w-4 h-4" />Back</button>
        {stage < total - 1 ? (
          <button onClick={() => setStage((s) => s + 1)}
            className="flex items-center gap-1 px-4 py-2 rounded-lg text-sm ds-mono bg-indigo-600 text-white font-semibold">Next stage <ChevronRight className="w-4 h-4" /></button>
        ) : (
          <div className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm ds-mono bg-green-50 text-green-700 border border-green-200 font-semibold"><CircleCheck className="w-4 h-4" />Model complete — go test it</div>
        )}
      </div>
    </div>
  );
}

// Interactive mini-worksheet: enter values, watch each line (and the result)
// compute live — so a learner can self-check in the app without Excel open.
// The spec supplies inputs, a rows(v) function, and a result(v) function.
function Worksheet({ spec }) {
  const [v, setV] = useState(() =>
    Object.fromEntries(spec.inputs.map((i) => [i.key, i.def])),
  );
  const set = (k, raw) => setV((p) => ({ ...p, [k]: raw === "" ? 0 : Number(raw) }));
  const rows = spec.rows(v);
  const result = spec.result(v);
  const fmt = (n) => Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3 mt-3">
      <div className="text-[11px] ds-mono uppercase tracking-widest text-slate-400 mb-2">
        {spec.title || "Try it live"}
      </div>
      <div className="flex flex-wrap gap-x-5 gap-y-2 mb-3">
        {spec.inputs.map((i) => (
          <label key={i.key} className="text-xs ds-mono text-slate-600 flex flex-col gap-1">
            <span>{i.label}</span>
            {i.min != null ? (
              <span className="flex items-center gap-2">
                <input type="range" min={i.min} max={i.max} step={i.step || 1} value={v[i.key]}
                  onChange={(e) => set(i.key, e.target.value)} className="accent-indigo-600 w-40" />
                <span className="w-14 text-right text-slate-900">{fmt(v[i.key])}</span>
              </span>
            ) : (
              <input type="number" value={v[i.key]} onChange={(e) => set(i.key, e.target.value)}
                className="w-24 px-2 py-1 rounded border border-slate-300 text-sm ds-mono text-center focus:outline-none focus:ring-2 focus:ring-indigo-300" />
            )}
          </label>
        ))}
      </div>
      <div className="space-y-1 mb-2">
        {rows.map((r, i) => {
          const over = r.limit != null && r.value > r.limit + 1e-9;
          const off = r.need != null && Math.abs(r.value - r.need) > 1e-9;
          const bad = over || off;
          return (
            <div key={i} className="flex items-center justify-between gap-3 text-[12.5px] ds-mono px-2 py-1 rounded bg-slate-50">
              <span className="text-slate-500">{r.label}</span>
              <span className={bad ? "text-red-600 font-semibold shrink-0" : "text-slate-800 shrink-0"}>
                {r.unit === "$" ? "$" : ""}{fmt(r.value)}
                {r.limit != null ? ` / ${r.limit}${over ? " ⚠" : " ✓"}` : ""}
                {r.need != null ? (off ? " ✗" : " ✓") : ""}
              </span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center justify-between gap-3 px-3 py-2 rounded-lg bg-indigo-50 border border-indigo-200">
        <span className="text-[12.5px] ds-mono text-indigo-700">{result.label}</span>
        <span className="ds-mono text-lg font-semibold text-indigo-800 shrink-0">
          {result.unit === "$" ? "$" : ""}{fmt(result.value)}
        </span>
      </div>
    </div>
  );
}

function Checkpoint({ cp, onPass }) {
  const [val, setVal] = useState("");
  const [state, setState] = useState(null); // 'pass' | 'fail'
  const [tries, setTries] = useState(0);
  const [reveal, setReveal] = useState(false);
  const [passed, setPassed] = useState(false);
  const hints = cp.hints ?? (cp.hint ? [cp.hint] : []);
  const fmtAns = `${cp.unit === "$" ? "$" : ""}${cp.answer.toLocaleString()}${cp.unit && cp.unit !== "$" ? " " + cp.unit : ""}`;
  const check = () => {
    const n = parseFloat(String(val).replace(/[^0-9.\-]/g, ""));
    if (isNaN(n)) { setState("fail"); setTries((t) => t + 1); return; }
    if (Math.abs(n - cp.answer) <= cp.tol) {
      setState("pass");
      if (!passed) { setPassed(true); onPass && onPass(); }
    } else {
      setState("fail");
      setTries((t) => t + 1);
    }
  };
  const shown = Math.min(tries, hints.length); // how many progressive hints to reveal

  return (
    <div className="rounded-lg border border-indigo-200 bg-indigo-50/60 p-4 mt-2">
      <div className="text-[11px] ds-mono uppercase tracking-widest text-indigo-500 mb-1.5">Checkpoint</div>
      <p className="text-sm text-slate-800 mb-1">{cp.prompt}</p>

      {cp.worksheet && <Worksheet spec={cp.worksheet} />}

      <div className="flex items-center gap-2 mt-3">
        <input value={val} onChange={(e) => { setVal(e.target.value); setState(null); }} onKeyDown={(e) => e.key === "Enter" && check()}
          placeholder={cp.unit === "$" ? "e.g. 168" : "your result"}
          className="w-40 px-3 py-2 rounded-lg border border-slate-300 ds-mono text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300" />
        <button onClick={check} className="px-3 py-2 rounded-lg bg-indigo-600 text-white text-sm ds-mono font-semibold">Check</button>
      </div>

      {state === "pass" && (
        <div className="mt-2.5">
          <div className="flex items-center gap-2 text-sm text-green-700 font-medium"><CircleCheck className="w-4 h-4" />Correct — {fmtAns} matches the solver.</div>
          {cp.why && <p className="mt-1.5 text-[13px] text-slate-600"><span className="font-semibold text-slate-700">Why:</span> {cp.why}</p>}
        </div>
      )}

      {state === "fail" && (
        <div className="mt-2.5 text-sm text-amber-800">
          <div className="flex items-center gap-2 font-medium"><CircleAlert className="w-4 h-4 text-amber-500" />Not quite — try again.</div>
          {shown > 0 && (
            <ul className="mt-1.5 space-y-1.5">
              {hints.slice(0, shown).map((h, i) => (
                <li key={i} className="flex gap-1.5 text-[13px] text-slate-600"><Lightbulb className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />{h}</li>
              ))}
            </ul>
          )}
          {tries >= hints.length && !reveal && (
            <button onClick={() => setReveal(true)} className="mt-2 text-[12px] ds-mono text-indigo-600 underline">Show the worked solution</button>
          )}
          {reveal && (
            <div className="mt-2 rounded-lg bg-white border border-slate-200 p-3">
              <div className="text-[11px] ds-mono uppercase tracking-widest text-slate-400 mb-1.5">Worked solution</div>
              <ol className="list-decimal ml-4 space-y-1 text-[13px] text-slate-700">
                {(cp.solution ?? []).map((s, i) => (<li key={i}>{s}</li>))}
              </ol>
              <div className="mt-2 ds-mono text-[13px] text-slate-900 font-semibold">Answer: {fmtAns}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/* ============================================================================
   APP SHELL
   ============================================================================ */
const TABS = [
  { id: "framework", label: "Framework", icon: Grid3x3, el: Framework },
  { id: "formulas", label: "Formulas", icon: BookOpen, el: Formulas },
  { id: "howto", label: "Show Me How", icon: ListChecks, el: HowTo },
  { id: "builder", label: "Build & Solve", icon: Wrench, el: Builder },
  { id: "test", label: "Test & Practice", icon: ClipboardCheck, el: TestPractice },
  { id: "examples", label: "Examples", icon: ShoppingBag, el: Examples },
  { id: "project", label: "Project Mode", icon: FileSpreadsheet, el: ProjectMode },
];

export default function App() {
  const [tab, setTab] = useState("framework");
  const [platform, setPlatform] = useState("mac");
  const Active = TABS.find((t) => t.id === tab).el;
  return (
    <PlatformCtx.Provider value={platform}>
    <div className="ds-body min-h-screen text-slate-800 ds-grid">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');
        .ds-display{font-family:'Space Grotesk',system-ui,sans-serif;}
        .ds-mono{font-family:'IBM Plex Mono',ui-monospace,monospace;}
        .ds-body{font-family:'Inter',system-ui,sans-serif;background:#f8fafc;}
        .ds-grid{background-image:linear-gradient(#eef2f7 1px,transparent 1px),linear-gradient(90deg,#eef2f7 1px,transparent 1px);background-size:26px 26px;background-position:-1px -1px;}
        input[type=range]{height:6px;border-radius:9999px;}
      `}</style>

      {/* Formula-bar header — the signature */}
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/85 backdrop-blur">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex items-center gap-3 py-3">
            <div className="w-9 h-9 rounded-lg bg-slate-900 flex items-center justify-center shrink-0"><Boxes className="w-5 h-5 text-green-300" /></div>
            <div className="leading-tight">
              <div className="ds-display font-bold text-slate-900">DS852 Model Lab</div>
              <div className="text-[11px] ds-mono text-slate-400">Managerial Decision Making · spreadsheet modeling companion</div>
            </div>
            {/* platform toggle — flips OS-specific steps (Solver setup, shortcuts, menus) app-wide */}
            <div className="ml-auto flex items-center gap-1.5">
              <span className="text-[10px] ds-mono uppercase tracking-widest text-slate-400 hidden sm:inline">Excel for</span>
              <div className="flex rounded-lg border border-slate-300 overflow-hidden">
                {["windows", "mac"].map((pf) => (
                  <button key={pf} onClick={() => setPlatform(pf)}
                    className={`px-3 py-1 text-xs ds-mono capitalize transition ${platform === pf ? "bg-slate-900 text-white" : "bg-white text-slate-500 hover:text-slate-800"}`}>{pf}</button>
                ))}
              </div>
            </div>
          </div>
          {/* fake formula bar */}
          <div className="flex items-center gap-2 pb-2">
            <span className="ds-mono text-slate-400 text-sm italic px-2 border border-slate-200 rounded bg-slate-50">fx</span>
            <div className="flex-1 ds-mono text-[12px] text-slate-500 border border-slate-200 rounded bg-white px-3 py-1 truncate">
              =STUDY(<span className="text-blue-600">inputs</span>, <span className="text-amber-600">decisions</span>, <span className="text-green-600">uncertainty</span>) → <span className="text-indigo-600">better decisions</span>
            </div>
          </div>
          {/* tabs */}
          <nav className="flex gap-1 overflow-x-auto -mb-px">
            {TABS.map((t) => {
              const I = t.icon;
              return (
                <button key={t.id} onClick={() => setTab(t.id)}
                  className={`flex items-center gap-1.5 px-3 py-2 text-sm ds-mono whitespace-nowrap border-b-2 transition ${tab === t.id ? "border-indigo-600 text-indigo-700" : "border-transparent text-slate-500 hover:text-slate-800"}`}>
                  <I className="w-4 h-4" />{t.label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-7">
        <Active />
      </main>

      <footer className="max-w-6xl mx-auto px-4 py-8 text-center">
        <p className="text-[12px] ds-mono text-slate-400">
          A study companion — clarify concepts, check formula syntax, and practice building & testing models.
          Do your own labs and submissions per the DS852 AI policy.
        </p>
      </footer>
    </div>
    </PlatformCtx.Provider>
  );
}
