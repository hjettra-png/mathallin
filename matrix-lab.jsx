import { useState, useEffect } from "react";

/* =========================================================
   PURE MATH HELPERS & UTILITIES
   ========================================================= */
function makeMatrix(rows, cols, prev) {
  const m = [];
  for (let i = 0; i < rows; i++) {
    const row = [];
    for (let j = 0; j < cols; j++) {
      row.push(prev && prev[i] && prev[i][j] !== undefined ? prev[i][j] : 0);
    }
    m.push(row);
  }
  return m;
}

function toFraction(val, tolerance = 1e-4) {
  if (Math.abs(val) < 1e-9) return "0";
  if (Number.isInteger(val)) return val.toString();
  
  let h1 = 1, h2 = 0, k1 = 0, k2 = 1;
  let b = val;
  do {
    let a = Math.floor(b);
    let aux = h1; h1 = a * h1 + h2; h2 = aux;
    aux = k1; k1 = a * k1 + k2; k2 = aux;
    b = 1 / (b - a);
  } while (Math.abs(val - h1 / k1) > val * tolerance && k1 < 100);

  if (k1 === 1) return `${h1}`;
  return `${h1}/${k1}`;
}

function fmt(n, showFraction = false) {
  if (Math.abs(n) < 1e-9) n = 0;
  if (showFraction) return toFraction(n);
  const r = Math.round(n * 1000) / 1000;
  return r.toString();
}

function transpose(A) {
  const rows = A.length, cols = A[0].length;
  const T = makeMatrix(cols, rows);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) T[j][i] = A[i][j];
  return T;
}

function mul(A, B) {
  const n = A.length, m = A[0].length, p = B[0].length;
  const R = makeMatrix(n, p);
  for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) {
    let s = 0;
    for (let k = 0; k < m; k++) s += A[i][k] * B[k][j];
    R[i][j] = s;
  }
  return R;
}

function exportToLaTeX(M) {
  const body = M.map((row) => row.map((v) => fmt(v)).join(" & ")).join(" \\\\\n");
  return `\\begin{bmatrix}\n${body}\n\\end{bmatrix}`;
}

function getDeterminant(A) {
  const n = A.length;
  if (n !== A[0].length) return null;
  let M = A.map((r) => [...r]);
  let det = 1;
  for (let i = 0; i < n; i++) {
    let k = i;
    for (let j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[k][i])) k = j;
    if (Math.abs(M[k][i]) < 1e-9) return 0;
    if (i !== k) {
      [M[i], M[k]] = [M[k], M[i]];
      det = -det;
    }
    det *= M[i][i];
    for (let j = i + 1; j < n; j++) {
      let c = M[j][i] / M[i][i];
      for (let p = i + 1; p < n; p++) M[j][p] -= c * M[i][p];
    }
  }
  return det;
}

/* ---- Part 1: Gaussian Elimination ---- */
function gaussianElimination(A, b, showFraction) {
  let M = A.map((row, i) => [...row, b[i][0]]);
  const rows = M.length, cols = A[0].length;
  const steps = [];
  let pivotRow = 0;

  for (let col = 0; col < cols && pivotRow < rows; col++) {
    let maxRow = pivotRow;
    for (let r = pivotRow + 1; r < rows; r++) if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
    if (Math.abs(M[maxRow][col]) < 1e-9) continue;
    if (maxRow !== pivotRow) {
      [M[pivotRow], M[maxRow]] = [M[maxRow], M[pivotRow]];
      steps.push({ desc: `Swap R${pivotRow + 1} and R${maxRow + 1}`, matrix: M.map((r) => r.slice()), activeRows: [pivotRow, maxRow] });
    }
    for (let r = pivotRow + 1; r < rows; r++) {
      const factor = M[r][col] / M[pivotRow][col];
      if (Math.abs(factor) < 1e-9) continue;
      for (let c = col; c <= cols; c++) M[r][c] -= factor * M[pivotRow][c];
      steps.push({ desc: `R${r + 1} \u2190 R${r + 1} \u2212 (${fmt(factor, showFraction)})\u00b7R${pivotRow + 1}`, matrix: M.map((r) => r.slice()), activeRows: [r] });
    }
    pivotRow++;
  }

  if (rows === cols && pivotRow === rows) {
    const x = Array(rows).fill(0);
    for (let r = rows - 1; r >= 0; r--) {
      let s = M[r][cols];
      for (let c = r + 1; c < cols; c++) s -= M[r][c] * x[c];
      x[r] = s / M[r][r];
    }
    steps.push({ desc: `<b>Final Unique Solution:</b><br/>` + x.map((v, i) => `x<sub>${i + 1}</sub> = <b>${fmt(v, showFraction)}</b>`).join(", "), matrix: null, activeRows: [] });
  }

  return { resultMatrix: M, steps };
}

/* ---- Part 1: Gauss-Jordan Inverse Solver ---- */
function gaussJordanInverse(A, b, showFraction) {
  const n = A.length;
  if (n !== A[0].length) return { error: "Matrix A must be square for Matrix Inverse method." };

  let M = A.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  const steps = [];

  for (let col = 0; col < n; col++) {
    let maxRow = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
    if (Math.abs(M[maxRow][col]) < 1e-9) return { error: "Matrix A is singular and cannot be inverted." };
    if (maxRow !== col) {
      [M[col], M[maxRow]] = [M[maxRow], M[col]];
      steps.push({ desc: `Swap R${col + 1} and R${maxRow + 1}`, matrix: M.map((r) => r.slice()), activeRows: [col, maxRow] });
    }
    const pivotVal = M[col][col];
    if (Math.abs(pivotVal - 1) > 1e-9) {
      M[col] = M[col].map((v) => v / pivotVal);
      steps.push({ desc: `R${col + 1} \u2190 R${col + 1} / ${fmt(pivotVal, showFraction)}`, matrix: M.map((r) => r.slice()), activeRows: [col] });
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      if (Math.abs(factor) < 1e-9) continue;
      M[r] = M[r].map((v, c) => v - factor * M[col][c]);
      steps.push({ desc: `R${r + 1} \u2190 R${r + 1} \u2212 (${fmt(factor, showFraction)})\u00b7R${col + 1}`, matrix: M.map((r) => r.slice()), activeRows: [r] });
    }
  }

  const invA = M.map((row) => row.slice(n));
  const xMat = mul(invA, b);
  const x = xMat.map((r) => r[0]);

  steps.push({ desc: `Computed Inverse A<sup>-1</sup>`, matrix: invA, activeRows: [] });
  steps.push({ desc: `<b>Calculated x = A<sup>-1</sup>b:</b><br/>` + x.map((v, i) => `x<sub>${i + 1}</sub> = <b>${fmt(v, showFraction)}</b>`).join(", "), matrix: null, activeRows: [] });

  return { invA, solution: x, steps };
}

/* ---- Part 2: Parametric Solution ---- */
function solveParametric(A, b, showFraction) {
  let M = A.map((row, i) => [...row, b[i][0]]);
  const rows = M.length, cols = A[0].length;
  const steps = [];

  let pivotRow = 0;
  const pivotCols = [];

  for (let col = 0; col < cols && pivotRow < rows; col++) {
    let maxRow = pivotRow;
    for (let r = pivotRow + 1; r < rows; r++) if (Math.abs(M[r][col]) > Math.abs(M[maxRow][col])) maxRow = r;
    if (Math.abs(M[maxRow][col]) < 1e-9) continue;
    if (maxRow !== pivotRow) {
      [M[pivotRow], M[maxRow]] = [M[maxRow], M[pivotRow]];
    }
    const pivotVal = M[pivotRow][col];
    M[pivotRow] = M[pivotRow].map((v) => v / pivotVal);

    for (let r = 0; r < rows; r++) {
      if (r === pivotRow) continue;
      const factor = M[r][col];
      if (Math.abs(factor) < 1e-9) continue;
      M[r] = M[r].map((v, c) => v - factor * M[pivotRow][c]);
    }
    pivotCols.push(col);
    steps.push({ desc: `Reduced column ${col + 1}`, matrix: M.map((r) => r.slice()), activeRows: [pivotRow] });
    pivotRow++;
  }

  const freeCols = [];
  for (let c = 0; c < cols; c++) if (!pivotCols.includes(c)) freeCols.push(c);

  const expressions = [];
  for (let i = 0; i < cols; i++) {
    if (freeCols.includes(i)) {
      const paramIdx = freeCols.indexOf(i) + 1;
      expressions.push(`x<sub>${i + 1}</sub> = t<sub>${paramIdx}</sub> (free)`);
    } else {
      const rIdx = pivotCols.indexOf(i);
      let expr = `${fmt(M[rIdx][cols], showFraction)}`;
      freeCols.forEach((fc) => {
        const coeff = -M[rIdx][fc];
        const paramIdx = freeCols.indexOf(fc) + 1;
        if (Math.abs(coeff) > 1e-9) {
          const sign = coeff > 0 ? " + " : " - ";
          expr += `${sign}${fmt(Math.abs(coeff), showFraction)}t<sub>${paramIdx}</sub>`;
        }
      });
      expressions.push(`x<sub>${i + 1}</sub> = ${expr}`);
    }
  }

  steps.push({ desc: `<b>Parametric Solution Form:</b><br/>` + expressions.join("<br/>"), matrix: null, activeRows: [] });
  return { steps, expressions };
}

/* ---- Part 3: Least Squares Solver ---- */
function solveLeastSquares(A, b, showFraction) {
  const steps = [];
  const AT = transpose(A);
  const ATA = mul(AT, A);
  const ATb = mul(AT, b);

  steps.push({ desc: "Calculated A<sup>T</sup>A", matrix: ATA, activeRows: [] });
  steps.push({ desc: "Calculated A<sup>T</sup>b", matrix: ATb, activeRows: [] });

  const invRes = gaussJordanInverse(ATA, ATb, showFraction);
  if (invRes.error) return { error: invRes.error };

  const xHat = invRes.solution;
  const xHatMat = xHat.map((v) => [v]);
  const AxHat = mul(A, xHatMat);

  const e = b.map((row, i) => row[0] - AxHat[i][0]);
  const squareSum = e.reduce((sum, val) => sum + val * val, 0);

  steps.push({
    desc: `<b>Least Squares Solution Vector x&#770;:</b><br/>` +
      xHat.map((v, i) => `x<sub>${i + 1}</sub> = <b>${fmt(v, showFraction)}</b>`).join("<br/>"),
    matrix: null,
    activeRows: []
  });

  steps.push({
    desc: `<b>Sum of Squared Residuals (SSE):</b><br/>` +
      `Residual Vector e = b - Ax&#770; = [ ${e.map((v) => fmt(v, showFraction)).join(", ")} ]<sup>T</sup><br/>` +
      `SSE = &sum; r<sub>i</sub>&sup2; = r<sup>T</sup>r = ${e.map((v) => `(${fmt(v, showFraction)})&sup2;`).join(" + ")} = <b>${fmt(squareSum, showFraction)}</b>`,
    matrix: null,
    activeRows: []
  });

  return { steps, xHat, squareSum };
}

/* =========================================================
   UI COMPONENTS
   ========================================================= */
const AnimStyles = () => (
  <style>{`
    @keyframes mlFadeUp {
      from { opacity: 0; transform: translateY(10px); }
      to   { opacity: 1; transform: translateY(0); }
    }
    .ml-fade-up { animation: mlFadeUp 0.4s ease-out both; }
  `}</style>
);

function MatrixGrid({ M, onChange }) {
  const cols = M[0] ? M[0].length : 1;
  return (
    <div className="flex items-stretch gap-1.5 w-fit">
      <div className="matrix-bracket-left w-2 shrink-0 my-1" />
      <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0,1fr))` }}>
        {M.map((row, i) =>
          row.map((v, j) => {
            let heatClass = "border-slate-300 bg-white/90 text-slate-800 focus:ring-sky-500 focus:border-sky-500 shadow-xs";
            if (v > 0) heatClass = "border-sky-300 bg-sky-50/90 text-sky-900 font-semibold focus:ring-sky-500 shadow-xs";
            else if (v < 0) heatClass = "border-amber-300 bg-amber-50/90 text-amber-900 font-semibold focus:ring-amber-500 shadow-xs";

            return (
              <input
                key={`${i}-${j}`}
                type="number"
                value={v}
                onChange={(e) => onChange(i, j, e.target.value)}
                className={`w-14 h-12 text-base font-mono font-bold text-center rounded-xl border focus:outline-none focus:ring-2 transition-all ${heatClass}`}
              />
            );
          })
        )}
      </div>
      <div className="matrix-bracket-right w-2 shrink-0 my-1" />
    </div>
  );
}

function VisualMatrixGrid({ matrix, showFraction }) {
  if (!matrix) return null;
  const cols = matrix[0] ? matrix[0].length : 1;

  return (
    <div className="flex items-stretch gap-1.5 w-fit">
      <div className="matrix-bracket-left w-2 shrink-0 my-1" />
      <div className="grid gap-2 my-auto" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
        {matrix.map((row, i) =>
          row.map((v, j) => {
            const formatted = fmt(v, showFraction);
            return (
              <div
                key={`${i}-${j}`}
                className="w-16 h-12 flex items-center justify-center font-mono text-base font-bold bg-sky-50/80 text-sky-950 border border-sky-200 rounded-xl shadow-xs"
              >
                {formatted}
              </div>
            );
          })
        )}
      </div>
      <div className="matrix-bracket-right w-2 shrink-0 my-1" />
    </div>
  );
}

function StepPlayback({ steps, showFraction }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  useEffect(() => {
    setCurrentStep(0);
    setIsPlaying(false);
  }, [steps]);

  useEffect(() => {
    let timer;
    if (isPlaying && currentStep < steps.length - 1) {
      timer = setTimeout(() => setCurrentStep((prev) => prev + 1), 1200);
    } else if (currentStep >= steps.length - 1) {
      setIsPlaying(false);
    }
    return () => clearTimeout(timer);
  }, [isPlaying, currentStep, steps.length]);

  const step = steps[currentStep] || steps[0];

  return (
    <div className="space-y-6">
      {/* Playback Controls */}
      <div className="flex items-center justify-between bg-white/80 p-4 rounded-2xl border border-slate-200/80 shadow-sm backdrop-blur-md">
        <div className="flex items-center gap-3">
          <button
            onClick={() => { setIsPlaying(false); setCurrentStep((p) => Math.max(0, p - 1)); }}
            disabled={currentStep === 0}
            className="px-4 py-2 font-mono text-sm font-semibold bg-slate-100 hover:bg-slate-200 disabled:opacity-40 rounded-xl text-slate-700 border border-slate-300 transition-all shadow-xs cursor-pointer"
          >
            ◀ Prev
          </button>
          <button
            onClick={() => setIsPlaying(!isPlaying)}
            className="px-5 py-2 font-mono text-sm font-semibold bg-sky-50 hover:bg-sky-100 text-sky-700 border border-sky-300 rounded-xl transition-all shadow-xs cursor-pointer"
          >
            {isPlaying ? "Pause ❚❚" : "Auto Play ▶"}
          </button>
          <button
            onClick={() => { setIsPlaying(false); setCurrentStep((p) => Math.min(steps.length - 1, p + 1)); }}
            disabled={currentStep === steps.length - 1}
            className="px-4 py-2 font-mono text-sm font-semibold bg-slate-100 hover:bg-slate-200 disabled:opacity-40 rounded-xl text-slate-700 border border-slate-300 transition-all shadow-xs cursor-pointer"
          >
            Next ▶
          </button>
        </div>
        <div className="font-mono text-sm text-slate-600 font-semibold">
          Step <span className="text-sky-600 font-bold text-base">{currentStep + 1}</span> of {steps.length}
        </div>
      </div>

      {/* Visual Step Detail Card matching Matrix Input UI */}
      <div className="ml-fade-up bg-white p-8 rounded-2xl border border-slate-200 shadow-lg space-y-6">
        <div 
          className="font-mono text-base md:text-lg text-slate-800 font-medium leading-relaxed" 
          dangerouslySetInnerHTML={{ __html: step.desc }} 
        />
        
        {step.matrix && (
          <div className="space-y-4">
            <div className="w-full overflow-x-auto bg-slate-50/50 border border-slate-200 rounded-2xl p-6 flex justify-center items-center">
              <VisualMatrixGrid matrix={step.matrix} showFraction={showFraction} />
            </div>

            {step.activeRows && step.activeRows.length > 0 && (
              <div>
                <span className="inline-block font-mono text-sm font-bold text-amber-900 bg-amber-100 border-2 border-amber-300 px-4 py-1.5 rounded-xl shadow-xs">
                  Active Row(s): {step.activeRows.map((r) => `R${r + 1}`).join(", ")}
                </span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* =========================================================
   MAIN APP COMPONENT
   ========================================================= */
export default function Project1Lab() {
  const [A, setA] = useState([[1, 2, 1], [2, 1, 1], [1, 3, 2]]);
  const [b, setB] = useState([[10], [11], [14]]);
  const [result, setResult] = useState(null);
  const [showFraction, setShowFraction] = useState(false);
  const [copied, setCopied] = useState(false);

  function resize(rows, cols) {
    setA(makeMatrix(rows, cols, A));
    setB(makeMatrix(rows, 1, b));
  }

  function updateA(i, j, val) {
    const next = A.map((r) => r.slice());
    next[i][j] = val === "" ? 0 : parseFloat(val) || 0;
    setA(next);
  }

  function updateB(i, j, val) {
    const next = b.map((r) => r.slice());
    next[i][0] = val === "" ? 0 : parseFloat(val) || 0;
    setB(next);
  }

  function loadPreset(part) {
    if (part === 1) {
      setA([[1, 2, 1], [2, 1, 1], [1, 3, 2]]);
      setB([[10], [11], [14]]);
    } else if (part === 2) {
      setA([[1, 2, 1, 1, 2], [2, 1, 3, 2, 1], [1, 1, 2, 3, 1]]);
      setB([[10], [11], [14]]);
    } else if (part === 3) {
      setA([[1, 2, 1], [2, 1, 1], [1, 3, 2], [2, 2, 1]]);
      setB([[10], [11], [14], [12]]);
    }
    setResult(null);
  }

  function runPart1Gaussian() {
    const res = gaussianElimination(A, b, showFraction);
    setResult({ label: "Part 1: Gaussian Elimination", steps: res.steps });
  }

  function runPart1Inverse() {
    const res = gaussJordanInverse(A, b, showFraction);
    if (res.error) setResult({ label: "Error", steps: [{ desc: res.error }] });
    else setResult({ label: "Part 1: Matrix Inverse (Ax = b)", steps: res.steps });
  }

  function runPart2Parametric() {
    const res = solveParametric(A, b, showFraction);
    setResult({ label: "Part 2: Parametric Solution (Underdetermined)", steps: res.steps });
  }

  function runPart3LeastSquares() {
    const res = solveLeastSquares(A, b, showFraction);
    if (res.error) setResult({ label: "Error", steps: [{ desc: res.error }] });
    else setResult({ label: "Part 3: Least Squares Method & Error Analysis", steps: res.steps });
  }

  function copyLaTeX() {
    const code = exportToLaTeX(A);
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  const det = getDeterminant(A);
  const isSquare = A.length === A[0].length;

  const btnStyle = "font-mono text-xs font-semibold bg-white/90 hover:bg-sky-50 text-sky-800 border border-sky-300 hover:border-sky-400 px-3.5 py-2 rounded-lg transition-all backdrop-blur-md shadow-xs cursor-pointer active:scale-[0.98]";
  const presetBtnStyle = "font-mono text-xs font-semibold bg-amber-50/90 hover:bg-amber-100/90 text-amber-800 border border-amber-300 px-3 py-1.5 rounded-lg backdrop-blur-md transition-all shadow-2xs cursor-pointer active:scale-[0.98]";

  return (
    <div className="min-h-screen bg-transparent text-slate-800 py-10 px-6 font-sans">
      <AnimStyles />
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header & Controls Card */}
        <div className="glass-card p-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div>
            <h1 className="text-3xl font-bold text-slate-900 mb-1 tracking-tight">
              Project 1: Linear Regression & Least Squares
            </h1>
            <p className="text-slate-600 text-sm font-medium">
              Linear Equations, Parametric Solutions & Least Square Analysis.
            </p>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <button
              onClick={() => setShowFraction(!showFraction)}
              className="font-mono text-xs font-semibold bg-purple-50/90 hover:bg-purple-100 text-purple-800 border border-purple-300 px-3 py-1.5 rounded-lg backdrop-blur-md transition-all shadow-2xs cursor-pointer"
            >
              Mode: {showFraction ? "Fraction (½)" : "Decimal (0.5)"}
            </button>
            <button
              onClick={copyLaTeX}
              className="font-mono text-xs font-semibold bg-sky-50/90 hover:bg-sky-100 text-sky-800 border border-sky-300 px-3 py-1.5 rounded-lg backdrop-blur-md transition-all shadow-2xs cursor-pointer"
            >
              {copied ? "Copied LaTeX! ✓" : "Copy LaTeX"}
            </button>
          </div>
        </div>

        {/* Quick Load Presets Card */}
        <div className="glass-card p-6 space-y-3">
          <div className="flex flex-wrap justify-between items-center gap-2">
            <div className="font-mono text-xs text-slate-500 font-bold uppercase tracking-wider">Quick Load Project Presets</div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-[11px] font-semibold px-2.5 py-0.5 rounded-md bg-slate-100 border border-slate-300 text-slate-700">
                {A.length}×{A[0].length} Matrix
              </span>
              {isSquare && (
                <span className={`font-mono text-[11px] font-semibold px-2.5 py-0.5 rounded-md border ${Math.abs(det) < 1e-9 ? "bg-red-50 border-red-300 text-red-800" : "bg-emerald-50 border-emerald-300 text-emerald-800"}`}>
                  {Math.abs(det) < 1e-9 ? "Singular (det=0)" : `det(A) = ${fmt(det)}`}
                </span>
              )}
            </div>
          </div>
          <div className="flex flex-wrap gap-2.5">
            <button className={presetBtnStyle} onClick={() => loadPreset(1)}>Part 1 (3×3 Standard)</button>
            <button className={presetBtnStyle} onClick={() => loadPreset(2)}>Part 2 (3×5 Underdetermined)</button>
            <button className={presetBtnStyle} onClick={() => loadPreset(3)}>Part 3 (4×3 Overdetermined)</button>
          </div>
        </div>

        {/* Matrix Inputs Card */}
        <div className="glass-card p-6 flex flex-wrap items-center justify-between gap-6">
          <div className="space-y-3">
            <span className="block font-mono text-sm font-bold text-sky-800">Matrix A (Ingredients)</span>
            <MatrixGrid M={A} onChange={updateA} />
          </div>

          <div className="font-mono text-xs text-slate-600 font-semibold flex items-center gap-1.5 bg-white/80 p-2 rounded-xl border border-slate-200">
            <input type="number" min={1} max={6} value={A.length} onChange={(e) => resize(+e.target.value || 1, A[0].length)} className="w-10 h-8 bg-white border border-slate-300 text-center rounded-lg font-bold text-slate-800" />
            <span className="text-slate-400">×</span>
            <input type="number" min={1} max={6} value={A[0].length} onChange={(e) => resize(A.length, +e.target.value || 1)} className="w-10 h-8 bg-white border border-slate-300 text-center rounded-lg font-bold text-slate-800" />
          </div>

          <div className="space-y-3">
            <span className="block font-mono text-sm font-bold text-amber-800">Vector b (Nutritional Targets)</span>
            <MatrixGrid M={b} onChange={updateB} />
          </div>
        </div>

        {/* Action Solvers Card */}
        <div className="glass-card p-6 space-y-3">
          <div className="font-mono text-xs text-slate-500 font-bold uppercase tracking-wider">Execute Project Tasks</div>
          <div className="flex flex-wrap gap-3">
            <button className={btnStyle} onClick={runPart1Gaussian}>Gaussian Elimination</button>
            <button className={btnStyle} onClick={runPart1Inverse}>Matrix Inverse (A⁻¹b)</button>
            <button className={btnStyle} onClick={runPart2Parametric}>Parametric Solution</button>
            <button className={btnStyle} onClick={runPart3LeastSquares}>Least Square Solver</button>
          </div>
        </div>

        {/* Output Section with Visual Grid Stepper */}
        {result && (
          <div className="glass-card p-6 ml-fade-up space-y-4">
            <h2 className="text-xl font-serif italic font-bold text-amber-900">{result.label}</h2>
            <StepPlayback steps={result.steps} showFraction={showFraction} />
          </div>
        )}

      </div>
    </div>
  );
}