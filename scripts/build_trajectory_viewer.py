#!/usr/bin/env python3
"""Build a self-contained, zero-dependency interactive HTML trajectory replay viewer.

Embeds ``results/sample_trajectories.json`` directly into ``demos/trajectory_viewer.html`` so it
can be opened directly via ``file://`` or hosted on GitHub Pages without a backend server.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJ_PATH = REPO_ROOT / "results" / "sample_trajectories.json"
OUT_HTML = REPO_ROOT / "demos" / "trajectory_viewer.html"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>MiniBooking-RLVR — Interactive MPC Trajectory Replay Viewer</title>
<style>
  :root {
    --bg: #f8fafc;
    --surface: #ffffff;
    --surface-alt: #f1f5f9;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #475569;
    --accent: #0d9488;
    --blue: #2563eb;
    --amber: #d97706;
    --red: #dc2626;
    --green-bg: #ecfdf5;
    --green-border: #a7f3d0;
    --amber-bg: #fffbeb;
    --amber-border: #fde68a;
    --red-bg: #fef2f2;
    --red-border: #fecaca;
    --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
  }
  header {
    background: #0f172a;
    color: #f8fafc;
    padding: 18px 24px;
    border-bottom: 1px solid #1e293b;
  }
  header h1 {
    margin: 0 0 4px 0;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.01em;
  }
  header p {
    margin: 0;
    font-size: 0.86rem;
    color: #cbd5e1;
  }
  .container {
    max-width: 1360px;
    margin: 0 auto;
    padding: 18px 20px 40px;
  }
  .controls {
    display: grid;
    grid-template-columns: 1.5fr 1fr 1fr;
    gap: 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 16px;
  }
  @media (max-width: 900px) {
    .controls { grid-template-columns: 1fr; }
    .compare-grid { grid-template-columns: 1fr !important; }
  }
  label {
    display: block;
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
    margin-bottom: 5px;
  }
  select {
    width: 100%;
    padding: 8px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--surface);
    color: var(--text);
    font-size: 0.9rem;
  }
  .task-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 16px;
  }
  .task-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
  }
  .task-title {
    font-size: 1.05rem;
    font-weight: 700;
  }
  .badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--surface-alt);
    color: var(--text);
  }
  .badge.ok { background: var(--green-bg); border-color: var(--green-border); color: #065f46; }
  .badge.warn { background: var(--amber-bg); border-color: var(--amber-border); color: #92400e; }
  .badge.err { background: var(--red-bg); border-color: var(--red-border); color: #991b1b; }
  .task-meta {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 10px;
    margin-top: 10px;
  }
  .kv-box {
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.83rem;
  }
  .kv-box h4 {
    margin: 0 0 6px 0;
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--muted);
  }
  code, pre {
    font-family: var(--mono);
    font-size: 0.8rem;
  }
  .outcome-strip {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 8px;
    margin-top: 12px;
  }
  @media (max-width: 780px) {
    .outcome-strip { grid-template-columns: repeat(3, 1fr); }
  }
  .outcome-pill {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 7px 9px;
    text-align: center;
    background: var(--surface);
    cursor: pointer;
    transition: border-color 0.15s ease;
  }
  .outcome-pill:hover { border-color: var(--blue); }
  .outcome-pill .mname { font-weight: 700; font-size: 0.84rem; }
  .outcome-pill .mstat { font-size: 0.74rem; margin-top: 2px; }
  .compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    align-items: start;
  }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
  }
  .panel-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .panel-head h2 {
    margin: 0;
    font-size: 1rem;
  }
  .step-nav {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin-bottom: 12px;
  }
  .step-btn {
    border: 1px solid var(--border);
    background: var(--surface-alt);
    color: var(--text);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.8rem;
    cursor: pointer;
    font-weight: 600;
  }
  .step-btn.active {
    background: var(--text);
    color: #ffffff;
    border-color: var(--text);
  }
  .state-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 6px;
    margin-bottom: 12px;
  }
  .state-cell {
    background: var(--surface-alt);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 8px;
    text-align: center;
  }
  .state-cell .sk { font-size: 0.68rem; color: var(--muted); }
  .state-cell .sv { font-family: var(--mono); font-size: 0.84rem; font-weight: 700; margin-top: 2px; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    margin-bottom: 12px;
  }
  th, td {
    border: 1px solid var(--border);
    padding: 6px 7px;
    text-align: left;
    vertical-align: top;
  }
  th {
    background: var(--surface-alt);
    font-size: 0.74rem;
    color: var(--muted);
  }
  tr.selected-row {
    background: #eff6ff;
  }
  .exec-box {
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 0.82rem;
    border: 1px solid var(--border);
    background: var(--surface-alt);
  }
  .exec-box.illegal { background: var(--red-bg); border-color: var(--red-border); }
  .exec-box.legal { background: var(--green-bg); border-color: var(--green-border); }
  .exec-box.stop { background: var(--amber-bg); border-color: var(--amber-border); }
  .footer-note {
    margin-top: 20px;
    font-size: 0.8rem;
    color: var(--muted);
    text-align: center;
  }
</style>
</head>
<body>
<header>
  <h1>MiniBooking-RLVR — Interactive 1-Step MPC Trajectory Replay Viewer</h1>
  <p>Inspect world-model candidate scoring, tie-break loops, horizon stops, and irreversible traps across 6 predictors (SFT, G1, G2, G3, oracle, null).</p>
</header>
<div class="container">
  <div class="controls">
    <div>
      <label for="caseSelect">Curated Test Case (8 Real Closed-Loop Trajectories)</label>
      <select id="caseSelect"></select>
    </div>
    <div>
      <label for="leftModel">Left Predictor</label>
      <select id="leftModel"></select>
    </div>
    <div>
      <label for="rightModel">Right Predictor (Side-by-Side Comparison)</label>
      <select id="rightModel"></select>
    </div>
  </div>

  <div id="taskSummary" class="task-card"></div>

  <div class="compare-grid">
    <div id="leftPanel" class="panel"></div>
    <div id="rightPanel" class="panel"></div>
  </div>

  <div class="footer-note">
    Frozen planner weights: w_risk=0.0, w_irr=0.0, w_conf=0.1, tau=0.0, max_steps=5. Data exported verbatim from MiniBooking-RLVR single-seed test runs.
  </div>
</div>

<script>
const DATA = __EMBEDDED_DATA__;
const PREDICTORS = ["G2", "G3", "G1", "SFT", "oracle", "null"];
const stateIdx = { left: 0, right: 0 };

const caseSelect = document.getElementById("caseSelect");
const leftSelect = document.getElementById("leftModel");
const rightSelect = document.getElementById("rightModel");

DATA.cases.forEach((c, i) => {
  const opt = document.createElement("option");
  opt.value = i;
  opt.textContent = `${c.task_id} [${c.tag}] — ${c.summary}`;
  caseSelect.appendChild(opt);
});

PREDICTORS.forEach(m => {
  const o1 = document.createElement("option");
  o1.value = m; o1.textContent = m;
  leftSelect.appendChild(o1);
  const o2 = document.createElement("option");
  o2.value = m; o2.textContent = m;
  rightSelect.appendChild(o2);
});
leftSelect.value = "G2";
rightSelect.value = "G3";

function fmtAction(a) {
  return a.day !== undefined && a.day !== null ? `${a.name}(${a.day})` : `${a.name}()`;
}
function fmtChanges(ch) {
  if (!ch) return "<i>invalid</i>";
  if (ch.length === 0) return "<code>[]</code>";
  return ch.map(x => `<code>${x.field}=${JSON.stringify(x.value)}</code>`).join(", ");
}
function fmtGoal(goal) {
  const parts = goal.conditions.map(c => `${c.field} ${c.op} ${JSON.stringify(c.value)}`);
  if (goal.loss_limit_cents !== undefined && goal.loss_limit_cents !== null) {
    parts.push(`loss_cents <= ${goal.loss_limit_cents}`);
  }
  return parts.join(" AND ");
}
function outcomeBadge(ep) {
  if (ep.success) return `<span class="badge ok">SUCCESS (${ep.n_steps} steps)</span>`;
  if (ep.irreversible_executed) return `<span class="badge err">FAIL + IRREVERSIBLE (${ep.end_reason})</span>`;
  if (ep.end_reason === "max_steps") return `<span class="badge err">MAX_STEPS (${ep.n_steps} steps)</span>`;
  return `<span class="badge warn">${ep.end_reason.toUpperCase()} (${ep.n_steps} steps)</span>`;
}

function renderTaskSummary(c) {
  const r = c.rules;
  const refSol = c.reference_solution ? c.reference_solution.map(fmtAction).join(" -> ") : "None (unsolvable in <= 5 steps)";
  const pills = PREDICTORS.map(m => {
    const ep = c.predictors[m];
    const cls = ep.success ? "ok" : (ep.irreversible_executed || ep.end_reason === "max_steps" ? "err" : "warn");
    const label = ep.success ? `OK (${ep.n_steps}s)` : `${ep.end_reason} (${ep.n_steps}s)`;
    return `<div class="outcome-pill" onclick="setRight('${m}')" title="Click to load ${m} on right panel">
      <div class="mname">${m}</div>
      <div class="mstat"><span class="badge ${cls}">${label}</span></div>
    </div>`;
  }).join("");

  document.getElementById("taskSummary").innerHTML = `
    <div class="task-header">
      <div class="task-title">${c.task_id} — ${c.summary}</div>
      <div class="badges">
        <span class="badge">${c.split}</span>
        <span class="badge">${c.category}</span>
        <span class="badge ${c.solvable ? 'ok' : 'warn'}">${c.solvable ? `BFS Solvable (${c.shortest_steps} steps)` : 'Unsolvable'}</span>
      </div>
    </div>
    <div class="task-meta">
      <div class="kv-box">
        <h4>Scenario Rules</h4>
        <code>price=${r.price_cents}, refund=${r.refund_num}/${r.refund_den}, allow_change=${r.allow_change}, fee=${r.change_fee_cents}</code>
      </div>
      <div class="kv-box">
        <h4>Goal Condition</h4>
        <code>${fmtGoal(c.goal)}</code>
      </div>
      <div class="kv-box">
        <h4>BFS Reference Path</h4>
        <code>${refSol}</code>
      </div>
    </div>
    <div class="outcome-strip">${pills}</div>
  `;
}

function buildRounds(ep) {
  const rounds = ep.steps.map(s => ({ kind: "exec", data: s }));
  if (ep.stop_decision) {
    rounds.push({ kind: "stop", data: ep.stop_decision });
  }
  return rounds;
}

function renderPanel(side, c, modelName) {
  const el = document.getElementById(side === "left" ? "leftPanel" : "rightPanel");
  const ep = c.predictors[modelName];
  const rounds = buildRounds(ep);
  if (stateIdx[side] >= rounds.length) stateIdx[side] = 0;

  const navBtns = rounds.map((rd, idx) => {
    const lbl = rd.kind === "stop" ? `Stop (t=${rd.data.step})` : `Step ${rd.data.step}`;
    return `<button class="step-btn ${idx === stateIdx[side] ? 'active' : ''}" onclick="setStep('${side}', ${idx})">${lbl}</button>`;
  }).join("");

  let bodyHtml = "";
  if (rounds.length === 0) {
    bodyHtml = `<div class="exec-box">Ended immediately: <b>${ep.end_reason}</b></div>`;
  } else {
    const cur = rounds[stateIdx[side]];
    const d = cur.data;
    const st = d.state;
    const stateHtml = ["status", "day", "balance_cents", "paid_cents", "loss_cents"].map(k => `
      <div class="state-cell">
        <div class="sk">${k}</div>
        <div class="sv">${JSON.stringify(st[k])}</div>
      </div>
    `).join("");

    const rowsHtml = d.candidates.map((cand, j) => {
      const sel = j === d.best_idx ? 'class="selected-row"' : "";
      const mark = j === d.best_idx ? " [BEST]" : "";
      const sc = cand.score === null ? "-inf" : Number(cand.score).toFixed(2);
      return `<tr ${sel}>
        <td><code>${fmtAction(cand.action)}</code>${mark}</td>
        <td>${fmtChanges(cand.pred_changes)}</td>
        <td>${cand.pred_reversible === null ? "-" : cand.pred_reversible} / ${cand.pred_cost === null ? "-" : cand.pred_cost}</td>
        <td><b>${sc}</b></td>
      </tr>`;
    }).join("");

    let execHtml = "";
    if (cur.kind === "stop") {
      execHtml = `<div class="exec-box stop">
        <b>Decision: STOP (${ep.end_reason})</b> — current progress = <code>${d.progress.toFixed(2)}</code>, best candidate score = <code>${d.best_score === null ? "-inf" : d.best_score.toFixed(2)}</code> (gain &lt;= tau=0.0).
      </div>`;
    } else {
      const ex = d.executed;
      const cls = ex.legal ? "legal" : "illegal";
      execHtml = `<div class="exec-box ${cls}">
        <b>Executed:</b> <code>${fmtAction(ex.action)}</code> |
        <b>legal:</b> <code>${ex.legal}</code> (<code>${ex.reason}</code>) |
        <b>changed:</b> <code>${ex.changed}</code> |
        <b>true_rev:</b> <code>${ex.reversible_true}</code> |
        <b>true_cost:</b> <code>${ex.cost_true}</code><br/>
        <b>True Next State:</b> <code>${JSON.stringify(ex.state)}</code>
      </div>`;
    }

    bodyHtml = `
      <div class="step-nav">${navBtns}</div>
      <div style="font-size:0.78rem;color:var(--muted);margin-bottom:6px;">
        State before decision (progress = <b>${d.progress.toFixed(2)}</b>):
      </div>
      <div class="state-grid">${stateHtml}</div>
      <table>
        <thead>
          <tr>
            <th>Candidate</th>
            <th>Predicted Delta (changes)</th>
            <th>Pred Rev / Cost</th>
            <th>Score</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
      ${execHtml}
    `;
  }

  el.innerHTML = `
    <div class="panel-head">
      <h2>Predictor: ${modelName}</h2>
      <div>${outcomeBadge(ep)}</div>
    </div>
    ${bodyHtml}
  `;
}

function renderAll() {
  const c = DATA.cases[Number(caseSelect.value)];
  renderTaskSummary(c);
  renderPanel("left", c, leftSelect.value);
  renderPanel("right", c, rightSelect.value);
}

window.setStep = function(side, idx) {
  stateIdx[side] = idx;
  renderAll();
};
window.setRight = function(modelName) {
  rightSelect.value = modelName;
  stateIdx.right = 0;
  renderAll();
};

caseSelect.addEventListener("change", () => { stateIdx.left = 0; stateIdx.right = 0; renderAll(); });
leftSelect.addEventListener("change", () => { stateIdx.left = 0; renderAll(); });
rightSelect.addEventListener("change", () => { stateIdx.right = 0; renderAll(); });

renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    raw = TRAJ_PATH.read_text(encoding="utf-8")
    # Ensure valid JSON before embedding
    parsed = json.loads(raw)
    embedded = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__EMBEDDED_DATA__", embedded)
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8", newline="\n")
    print(f"Built {OUT_HTML} ({len(html):,} bytes, {len(parsed['cases'])} cases)")


if __name__ == "__main__":
    main()
