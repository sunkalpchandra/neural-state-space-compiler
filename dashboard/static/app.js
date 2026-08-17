/* nssc dashboard — front-end. Plain JS + Plotly (no build step). */
(() => {
  "use strict";

  // ------------------------------------------------------------------ palette
  const C = {
    ink: "#e8e7e1", ink2: "#b7b6ac", ink3: "#75746c", line: "#2b2b29", lineStrong: "#3a3a37",
    series: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    truth: "#8f8e86", accent: "#3987e5", cf: "#d95926", good: "#0ca30c", warn: "#fab219", bad: "#e66767",
    seq: [[0, "#184f95"], [0.5, "#3987e5"], [1, "#b7d3f6"]],   // single-hue sequential (dark surface)
    seqSoft: [[0, "rgba(24,79,149,0.0)"], [1, "rgba(57,135,229,0.55)"]],
  };
  const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';

  const baseLayout = (extra = {}) => Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: MONO, size: 10.5, color: C.ink2 },
    margin: { l: 44, r: 12, t: 8, b: 32 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: "#202020", bordercolor: C.lineStrong, font: { family: MONO, size: 10.5, color: C.ink } },
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 10 }, bgcolor: "rgba(0,0,0,0)" },
    xaxis: axis(), yaxis: axis(),
    showlegend: true,
  }, extra);
  function axis(extra = {}) {
    return Object.assign({
      gridcolor: C.line, zeroline: false, linecolor: C.lineStrong, tickcolor: C.lineStrong,
      tickfont: { size: 10 }, titlefont: { size: 10.5, color: C.ink3 }, automargin: true,
    }, extra);
  }
  const CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"], displayModeBar: "hover" };

  // ------------------------------------------------------------------ state
  const S = {
    sources: null, summary: null, traj: null, cf: null,
    idx: 0, context: 20, horizon: 50, envelope: true,
    z0: null, // counterfactual z0 (array)
  };
  const $ = (id) => document.getElementById(id);
  const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : (typeof v === "number" ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d)) : String(v));
  const fmtInt = (v) => (v === null || v === undefined) ? "—" : Math.round(v).toLocaleString();

  let inflight = 0;
  function net(state, text) {
    const el = $("net"); el.className = state; $("net-text").textContent = text;
  }
  async function api(path, opts) {
    inflight++; net("busy", path.replace("/api/", ""));
    try {
      const r = await fetch(path, opts);
      if (!r.ok) {
        let msg = r.statusText;
        try { msg = (await r.json()).detail || msg; } catch (_) { /* ignore */ }
        throw new Error(`${r.status} ${msg}`);
      }
      return await r.json();
    } catch (e) {
      toast(e.message); net("err", "error"); throw e;
    } finally {
      inflight--; if (inflight === 0) setTimeout(() => { if (inflight === 0) net("", "idle"); }, 200);
    }
  }
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  let toastTimer;
  function toast(msg) {
    const t = $("toast"); t.textContent = msg; t.classList.add("show");
    clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove("show"), 4000);
  }
  function empty(id, msg) {
    const el = $(id); Plotly.purge(el); el.innerHTML = `<div class="empty">${msg}</div>`;
  }
  function draw(id, traces, layout) {
    const el = $(id); el.querySelectorAll(".empty").forEach((n) => n.remove());
    return Plotly.react(el, traces, layout);
  }
  const methodLabel = (m) => !m ? "" : m.startsWith("gaussian") ? "gaussian MC" : m.startsWith("initial_perturbation") ? "ε-ensemble" : m.split("(")[0];
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  // ------------------------------------------------------------------ sources
  async function loadSources() {
    S.sources = await api("/api/sources");
    const sel = $("source"); sel.innerHTML = "";
    const { experiments, compiles } = S.sources;
    if (compiles.length) {
      const g = document.createElement("optgroup"); g.label = "compile dirs";
      compiles.forEach((c) => {
        const o = document.createElement("option");
        o.value = "compile:" + c.dir;
        const s = c.selected || {};
        o.textContent = `${c.dir.replace(/^results\/compile\//, "")} · ${s.encoder}+${s.dynamics}@d${s.latent_dim} · ${c.dataset || ""}${c.has_checkpoint ? "" : " (no ckpt)"}`;
        o.disabled = !c.has_checkpoint; g.appendChild(o);
      });
      sel.appendChild(g);
    }
    if (experiments.length) {
      const g = document.createElement("optgroup"); g.label = "registry experiments";
      experiments.slice().reverse().forEach((e) => {
        const o = document.createElement("option");
        o.value = "exp:" + e.load_id;
        const roll = e.metrics["test/recursive/nrmse_mean"];
        o.textContent = `${e.id} · ${e.model} · ${e.dataset}` + (roll !== undefined ? ` · roll ${fmt(roll, 3)}` : "");
        g.appendChild(o);
      });
      sel.appendChild(g);
    }
    if (!experiments.length && !compiles.length) {
      sel.innerHTML = '<option value="">no completed runs found — run `nssc smoke`</option>';
    }
    $("source-hint").textContent = `${experiments.length} run(s), ${compiles.length} compile dir(s)`;
  }

  async function loadSelected() {
    const v = $("source").value; if (!v) return;
    const body = v.startsWith("compile:") ? { compile_dir: v.slice(8) } : { experiment_id: v.slice(4) };
    $("btn-load").disabled = true;
    try {
      S.summary = await post("/api/load", body);
    } finally { $("btn-load").disabled = false; }
    onLoaded();
  }

  function onLoaded() {
    const s = S.summary;
    $("st-exp").textContent = s.experiment_id;
    $("st-model").textContent = s.model_name;
    $("st-dataset").textContent = s.dataset;
    $("st-params").textContent = fmtInt(s.params.total) + ` (E ${fmtInt(s.params.encoder)} / F ${fmtInt(s.params.dynamics)} / D ${fmtInt(s.params.decoder)})`;
    $("st-d").textContent = s.latent_dim + (s.has_z_true ? ` (true ${s.true_latent_dim})` : "");
    $("st-D").textContent = s.obs_dim;
    $("st-nT").textContent = `${s.n_traj} × ${s.T}`;
    $("st-dyn").textContent = s.dynamics + (s.is_stochastic ? " (stochastic)" : "");
    $("st-commit").textContent = (s.git_commit || "—").slice(0, 10);
    // sliders
    const idx = $("idx"); idx.max = Math.max(0, s.n_traj - 1); idx.value = Math.min(S.idx, s.n_traj - 1); S.idx = +idx.value;
    const ctx = $("context"); ctx.max = s.T - 1; S.context = Math.min(S.context, s.T - 2); ctx.value = S.context;
    const hor = $("horizon"); hor.max = s.T - 1; S.horizon = Math.min(S.horizon, s.T - S.context); hor.value = S.horizon;
    syncSliderLabels();
    // dim selectors
    fillDimSelects(s.latent_dim);
    S.z0 = null;
    refreshAll();
  }

  function fillDimSelects(d) {
    const labels = Array.from({ length: d }, (_, i) => `z${i + 1}`);
    const fill = (id, def, allowNone) => {
      const el = $(id); const prev = el.value; el.innerHTML = "";
      if (allowNone) { const o = document.createElement("option"); o.value = "-1"; o.textContent = "—"; el.appendChild(o); }
      labels.forEach((l, i) => { const o = document.createElement("option"); o.value = String(i); o.textContent = l; el.appendChild(o); });
      el.value = (prev !== "" && +prev < d && prev !== undefined) ? prev : String(def);
      if (el.value === "") el.value = String(def);
    };
    fill("ph-a", 0, false); fill("ph-b", Math.min(1, d - 1), false); fill("ph-c", d >= 3 ? 2 : -1, true);
    fill("fd-a", 0, false); fill("fd-b", Math.min(1, d - 1), false);
    if (d < 3) $("ph-c").value = "-1";
  }

  function syncSliderLabels() {
    $("idx-val").textContent = S.idx; $("context-val").textContent = S.context; $("horizon-val").textContent = S.horizon;
  }

  // ------------------------------------------------------------------ refresh
  async function refreshAll() {
    if (!S.summary) return;
    await Promise.all([refreshTrajectory(), refreshField(), refreshStability(), refreshCompile()]);
  }

  async function refreshTrajectory() {
    const t = await api(`/api/trajectory?idx=${S.idx}&context=${S.context}&horizon=${S.horizon}`);
    S.traj = t;
    // server may clamp
    if (t.context !== S.context || t.horizon !== S.horizon) {
      S.context = t.context; S.horizon = t.horizon; $("context").value = S.context; $("horizon").value = S.horizon; syncSliderLabels();
    }
    plotRaw(t); plotLatent(t); plotPhase(t); plotPred(t);
    buildCfSliders(t);
    await refreshCounterfactual();
  }

  // ------------------------------------------------------------------ panel 1
  function plotRaw(t) {
    const D = t.dims.length;
    const traces = t.dims.map((d, k) => ({
      type: "scatter", mode: "lines", name: t.dim_labels[k], x: t.t, y: t.x.map((r) => r[k]),
      line: { width: 1.5, color: C.series[k % 8] }, hovertemplate: "%{y:.3f}",
    }));
    $("m-raw").textContent = `test[${t.idx}] · D=${S.summary.obs_dim}${S.summary.obs_dim > D ? ` (showing ${D})` : ""} · T=${t.T} · dt=${t.dt}`;
    draw("plot-raw", traces, baseLayout({
      xaxis: axis({ title: "t" }), yaxis: axis({ title: "x (normalised)" }), showlegend: D > 1,
    }));
  }

  // ------------------------------------------------------------------ panel 2
  function plotLatent(t) {
    const d = t.z[0].length;
    const traces = [];
    for (let k = 0; k < d; k++) {
      traces.push({ type: "scatter", mode: "lines", name: `z${k + 1}`, x: t.t, y: t.z.map((r) => r[k]),
        line: { width: 1.5, color: C.series[k % 8] }, hovertemplate: "%{y:.3f}" });
    }
    const shapes = [{ type: "rect", xref: "x", yref: "paper", x0: t.t[0], x1: t.t[t.context - 1], y0: 0, y1: 1,
      fillcolor: "rgba(255,255,255,0.035)", line: { width: 0 }, layer: "below" }];
    $("m-latent").textContent = `d=${d} · encoder ${S.summary.encoder} · shaded = context`;
    draw("plot-latent", traces, baseLayout({
      xaxis: axis({ title: "t" }), yaxis: axis({ title: "z" }), shapes, showlegend: d > 1,
    }));
  }

  // ------------------------------------------------------------------ panel 3
  function plotPhase(t) {
    const a = +$("ph-a").value, b = +$("ph-b").value, c = +$("ph-c").value;
    const d = t.z[0].length;
    if (d < 2) { empty("plot-phase", "latent dim < 2 — no phase portrait"); return; }
    const time = t.z.map((_, i) => i);
    const marker = { size: 4, color: time, colorscale: C.seq, showscale: true,
      colorbar: { title: { text: "t idx", font: { size: 10 } }, thickness: 8, len: 0.6, tickfont: { size: 9 }, outlinewidth: 0 } };
    const line = { width: 1, color: "rgba(183,182,172,0.35)" };
    const traces = []; let layout;
    if (c >= 0 && d >= 3) {
      traces.push({ type: "scatter3d", mode: "lines+markers", name: "z(t)", marker: { ...marker, size: 2.5 }, line,
        x: t.z.map((r) => r[a]), y: t.z.map((r) => r[b]), z: t.z.map((r) => r[c]), hovertemplate: "z%{x:.3f} / %{y:.3f} / %{z:.3f}" });
      const ax3 = (title) => ({ title: { text: title, font: { size: 10 } }, gridcolor: C.line, zerolinecolor: C.lineStrong, backgroundcolor: "rgba(0,0,0,0)", showbackground: false, tickfont: { size: 9 } });
      layout = baseLayout({ margin: { l: 0, r: 0, t: 0, b: 0 }, hovermode: "closest", showlegend: false,
        scene: { xaxis: ax3(`z${a + 1}`), yaxis: ax3(`z${b + 1}`), zaxis: ax3(`z${c + 1}`), camera: { eye: { x: 1.5, y: 1.4, z: 1.1 } } } });
      $("m-phase").textContent = `z${a + 1} × z${b + 1} × z${c + 1} · colour = time`;
    } else {
      traces.push({ type: "scatter", mode: "lines+markers", name: "z(t)", marker, line,
        x: t.z.map((r) => r[a]), y: t.z.map((r) => r[b]), hovertemplate: "z%{x:.3f} / %{y:.3f}" });
      traces.push({ type: "scatter", mode: "markers", name: "z₀ (t=0)", x: [t.z[0][a]], y: [t.z[0][b]],
        marker: { size: 9, color: "rgba(0,0,0,0)", line: { color: C.ink, width: 1.5 } }, hoverinfo: "skip" });
      layout = baseLayout({ hovermode: "closest", showlegend: false,
        xaxis: axis({ title: `z${a + 1}` }), yaxis: axis({ title: `z${b + 1}`, scaleanchor: "x", scaleratio: 1 }) });
      $("m-phase").textContent = `z${a + 1} × z${b + 1} · colour = time · ring = start`;
    }
    draw("plot-phase", traces, layout);
  }

  // ------------------------------------------------------------------ panel 4
  function plotPred(t) {
    const D = t.dims.length;
    const nRows = D + 1;   // + nrmse row
    const tPred = t.t.slice(t.context, t.context + t.horizon);
    const traces = [], layout = baseLayout({ margin: { l: 44, r: 12, t: 8, b: 30 }, showlegend: true, hovermode: "x unified" });
    const gap = 0.035; const h = (1 - gap * (nRows - 1)) / nRows;
    const shapes = [];
    for (let k = 0; k < nRows; k++) {
      const yk = k === 0 ? "y" : `y${k + 1}`;
      const dom0 = 1 - (k + 1) * h - k * gap, dom1 = 1 - k * h - k * gap;
      const isErr = k === D;
      layout[k === 0 ? "yaxis" : `yaxis${k + 1}`] = axis({ domain: [dom0, dom1], title: isErr ? "nrmse" : t.dim_labels[k], titlefont: { size: 9.5, color: C.ink3 }, tickfont: { size: 9 }, nticks: 4 });
      if (!isErr) {
        traces.push({ type: "scatter", mode: "lines", name: "truth", legendgroup: "truth", showlegend: k === 0,
          x: t.t, y: t.x.map((r) => r[k]), yaxis: yk, line: { width: 1.2, color: C.truth }, hovertemplate: "%{y:.3f}" });
        if (S.envelope && t.std) {
          const up = t.x_hat.map((r, i) => r[k] + 2 * t.std[i][k]), lo = t.x_hat.map((r, i) => r[k] - 2 * t.std[i][k]);
          traces.push({ type: "scatter", mode: "lines", x: tPred, y: up, yaxis: yk, line: { width: 0 }, hoverinfo: "skip", showlegend: false, legendgroup: "env" });
          traces.push({ type: "scatter", mode: "lines", x: tPred, y: lo, yaxis: yk, line: { width: 0 }, fill: "tonexty",
            fillcolor: "rgba(57,135,229,0.16)", hoverinfo: "skip", name: "±2σ", legendgroup: "env", showlegend: k === 0 });
        }
        traces.push({ type: "scatter", mode: "lines", name: "prediction", legendgroup: "pred", showlegend: k === 0,
          x: tPred, y: t.x_hat.map((r) => r[k]), yaxis: yk, line: { width: 1.6, color: C.accent }, hovertemplate: "%{y:.3f}" });
      } else {
        traces.push({ type: "scatter", mode: "lines", name: "nrmse(k)", showlegend: true, x: tPred, y: t.nrmse, yaxis: yk,
          line: { width: 1.4, color: C.series[3] }, hovertemplate: "%{y:.3f}" });
        shapes.push({ type: "line", xref: "paper", yref: yk, x0: 0, x1: 1, y0: 1, y1: 1, line: { color: C.lineStrong, width: 1, dash: "dot" } });
      }
    }
    // context shading (paper-wide) + end-of-context marker + nrmse=1 reference
    layout.shapes = [{ type: "rect", xref: "x", yref: "paper", x0: t.t[0], x1: t.t[t.context - 1], y0: 0, y1: 1,
      fillcolor: "rgba(255,255,255,0.04)", line: { width: 0 }, layer: "below" },
      { type: "line", xref: "x", yref: "paper", x0: t.t[t.context - 1], x1: t.t[t.context - 1], y0: 0, y1: 1, line: { color: C.lineStrong, width: 1, dash: "dot" } },
      ...shapes];
    layout.xaxis = axis({ title: "t", anchor: `y${nRows}` });
    const last = t.nrmse[t.nrmse.length - 1];
    $("m-pred").textContent = `ctx ${t.context} · H ${t.horizon} · nrmse@H ${fmt(last)}` + (t.std && S.envelope ? ` · σ: ${methodLabel(t.std_method)}` : "");
    $("env-method").textContent = t.std_method ? `(${methodLabel(t.std_method)})` : "";
    draw("plot-pred", traces, layout);
  }

  // ------------------------------------------------------------------ panel 5
  async function refreshField() {
    if (!S.summary) return;
    const d = S.summary.latent_dim;
    if (d < 2) { empty("plot-field", "latent dim < 2 — no plane"); return; }
    const a = +$("fd-a").value, b = +$("fd-b").value, g = +$("fd-grid").value;
    if (a === b) { empty("plot-field", "choose two distinct latent dims"); return; }
    const f = await api(`/api/field?dims=${a},${b}&grid=${g}&idx=${S.idx}`);
    plotField(f, a, b);
  }
  function plotField(f, a, b) {
    const n = f.grid; const dx = (f.x[n - 1] - f.x[0]) / (n - 1), dy = (f.y[n - 1] - f.y[0]) / (n - 1);
    let vmax = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) vmax = Math.max(vmax, Math.hypot(f.u[i][j], f.v[i][j]));
    const scale = vmax > 0 ? 0.85 * Math.min(dx, dy) / vmax : 1;
    const lx = [], ly = [];
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x0 = f.x[j], y0 = f.y[i], u = f.u[i][j] * scale, v = f.v[i][j] * scale;
      const x1 = x0 + u, y1 = y0 + v;
      const L = Math.hypot(u, v); if (L === 0) continue;
      const hx = -u / L, hy = -v / L, hl = Math.min(0.3 * L, 0.25 * Math.min(dx, dy));
      // shaft
      lx.push(x0, x1, null); ly.push(y0, y1, null);
      // head (two barbs)
      const cs = Math.cos(0.5), sn = Math.sin(0.5);
      lx.push(x1 + hl * (hx * cs - hy * sn), x1, x1 + hl * (hx * cs + hy * sn), null);
      ly.push(y1 + hl * (hx * sn + hy * cs), y1, y1 + hl * (-hx * sn + hy * cs), null);
    }
    const traces = [
      { type: "heatmap", x: f.x, y: f.y, z: f.speed, colorscale: C.seqSoft, showscale: true, hoverinfo: "skip", zsmooth: "best",
        colorbar: { title: { text: "|F(z)−z|", font: { size: 10 } }, thickness: 8, len: 0.6, tickfont: { size: 9 }, outlinewidth: 0 } },
      { type: "scatter", mode: "lines", x: lx, y: ly, name: "F(z)−z", line: { color: "rgba(232,231,225,0.55)", width: 1 }, hoverinfo: "skip", connectgaps: false },
      { type: "scatter", mode: "lines", x: f.traj.x, y: f.traj.y, name: "z(t) test traj", line: { color: C.series[1], width: 1.6 }, hovertemplate: "z%{x:.3f}/%{y:.3f}" },
      { type: "scatter", mode: "markers", x: [f.traj.x[0]], y: [f.traj.y[0]], name: "start", marker: { size: 8, color: "rgba(0,0,0,0)", line: { color: C.series[1], width: 1.5 } }, hoverinfo: "skip" },
    ];
    const others = f.centre.map((v, k) => (k === a || k === b) ? null : `z${k + 1}=${fmt(v, 2)}`).filter(Boolean);
    $("m-field").textContent = `z${a + 1} × z${b + 1} · ${n}² grid` + (others.length ? ` · plane at ${others.join(", ")}` : "");
    draw("plot-field", traces, baseLayout({ hovermode: "closest", showlegend: false,
      xaxis: axis({ title: `z${a + 1}`, range: [f.x[0] - dx / 2, f.x[n - 1] + dx / 2] }),
      yaxis: axis({ title: `z${b + 1}`, range: [f.y[0] - dy / 2, f.y[n - 1] + dy / 2] }) }));
  }

  // ------------------------------------------------------------------ panel 6
  async function refreshStability() {
    if (!S.summary) return;
    const s = await api(`/api/stability?idx=${S.idx}`);
    plotStability(s);
  }
  function plotStability(s) {
    const M = s.n_points, d = s.real[0].length;
    const re = [], im = [], col = [], txt = [];
    for (let i = 0; i < M; i++) for (let k = 0; k < d; k++) { re.push(s.real[i][k]); im.push(s.imag[i][k]); col.push(s.t_index[i]); txt.push(`t=${s.t_index[i]} λ${k + 1}`); }
    const th = Array.from({ length: 181 }, (_, i) => i * Math.PI / 90);
    const traces = [
      { type: "scatter", mode: "lines", x: th.map(Math.cos), y: th.map(Math.sin), line: { color: C.lineStrong, width: 1 }, hoverinfo: "skip", showlegend: false, xaxis: "x", yaxis: "y" },
      { type: "scatter", mode: "markers", x: re, y: im, text: txt, name: "eig ∂F/∂z", xaxis: "x", yaxis: "y",
        marker: { size: 5, color: col, colorscale: C.seq, showscale: false, line: { width: 0 } }, hovertemplate: "%{text}<br>%{x:.3f} %{y:+.3f}i" },
      { type: "histogram", x: s.spectral_radius, name: "ρ(z_t)", xaxis: "x2", yaxis: "y2", nbinsx: 24,
        marker: { color: C.accent, line: { color: "#1a1a19", width: 1 } }, hovertemplate: "ρ∈[%{x}] n=%{y}", opacity: 0.9 },
    ];
    const rmax = Math.max(1.05, ...re.map(Math.abs), ...im.map(Math.abs)) * 1.05;
    const m = s.metrics || {};
    const verdict = m.verdict || "n/a";
    $("m-stab").textContent = `${M} pts · ρ_max(local) ${fmt(s.rho_max_local)} · frac ρ>1 ${fmt(s.frac_expanding, 2)} · ${s.metrics_split} verdict: ${verdict}` + (m.lyapunov_max !== null && m.lyapunov_max !== undefined ? ` · λ_max ${fmt(m.lyapunov_max)}` : "");
    draw("plot-stab", traces, baseLayout({ hovermode: "closest", showlegend: false, margin: { l: 44, r: 12, t: 8, b: 32 },
      xaxis: axis({ title: "Re λ", domain: [0, 0.5], range: [-rmax, rmax], zeroline: true, zerolinecolor: C.line }),
      yaxis: axis({ title: "Im λ", range: [-rmax, rmax], scaleanchor: "x", scaleratio: 1, zeroline: true, zerolinecolor: C.line }),
      xaxis2: axis({ title: "spectral radius ρ", domain: [0.62, 1] }), yaxis2: axis({ title: "count", anchor: "x2" }),
      shapes: [{ type: "line", xref: "x2", yref: "paper", x0: 1, x1: 1, y0: 0, y1: 1, line: { color: C.warn, width: 1, dash: "dot" } }],
      annotations: [{ xref: "x2", yref: "paper", x: 1, y: 1, text: "ρ=1", showarrow: false, font: { size: 9, color: C.warn }, xanchor: "left", yanchor: "top" }],
    }));
  }

  // ------------------------------------------------------------------ panel 7
  async function refreshCompile() {
    const rep = await api("/api/compile_report");
    const tbl = $("compile-table"), reasons = $("compile-reasons"), funnel = $("compile-funnel");
    if (!rep) {
      tbl.innerHTML = '<div class="empty" style="position:static;padding:30px">no compile report for this source — load a compile dir (results/compile/*/compile_report.json)</div>';
      empty("plot-compile", "score decomposition unavailable"); reasons.innerHTML = ""; funnel.innerHTML = "";
      $("m-compile").textContent = "source is a single registry run";
      return;
    }
    const rk = rep.rollout_key || "";
    const selName = `${rep.selected.encoder}+${rep.selected.dynamics}@d${rep.selected.latent_dim}`;
    const rows = rep.ranking || [];
    const th = ["#", "candidate", "d", "recon", "1-step", `roll${rk.includes("@") ? "@" + rk.split("@")[1] : ""}`, "params", "ρ_max", "verdict", "J"];
    let html = `<table class="rank"><thead><tr>${th.map((h, i) => `<th class="${i === 1 ? "name" : ""}">${h}</th>`).join("")}</tr></thead><tbody>`;
    rows.forEach((r) => {
      const a = r.agg || {}; const name = r.name || r.candidate_id;
      const isSel = name.split("-")[0] === selName || name === selName;
      const v = a["val/stability/verdict"] || "?";
      html += `<tr class="${isSel ? "sel" : ""}"><td>${r.rank ?? ""}</td><td class="name">${name}</td><td>${a["val/latent_dim"] ?? name.match(/@d(\d+)/)?.[1] ?? ""}</td>` +
        `<td>${fmt(a["val/recon/nrmse"], 4)}</td><td>${fmt(a["val/teacher_forced/nrmse"], 4)}</td><td>${fmt(a[rk], 4)}</td>` +
        `<td>${fmtInt(a["val/params/total"])}</td><td>${fmt(a["val/stability/rho_max"])}</td><td><span class="verdict ${v}">${v}</span></td><td>${fmt(r.score)}</td></tr>`;
    });
    html += "</tbody></table>";
    tbl.innerHTML = html;
    reasons.innerHTML = (rep.reasons || []).map((r) => `<li>${r}</li>`).join("");
    const st = rep.stage_summaries || []; const maxN = Math.max(1, ...st.map((s) => s.n_candidates));
    funnel.innerHTML = st.map((s) => `<div class="funnel-row"><span>${s.stage}</span><span class="bar"><i style="width:${100 * s.n_survivors / maxN}%"></i></span><span class="n">${s.n_survivors}/${s.n_candidates}</span></div>`).join("")
      + `<div class="hint">${rep.n_runs} runs · ${rep.n_failed} failed · ${(rep.wall_time_s / 60).toFixed(1)} min</div>`;
    // stacked bar of weighted score terms (fixed term order → fixed colours)
    const W = rep.weights || {};
    const termDefs = [["recon", "reconstruction"], ["one_step", "one_step"], ["rollout", "rollout"], ["complexity", "complexity"], ["instability", "stability"], ["blowup", "blowup_penalty"]];
    const names = rows.map((r) => r.name || r.candidate_id);
    const traces = termDefs.map(([k, wk], i) => ({
      type: "bar", orientation: "h", name: `${k}${W[wk] !== undefined ? ` ×${W[wk]}` : ""}`, y: names,
      x: rows.map((r) => { const t = (r.terms || {})[k]; const w = W[wk] !== undefined ? W[wk] : 1; return (t === null || t === undefined) ? 0 : t * w; }),
      marker: { color: C.series[i], line: { color: "#1a1a19", width: 1 } }, hovertemplate: `${k}: %{x:.3f}<extra></extra>`,
    }));
    traces.push({ type: "scatter", mode: "markers", name: "J", y: names, x: rows.map((r) => r.score), marker: { symbol: "line-ns", size: 12, color: C.ink, line: { width: 1.5, color: C.ink } }, hovertemplate: "J=%{x:.3f}<extra></extra>" });
    $("m-compile").textContent = `selected ${selName} · criterion ${W.criterion || "weighted"} · rollout key ${rk}`;
    draw("plot-compile", traces, baseLayout({ barmode: "relative", hovermode: "y unified", margin: { l: 10, r: 12, t: 8, b: 32 },
      xaxis: axis({ title: "weighted score terms (lower is better)" }), yaxis: axis({ autorange: "reversed", tickfont: { size: 9 }, automargin: true }),
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 9.5 } } }));
  }

  // ------------------------------------------------------------------ panel 8
  function buildCfSliders(t) {
    const z = t.z, d = z[0].length, z0 = z[t.context - 1];
    if (!S.z0 || S.z0.length !== d) S.z0 = z0.slice();
    // range = z0 ± 3σ of latent along trajectory (min ±0.5)
    const sig = Array.from({ length: d }, (_, k) => { const col = z.map((r) => r[k]); const m = col.reduce((a, b) => a + b, 0) / col.length; return Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / col.length); });
    const box = $("cf-sliders"); box.innerHTML = "";
    for (let k = 0; k < d; k++) {
      const span = Math.max(3 * sig[k], 0.5);
      const w = document.createElement("div"); w.className = "cf-slider";
      w.innerHTML = `<label><span>z${k + 1}</span><span><span class="orig">${fmt(z0[k], 2)} →</span> <span class="val" id="cf-v${k}">${fmt(S.z0[k], 3)}</span></span></label>` +
        `<input type="range" id="cf-s${k}" min="${z0[k] - span}" max="${z0[k] + span}" step="${(span / 200).toPrecision(2)}" value="${S.z0[k]}">`;
      box.appendChild(w);
      w.querySelector("input").addEventListener("input", (e) => { S.z0[k] = +e.target.value; $(`cf-v${k}`).textContent = fmt(S.z0[k], 3); cfDebounced(); });
    }
    box.dataset.orig = JSON.stringify(z0);
  }
  const cfDebounced = debounce(() => refreshCounterfactual(), 120);
  async function refreshCounterfactual() {
    if (!S.summary || !S.z0) return;
    const c = await post("/api/counterfactual", { idx: S.idx, context: S.context, horizon: S.horizon, z0: S.z0 });
    S.cf = c; plotCf(c);
  }
  function plotCf(c) {
    const D = c.dims.length, tPred = c.t.slice(c.context, c.context + c.horizon);
    const traces = [], layout = baseLayout({ margin: { l: 44, r: 12, t: 8, b: 30 } });
    const gap = 0.035, h = (1 - gap * (D - 1)) / D;
    for (let k = 0; k < D; k++) {
      const yk = k === 0 ? "y" : `y${k + 1}`;
      layout[k === 0 ? "yaxis" : `yaxis${k + 1}`] = axis({ domain: [1 - (k + 1) * h - k * gap, 1 - k * h - k * gap], title: c.dim_labels[k], titlefont: { size: 9.5, color: C.ink3 }, tickfont: { size: 9 }, nticks: 4 });
      traces.push({ type: "scatter", mode: "lines", name: "truth", legendgroup: "t", showlegend: k === 0, x: tPred, y: c.truth.map((r) => r[k]), yaxis: yk, line: { width: 1.2, color: C.truth }, hovertemplate: "%{y:.3f}" });
      traces.push({ type: "scatter", mode: "lines", name: "rollout from encoded z₀", legendgroup: "o", showlegend: k === 0, x: tPred, y: c.x_hat_original.map((r) => r[k]), yaxis: yk, line: { width: 1.5, color: C.accent }, hovertemplate: "%{y:.3f}" });
      traces.push({ type: "scatter", mode: "lines", name: "counterfactual z₀′", legendgroup: "c", showlegend: k === 0, x: tPred, y: c.x_hat.map((r) => r[k]), yaxis: yk, line: { width: 1.5, color: C.cf, dash: "solid" }, hovertemplate: "%{y:.3f}" });
    }
    layout.xaxis = axis({ title: "t", anchor: `y${D}` });
    const dEnd = c.divergence[c.divergence.length - 1];
    $("m-cf").textContent = `‖Δz₀‖ ${fmt(c.delta_z0)} · decoded gap @H ${fmt(dEnd)}`;
    $("cf-info").textContent = `z₀′ = [${c.z0.map((v) => fmt(v, 2)).join(", ")}] · ‖Δz₀‖=${fmt(c.delta_z0)} · rms gap @H=${fmt(dEnd)}`;
    draw("plot-cf", traces, layout);
  }
  function perturb() {
    if (!S.z0) return;
    const eps = +$("cf-eps").value || 0;
    const g = S.z0.map(() => { let u = 0, v = 0; while (u === 0) u = Math.random(); while (v === 0) v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); });
    const n = Math.hypot(...g) || 1;
    const orig = JSON.parse($("cf-sliders").dataset.orig || "null") || S.z0;
    S.z0 = orig.map((z, k) => z + eps * g[k] / n);
    S.z0.forEach((v, k) => { const s = $(`cf-s${k}`); if (s) { s.value = v; $(`cf-v${k}`).textContent = fmt(v, 3); } });
    refreshCounterfactual();
  }
  function cfReset() {
    const orig = JSON.parse($("cf-sliders").dataset.orig || "null"); if (!orig) return;
    S.z0 = orig.slice();
    S.z0.forEach((v, k) => { const s = $(`cf-s${k}`); if (s) { s.value = v; $(`cf-v${k}`).textContent = fmt(v, 3); } });
    refreshCounterfactual();
  }

  // ------------------------------------------------------------------ wiring
  const trajDebounced = debounce(() => { refreshTrajectory(); }, 150);
  const trajAndStabDebounced = debounce(() => { S.z0 = null; refreshTrajectory(); refreshField(); refreshStability(); }, 150);
  $("idx").addEventListener("input", (e) => { S.idx = +e.target.value; syncSliderLabels(); trajAndStabDebounced(); });
  $("context").addEventListener("input", (e) => { S.context = +e.target.value; syncSliderLabels(); S.z0 = null; trajDebounced(); });
  $("horizon").addEventListener("input", (e) => { S.horizon = +e.target.value; syncSliderLabels(); trajDebounced(); });
  $("envelope").addEventListener("change", (e) => { S.envelope = e.target.checked; if (S.traj) plotPred(S.traj); });
  ["ph-a", "ph-b", "ph-c"].forEach((id) => $(id).addEventListener("change", () => S.traj && plotPhase(S.traj)));
  ["fd-a", "fd-b", "fd-grid"].forEach((id) => $(id).addEventListener("change", () => refreshField()));
  $("btn-load").addEventListener("click", loadSelected);
  $("btn-rescan").addEventListener("click", loadSources);
  $("btn-perturb").addEventListener("click", perturb);
  $("btn-cf-reset").addEventListener("click", cfReset);
  $("source").addEventListener("keydown", (e) => { if (e.key === "Enter") loadSelected(); });

  // resize plots with the grid
  const ro = new ResizeObserver(debounce(() => document.querySelectorAll(".plot").forEach((p) => { if (p.data) Plotly.Plots.resize(p); }), 80));
  document.querySelectorAll(".plot").forEach((p) => ro.observe(p));

  // ------------------------------------------------------------------ boot
  ["plot-raw", "plot-latent", "plot-phase", "plot-pred", "plot-field", "plot-stab", "plot-compile", "plot-cf"].forEach((id) => empty(id, "no source loaded"));
  (async () => {
    try {
      await loadSources();
      // resume if the server already has a source loaded, else auto-load the first option
      try { const r = await fetch("/api/summary"); S.summary = r.ok ? await r.json() : null; } catch (_) { S.summary = null; }
      if (S.summary) {
        const v = S.summary.key.startsWith("compile:") ? "compile:" + S.summary.key.slice(8) : "exp:" + S.summary.experiment_id;
        const opt = Array.from($("source").options).find((o) => o.value === v || o.value.startsWith(v + "@"));
        if (opt) $("source").value = opt.value;
        onLoaded();
      } else {
        const first = Array.from($("source").options).find((o) => o.value && !o.disabled);
        if (first) { $("source").value = first.value; loadSelected(); }
      }
    } catch (e) { console.error(e); }
  })();
})();
