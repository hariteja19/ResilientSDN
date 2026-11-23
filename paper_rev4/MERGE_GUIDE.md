# Merging revision 4 into the paper

The paper source (`paper/`) was not on the machine where revision 4 was made, so the changes are
drop-in files. `main_rev4.tex` shows them assembled and compiles on its own (`sh build.sh` in WSL).

## Preamble

```latex
\usepackage{amsthm,algorithm,algpseudocode,booktabs}   % skip any already loaded
\newtheorem{proposition}{Proposition}
\graphicspath{{figures/}{../paper_rev4/figures/}}     % or copy paper_rev4/figures/ into the paper
\input{generated/validation_macros}                   % \valRhoTime, \valAgree, ...
\input{generated/count_macros}                        % \simRunsUDP, ...
\input{generated/sweep_macros}                        % \sweepN, \sweepDefend, \sweepGap..., (Sec. sweep)
\input{generated/floor_macros}                        % \floorMaxRel, \floorFlips, ... (TCP floor check)
```
Paths assume `paper_rev4/` is the working directory; adjust them, or copy `generated/` and
`figures/` next to the paper's `main.tex`.

## Where each file goes

| File | Action in the paper |
|---|---|
| `tab_notation.tex` | Replace the notation table (or add its missing rows) |
| `sec_formulation.tex` | **Replace** Sec. III-C "Problem Formulation", including old Eq. (3) |
| `alg_resilientsdn.tex` | **Replace** Algorithm 1 |
| `sec_method_rev4.tex` | Insert in the evaluation setup, after the existing scenario table (adds S12–S14, new baselines, simulator + validation, statistics) |
| `sec_results_rev4.tex` | Insert as a new subsection after the existing results |
| `sec_sweep_rev4.tex` | Insert right after `sec_results_rev4.tex`: the randomised sweep over the scenario space (answers "the scenario is too simple") |
| `tab_scenario_space.tex` | Already `\input` by `sec_method_rev4.tex`; keep it next to it |
| `sec_related_limits.tex`, first half | Add the two paragraphs to Related Work |
| `sec_related_limits.tex`, second half | **Replace** the Limitations section |
| `sec_frontmatter_rev4.tex` | Proposed abstract, contribution list and conclusion (replace; resolve the `TODO-V3` lines first) |
| `references_rev4.bib` | Append to the paper's `.bib`; delete entries whose key or paper already exists |

## Label clashes to check

The new files use: `sec:formulation`, `sec:scenarios`, `sec:baselines`, `sec:sim`, `sec:stats`,
`sec:results`, `sec:limits`, `eq:load`, `eq:objective`, `eq:unsplit`, `eq:cooldown`, `eq:lp`,
`eq:dpp`, `eq:score`, `eq:phistar`, `alg:rsdn`, `alg:fallback`, `prop:monotone`, `prop:churn`,
`prop:evasion`, `prop:cut`, `sec:sweep`, `tab:v3`, `tab:notation`, `tab:scenario_space`, `tab:sweep`,
`tab:sweep_pattern`, `tab:sim_*`, `tab:paired_*`, `tab:ablation_v2`, `tab:wan_ext`, `fig:validation`,
`fig:tradeoff`, `fig:wan_adversarial`, `fig:multilink`, `fig:tcp_fct`, `fig:sweep_cdf`,
`fig:sweep_factors`.

* `eq:score` must be the paper's existing multi-objective score equation (Score = w1·C + w2·L − w3·K):
  put that label on it and delete the stand-in in `main_rev4.tex`.
* If the paper already has `sec:results` or `sec:formulation`, keep the paper's and rename here.

## Text in the existing sections that must change

* **Abstract / introduction / conclusion:** mention the generalised model, the simulator (30 seeds,
  S12–S14, TCP) and the LP baselines; state that TE-LP/TE-Φ achieve less congestion time by
  splitting flows (do not claim ResilientSDN beats every baseline).
* **Old "Welch t-test" wording:** replace with the paired tests of Sec. `sec:stats` wherever new
  tables are referenced (the old emulation tables may keep their own tests, stated as such).
* **Internet2 S11 discussion:** point to v2 / bottleneck drift as the fix tested in simulation.
* **WAN emulation table:** `generated/tab_wan_ext.tex` and the paragraph marked
  `WAN-EMULATION-PARAGRAPH` in `sec_results_rev4.tex` are filled in after `run_wan_ext.sh` finishes
  and `make_assets.py` is re-run.

## Rebuild everything

```powershell
.venv\Scripts\python paper_rev4\make_assets.py     # tables, figures, statistics from results/
wsl -u root sh paper_rev4/build.sh                  # standalone PDF check
```
