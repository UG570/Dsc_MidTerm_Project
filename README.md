# CSE558 Midterm Project: street lighting, motorcycles and crash severity

Hypothesis tests on UK police-reported road accidents (`accident_data.csv`,
670,679 rows × 14 columns, STATS19-derived Kaggle upload).

The outcome throughout is **KSI** (killed or seriously injured):
`Accident_Severity` is Fatal or Serious.

| Problem statement | Test | Script |
|---|---|---|
| PS1: is the KSI share of rural single-carriageway crashes in darkness higher with no street lighting than with lights lit? | Test 1: two-proportion z-test (+ chi-square, CMH by road surface) | `code/tests_ps1.py` |
| | Test 2: paired t-test on per-district differences (+ Wilcoxon, sign test, unpaired Welch) | `code/tests_ps1.py` |
| PS2: in two-vehicle crashes, are motorcycle crashes more often KSI but with fewer casualties than car crashes? | Test 3: two-proportion z-test (+ 3-level chi-square, relative risk) | `code/tests_ps2.py` |
| | Test 4: Welch t-test on casualties per crash (+ Levene/F, pooled t, z, Mann-Whitney, Poisson check) | `code/tests_ps2.py` |
| All four | Holm correction | `code/holm.py` |

## Layout

```
accident_data.csv        raw data (not committed: 93 MB; put it here)
code/
  stat_helpers.py        two-proportion z-test, CMH, Welch, Holm, JSON I/O
  prepare.py             cleaning, encoding, size check  -> results/preparation.json
  eda.py                 EDA aggregates                   -> results/eda.json, report/figdata/*.csv
  tests_ps1.py           Tests 1 and 2                    -> results/ps1.json
  tests_ps2.py           Tests 3 and 4                    -> results/ps2.json
  holm.py                Holm correction                  -> results/holm.json
  run_all.py             runs everything in order
results/                 every number quoted in the report
report/
  CSE558_Project_Report.tex   the report (course template)
  figdata/                    CSVs the report plots with pgfplots
```

## Running

Needs Python 3.9+ with pandas, numpy and scipy (`pip install -r requirements.txt`).

```bash
python code/run_all.py
```

The run takes under a minute. Each script can also be run on its own, but
`holm.py` needs the JSON files from `tests_ps1.py` and `tests_ps2.py`.

## Building the report

The figures are drawn by pgfplots from `report/figdata/*.csv`, so no plotting
library is needed. Compile `report/CSE558_Project_Report.tex` twice with
pdfLaTeX from inside `report/` (or upload the `report/` folder to Overleaf).

## Data preparation, in short

* 19 exact duplicate rows dropped. The `Index` (accident ID) was mangled into
  scientific notation (`2.01E+07`) for 239,884 rows, so it is not used as a key.
* `Accident Date` is shifted: for every readable ID the date year is the ID year
  plus 12. No test uses the date.
* Six district names had lost the letters "na" (`Stevege`, `Blaeu Gwent`,
  `North/South Larkshire`, `Clackmannshire`, `Crewe and ntwich`); 3,438 rows
  were merged back, so 422 district levels become 416.
* Missing categories become an explicit `Unknown` level rather than the mode,
  so unknown road types or light levels never enter a test's filter.
* Casualty and vehicle counts are kept as they are. Their IQR fence is degenerate
  (Q1 = Q3 = 1 for casualties), and the large values are real crashes.
* Encoded size: 670,660 rows × 52 columns = 34,874,320 cells (≥ 3M).
