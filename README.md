# Lit roads, lighter crashes?

**CSE558 Data Science · Midterm Project · Monsoon 2026 · Team Dark Horses**

We use hypothesis tests on 670,679 UK road accidents to answer two questions:

1. **Street lighting:** On rural single-carriageway roads at night, are crashes more severe when there are no street lights?
2. **Motorcycles:** In two-vehicle crashes, are motorcycle crashes more severe than car crashes, but with fewer people hurt?

"Severe" means **KSI** (killed or seriously injured), i.e. `Accident_Severity` is Fatal or Serious.

Dataset: [Accident Analysis Dataset on Kaggle](https://www.kaggle.com/datasets/deeplumiere/accident-analysis-dataset)

---

## Results at a glance

| Test | Question | Result | Decision |
|---|---|---|---|
| 1 | KSI share: unlit vs lit roads | Unlit is **+3.5 pp** higher (24.5% vs 21.0%) | Reject H₀ |
| 2 | Same gap, checked district by district | Unlit is **+3.1 pp** higher on average (247 districts) | Reject H₀ |
| 3 | KSI share: motorcycle vs car | No real difference (11.4% vs 11.5%) | Fail to reject H₀ |
| 4 | Casualties per crash: motorcycle vs car | Motorcycles slightly fewer (1.36 vs 1.38) | Reject H₀ |

All four decisions stay the same after the Holm correction for multiple testing.

**In short:** unlit rural roads are linked to more severe crashes. Motorcycle crashes in this file are not more severe than car crashes.

---

## How to run

**1. Install the packages** (Python 3.9 or newer):

```bash
pip install -r requirements.txt
```

**2. Add the data.** Download `accident_data.csv` (93 MB) from Kaggle and put it in the project root. It is not in the repo because of its size.

**3. Run everything:**

```bash
python code/run_all.py
```

This takes under a minute. Every number in the report is saved to the `results/` folder.

---

## Project structure

```
accident_data.csv      the dataset (download it yourself)
requirements.txt       Python packages
code/
  run_all.py           runs every step below, in order
  prepare.py           cleans and encodes the data
  eda.py               summary statistics and figure data
  tests_ps1.py         Tests 1 and 2 (street lighting)
  tests_ps2.py         Tests 3 and 4 (motorcycles)
  holm.py              Holm correction across the four tests
  stat_helpers.py      shared statistics functions
results/               JSON output from each script
report/
  DSc_Project.pdf      the final report
  figdata/             CSV data behind the report's figures
```

Each script can also run on its own. Run `tests_ps1.py` and `tests_ps2.py` before `holm.py`, because it reads their output.

---

## Data cleaning

| Problem | What we did |
|---|---|
| 19 duplicate rows | Removed them |
| Accident ID shown as `2.01E+07` in 239,884 rows | Did not use the ID as a key |
| Dates are 12 years off from the ID year | Did not use the date in any test |
| 6 district names lost the letters "na" (e.g. `Stevege`) | Fixed them; 422 districts became 416 |
| Missing categories | Labelled as `Unknown` instead of guessing |
| Very large casualty and vehicle counts | Kept them, since they are real crashes |

After encoding, the data is 670,660 rows × 52 columns = **34.9 million cells**, which is above the 3 million required.

---

## Building the report

The figures are drawn by LaTeX (pgfplots) from `report/figdata/*.csv`, so you don't need a plotting library. Compile `report/CSE558_Project_Report.tex` twice with pdfLaTeX, or upload the `report/` folder to Overleaf.

---

## Team

| Name | Roll no. | Contribution |
|---|---|---|
| Devan Goyal | 2023191 | Dataset, size check and data cleaning |
| Harsh Garg | 2023228 | EDA, figures and problem statements |
| Rushil Singha | 2023451 | Street lighting tests (Tests 1 and 2) |
| Sanchit Maini | 2023480 | Motorcycle tests (Tests 3 and 4) |
| Utkarsh Garg | 2023570 | Holm correction, limitations, README |
