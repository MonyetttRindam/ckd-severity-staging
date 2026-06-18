# Raw Data

Raw data files are **not committed** to this repository.

## Dataset: NHANES (National Health and Nutrition Examination Survey)

This project uses NHANES data from the CDC. The data must be downloaded manually
and placed in this directory before running any notebooks or scripts.

## Download Instructions

1. Go to the [NHANES data portal](https://wwwn.cdc.gov/nchs/nhanes/)
2. Select the survey cycle(s) relevant to this project (e.g. 2017–2020 Pre-Pandemic)
3. Download the following data components (XPT format):
   - **Demographics** — `DEMO_*.XPT`
   - **Laboratory** — `BIOPRO_*.XPT` (creatinine, albumin), `ALB_CR_*.XPT`
   - **Questionnaire** — `KIQ_U_*.XPT` (kidney conditions)
   - **Examination** — `BMX_*.XPT` (body measures)
4. Place all `.XPT` files in this `data/raw/` directory

## Alternative: CDC FTP

Files can also be bulk-downloaded from the CDC FTP:
```
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/
```

## Expected Files After Download

```
data/raw/
├── DEMO_J.XPT       # Demographics 2017-2018
├── BIOPRO_J.XPT     # Biochemistry 2017-2018
├── KIQ_U_J.XPT      # Kidney conditions 2017-2018
├── BMX_J.XPT        # Body measures 2017-2018
└── README.md        # This file
```

> Note: File suffixes vary by cycle (J = 2017–2018, P = 2017–2020 pre-pandemic).
> Check the NHANES website for the exact filenames for your chosen cycle.
