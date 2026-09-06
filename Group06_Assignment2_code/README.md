# AI Lab Assignment 1

## Data folder setup

Place the dataset files in the following structure under the project root:

```text
Group06_Assignment1_code/
├── classification_main.py
├── regression_main.py
├── data/
│   └── Group06/
│       ├── Classification/
│       │   ├── NLS_Group06.txt
│       │   └── LS_Group06/
│       │       ├── Class1.txt
│       │       ├── Class2.txt
│       │       └── Class3.txt
│       └── Regression/
│           ├── BivariateData/
│           │   └── 6.csv
│           └── UnivariateData/
│               └── 6.csv
└── ...
```

Important:
- The project expects the data directory to be named `data` at the project root.
- Inside it, the folder must be named `Group06`.
- The classification files must be under `data/Group06/Classification/`.
- The regression files must be under `data/Group06/Regression/`.

If these folders are missing or named differently, the scripts will not find the data and will fail with file-not-found errors.

## Run the scripts

From the project root:

```bash
python3 classification_main.py
python3 regression_main.py
```
