# CS601T Assignment 1 — Perceptron from Scratch

**Group number: 06**

## Status

Fully implemented and tested end-to-end on the **real Group06 dataset**
(`data/Group06/...`). Both `classification_main.py` and
`regression_main.py` load directly from the files you were given:

- Classification Dataset 1 (linearly separable): `data/Group06/Classification/LS_Group06/Class{1,2,3}.txt`
- Classification Dataset 2 (nonlinearly separable): `data/Group06/Classification/NLS_Group06.txt`
  (first 500 rows = class1, next 500 = class2, last 1000 = class3, per the file's own header line)
- Regression Dataset 1 (univariate): `data/Group06/Regression/UnivariateData/6.csv`
- Regression Dataset 2 (bivariate): `data/Group06/Regression/BivariateData/6.csv`

### Notable results already observed (worth writing up as inferences)

- **LS classification**: near-perfect (96–100% test accuracy) with both
  logistic and tanh — the three classes are cleanly linearly separable blobs.
- **NLS classification**: Class 3 forms a ring that fully **surrounds**
  Classes 1 and 2 (two spiral arcs). No straight line can isolate a ring
  from what's inside it, so the 1-vs-3 and 2-vs-3 pairwise classifiers
  perform only slightly better than chance, and the combined
  one-against-one vote is heavily biased toward Class 3. This is a clean,
  concrete illustration of why single-layer (linear) perceptrons fail on
  nonlinearly separable data — worth a paragraph in your report.
- **Regression**: both the univariate and bivariate targets are visibly
  **curved/nonlinear** functions of the input, so the linear-activation
  perceptron systematically over/under-shoots at the extremes even after
  convergence (%RMSE ~6–7%). Also worth discussing as a limitation of a
  linear model on nonlinear data.

## Files

| File | Purpose |
|---|---|
| `perceptron.py` | From-scratch `Perceptron` class: logistic / tanh / linear activations, batch gradient descent. No ML libraries used. |
| `data_utils.py` | Per-class 70/30 train-test split, one-against-one pair generation, majority-vote aggregation for multiclass. |
| `metrics.py` | From-scratch confusion matrix, accuracy, RMSE, %RMSE. |
| `plotting.py` | All required plots: error-vs-epoch, decision regions (pairwise + combined), regression fit plots (1D/2D), target-vs-model scatter plots. |
| `classification_main.py` | Full classification pipeline (Datasets 1 & 2, one-against-one, logistic + tanh). |
| `regression_main.py` | Full regression pipeline (Datasets 1 & 2, linear activation). |

## Remaining steps before submission

1. Inspect all plots in `results/classification/` and `results/regression/`
   and pick which ones go into the report.
2. Write the report (PDF) with inferences on:
   - How logistic vs. tanh activations compare (convergence speed, final accuracy)
   - Why Class 3 in the NLS dataset (a ring around the other two classes)
     defeats a linear one-against-one classifier
   - Train vs. test RMSE/%RMSE trends for the regression tasks, and why
     a linear model can't fully capture the visibly curved target functions
3. **Before zipping for Moodle**: delete the `data/` folder and the
   `results/` folder from the code zip — the instructions ask for code
   only. Keep them locally for your own re-runs / report screenshots.
4. Name the report `Group06_Assignment1_report.pdf`, the code folder
   `Group06_Assignment1_code`, and the zip `Group06_Assignment1_code.zip`.

## How to run

```bash
python3 classification_main.py
python3 regression_main.py
```

Plots are saved under `results/classification/` and
`results/regression/`. Each module also has a small self-test at the
bottom (`if __name__ == "__main__":`) you can run individually, e.g.
`python3 perceptron.py`.

## Constraint compliance

No perceptron / neural-network / gradient-descent libraries (e.g.
scikit-learn, PyTorch, TensorFlow) are used anywhere. Only NumPy
(array math) and Matplotlib (plotting) are used as general-purpose
numerical/plotting tools — the perceptron model, activation functions,
gradient computation, and weight updates are all written from scratch
in `perceptron.py`.
