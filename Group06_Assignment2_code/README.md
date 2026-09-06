# CS601T Assignment 2 — FCNN from Scratch

**Group number: 06**

## What this does

Implements a fully-connected neural network (FCNN) completely from
scratch (NumPy only -- no TensorFlow/PyTorch/sklearn), trained with
true stochastic gradient descent (`batch_size=1`) and **squared error**
loss (per the assignment spec), for both classification and
regression, and compares results against the Assignment 1
single-neuron perceptron baseline.

## Files

| File | Purpose |
|---|---|
| `NeuralNetwork.py` | The FCNN itself: forward pass, backprop, SGD weight updates. Supports logistic/tanh/relu hidden activations. Classification uses one-hot targets + squared error (same bounded activation at the output layer, NOT softmax/cross-entropy). Regression uses a linear output. |
| `perceptron_baseline.py` | Copy of the Assignment 1 single-neuron perceptron, used only to generate the required "compare with Assignment 1" numbers. |
| `data_utils.py` | 60/20/20 stratified train/val/test split (`train_val_test_split_per_class`) plus the one-against-one helpers reused for the baseline comparison. |
| `metrics.py` | Confusion matrix, accuracy, per-class + macro/micro precision/recall/F1, RMSE/%RMSE. (Same as Assignment 1.) |
| `plotting.py` | Error-vs-epoch, decision regions, regression fit/scatter plots (same as Assignment 1), plus the new `plot_node_outputs()` for the hidden/output node activation plots required by Assignment 2. |
| `classification_main.py` | Full classification pipeline: 60/20/20 split, architecture search (1 hidden layer for Dataset 1, 2 hidden layers for Dataset 2, several hidden-node counts x 2 activations), full metrics on the validation set for every architecture, then error curve / decision region / node plots / test metrics for the best architecture, plus the Assignment 1 comparison. |
| `regression_main.py` | Full regression pipeline: 60/20/20 split, architecture search (1 hidden layer for Dataset 1; BOTH 1 and 2 hidden layers for Dataset 2), RMSE/%RMSE on train+val for every architecture, then error curve / fit plots / scatter plots / node plots / test RMSE for the best architecture, plus the Assignment 1 comparison. |

## Data folder setup

Same as Assignment 1 -- place the Group06 dataset under:

```text
Group06_Assignment2_code/
├── data/
│   └── Group06/
│       ├── Classification/
│       │   ├── NLS_Group06.txt
│       │   └── LS_Group06/
│       │       ├── Class1.txt
│       │       ├── Class2.txt
│       │       └── Class3.txt
│       └── Regression/
│           ├── BivariateData/6.csv
│           └── UnivariateData/6.csv
```

## How to run

```bash
python3 classification_main.py     # ~1.5-2 min
python3 regression_main.py         # ~2-4 min (dataset2 has 10,201 points)
```

Plots are saved under `results/classification/` and
`results/regression/`. All console output (confusion matrices,
per-architecture validation metrics, final test metrics, and the
Assignment 1 comparison) prints directly -- redirect to a file if you
want to keep a full log for the report, e.g.:

```bash
python3 classification_main.py | tee classification_log.txt
python3 regression_main.py | tee regression_log.txt
```

## Design choices worth knowing about for the report

- **Squared error loss for classification.** The assignment explicitly
  requires squared error (not cross-entropy). This means the output
  layer uses the same bounded activation (logistic or tanh) as the
  hidden layers, applied elementwise to each of the one-hot output
  nodes, rather than softmax. Predicted class = `argmax` over the
  output vector.
- **Architecture search is a 2-value grid per dataset** (hidden layer
  counts fixed per the spec, node counts and activation varied) to
  keep runtime reasonable before the deadline. Feel free to widen the
  `hidden_options` lists in `main()` in either script if you have more
  time before submitting -- everything else (metrics, plots) will
  regenerate automatically for whatever the new best architecture is.
- **Regression Dataset 2 uses a shorter "search" epoch budget** (60
  epochs) before retraining the winning architecture for longer (300
  epochs), purely because it has 10,201 points and batch_size=1 SGD
  makes a full epoch relatively expensive. If you want the search
  numbers themselves to be more reliable, raise `search_epochs` in
  `regression_main.py`'s `main()` at the cost of runtime.
- **Comparison with Assignment 1** retrains the Assignment 1 perceptron
  on the *same* 60% training split used here (not the original 70/30
  split) so the comparison to the FCNN's test-set performance is
  apples-to-apples.

## Before submitting

- Delete `data/` and `results/` from the code zip -- the instructions
  ask for code only.
- Name the zip `Group06_Assignment2_code.zip`, containing a folder
  named `Group06_Assignment2_code`.
- Name the report `Group06_Assignment2_report.pdf`.
