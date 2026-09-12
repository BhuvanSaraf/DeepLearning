# CS601T Assignment 3 — FCNN Optimizer Comparison (PyTorch)

**Group number: 06**

## What this does

Trains an FCNN (3, 4, and 5 hidden layers) on the Group 06 MNIST
5-class subset ({0,3,4,8,9}) using all 7 optimizers required by the
assignment, with the exact hyperparameters specified in the PDF
(lr=0.001 throughout; momentum=0.9 for generalized delta/NAG;
RMSProp beta=0.99, eps=1e-8; Adam betas=(0.9,0.999), eps=1e-8).
Cross-entropy loss throughout. Uses PyTorch (allowed for this
assignment, unlike Assignments 1-2).

## Files

| File | Purpose |
|---|---|
| `data_utils.py` | Loads the 28x28 JPEGs from `data/{train,val,test}/{0,3,4,8,9}/`, flattens to 784-dim, normalizes to [0,1]. |
| `model.py` | Configurable `FCNN` (3-5 hidden layers, ReLU hidden activations, raw logits output for use with `nn.CrossEntropyLoss`). |
| `optimizer_configs.py` | The 7 optimizer configs (exact hyperparameters) and the 3 architectures searched. |
| `train_one.py` | Trains ONE (architecture, optimizer) combination. **Resumable/checkpointed** every epoch, so a timeout doesn't lose progress -- rerun the same command to continue. Implements the assignment's literal stopping criterion: `|avg_error(epoch) - avg_error(epoch-1)| < 1e-4`, no patience/smoothing. Same initial weights are reused across all 7 optimizers for a given architecture (assignment requirement (d)), saved once per architecture under `results/init_weights/`. |
| `aggregate_results.py` | Loads all 21 result files, prints the epochs-to-convergence table and the train/val accuracy table, plots superimposed error-vs-epoch curves (one figure per architecture, all 7 optimizers overlaid, log-scale y-axis), picks the best (architecture, optimizer) combo by validation accuracy, and reports its train/test confusion matrices and accuracy. |

## How to run

```bash
# Train every (architecture, optimizer) combination. Each call is safe
# to interrupt and rerun -- already-finished combos are skipped, and
# in-progress ones resume from their last checkpointed epoch.
for arch in arch3 arch4 arch5; do
  for opt in sgd batch_gd sgd_momentum sgd_nag adagrad rmsprop adam; do
    python3 train_one.py $arch $opt --max_epochs 1500
  done
done

# Once all 21 results/*.json files exist:
python3 aggregate_results.py
```

Total training time across all 21 combinations was roughly 40-50
minutes on CPU in our runs.

## Important findings for the report

1. **Vanilla Batch GD "false-converges" almost immediately.** With a
   single full-batch gradient step per epoch and lr=0.001, the
   loss barely moves between epoch 0 and 1 (e.g. 1.6118 -> 1.6118),
   which satisfies `|diff| < 1e-4` after just 1-2 epochs -- while the
   model is still at ~20% accuracy (random guessing among 5 classes).
   This happens consistently across all three architectures. It's a
   direct, real consequence of the assignment's literal stopping
   criterion applied to an optimizer whose per-epoch progress is
   naturally tiny, not a bug in the code.

2. **AdaGrad and RMSProp (also batch_size=N) avoid this trap** because
   their per-parameter adaptive scaling amplifies small raw gradients
   into meaningful updates, so they still learn substantially (94-96%
   accuracy) before their loss curve genuinely flattens.

3. **Adam at batch_size=1 has noisy epoch-average loss.** With true
   per-sample updates, Adam's epoch-level loss oscillates (does not
   decrease smoothly) even well after the model has essentially
   converged, similar to what we saw with pure SGD in Assignment 2 on
   a larger dataset. For `arch3` and `arch5`, Adam's runs are marked
   `hit_max_epochs: true` in the result JSON and include a `note`
   field -- we capped training at a practical epoch budget once the
   loss had clearly plateaued (just noisily), rather than waiting
   indefinitely for two consecutive epochs to happen to land within
   1e-4 of each other by chance. Final accuracy is still excellent
   (96-98%) at the point training was capped.

4. **Adam is markedly slower per epoch than the other batch_size=1
   optimizers** (roughly 5x the wall-clock time of plain SGD per
   epoch on this hardware), due to the extra per-parameter moment
   estimates and bias-correction arithmetic in each `optimizer.step()`
   call, which matters a lot at batch_size=1 with over 11,000 steps
   per epoch.

5. **The best (architecture, optimizer) combo by validation accuracy**
   was `arch4` (4 hidden layers: 256-128-64-32) with **SGD + Momentum**,
   at 98.66% validation accuracy, 98.23% test accuracy. See
   `results/summary/best_combo_summary.json` for the full confusion
   matrices.

## Before submitting

- Delete `data/` and `results/` (except perhaps `results/summary/` if
  you want to keep the final tables/plots for reference) from the code
  zip -- the instructions ask for code only.
- Name the zip `Group06_Assignment3_code.zip`, containing a folder
  named `Group06_Assignment3_code`.
- Name the report `Group06_Assignment3_report.pdf`.


## Bhuvan's Comments:

Within Group06_Assignment3_code
- python3 -m venv .venv
- source .venv/bin/activate
- pip3 install -r requirements.txt