"""Runs all tasks in order (each task depends on outputs of earlier ones), then writes results/summary.md."""
import runpy
for s in ["task1_pca", "task2_autoencoders", "task3_ae1_classify", "task4_ae3_classify",
          "task5_denoising", "task6_weights", "summarize"]:
    print(f"\n===== {s} =====")
    runpy.run_path(f"{s}.py", run_name="__main__")
