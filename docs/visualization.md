<!-- # Visualization

We provide the script to visualize the FreeAD prediction to a video [here](../tools/analysis_tools/visualization.py). -->

## Visualize prediction

```shell
cd /path/to/FreeAD/
conda activate freead
sh vis.sh
```

The inference results is a prefix_results_nusc.pkl automaticly saved to the work_dir after running evaluation. It's a list of prediction results for each validation sample.
