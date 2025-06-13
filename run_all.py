from main import run_dssr

datasets = ["BNCI2014_001"]
combinations = [
    ("GLRSQ", "GLRSQ") # DSSR-GG
]

for dataset in datasets:
    for fit_mode, clf_mode in combinations:
        suffix = f"{fit_mode[0]}{clf_mode[0]}"
        print(f"Running {dataset} - DSSR-{suffix}...")
        run_dssr(
            dataset_name=dataset,
            fitness_mode=fit_mode,
            classifier_mode=clf_mode,
            save_suffix=suffix
        )