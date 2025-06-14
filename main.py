import numpy as np
import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pyswarms.single.general_optimizer import GeneralOptimizerPSO
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.backend.topology import Star

from pyriemann.estimation import Covariances
from pso_utils import apply_mask_to_spd, threshold_mask, evaluate_fitness
from GLRSQ import GLRSQ
from dataloader_dssr import load_dataset
from joblib import Parallel, delayed


def run_dssr(dataset_name="BNCI2014_001", fitness_mode="GLRSQ", classifier_mode="GLRSQ", verbose=True, save_suffix=None):
    warnings.filterwarnings("ignore")

    THRESHOLD = 0.6    #바꿔봄직함
    ALPHA = 0.001        #바꿔봄직함
    N_PARTICLES = 20   #바꿔봄직함
    MAX_ITERS = 30
    PATIENCE = 5

    X, y, meta = load_dataset(dataset_name)

    le = LabelEncoder()
    all_results = []

    if meta is not None:
        subjects = meta.subject.unique()
    else:
        subjects = [0]

    mask_dir = f"masks/{dataset_name}/DSSR_{fitness_mode[0]}{classifier_mode[0]}"
    os.makedirs(mask_dir, exist_ok=True)

    for subject in subjects:
        sess = meta[meta.subject == subject].session.unique()
        idx = meta[meta.subject == subject].index
        rng = np.random.default_rng(seed=subject)  # reproducible per subject
        idx = rng.permutation(idx)
        n = len(idx)
        split = int(n * 2 / 3)
        train_idx = idx[:split]
        test_idx = idx[split:]

        X_train_raw = X[train_idx]
        y_train = le.fit_transform(y[train_idx])
        X_val_raw = X[test_idx]
        y_val = le.transform(y[test_idx])

        cov_dir = f"cached_covs/{dataset_name}"
        os.makedirs(cov_dir, exist_ok=True)

        subject_id = f"sub{subject}" if meta is not None else dataset_name
        cov_train_path = os.path.join(cov_dir, f"{subject_id}_train_cov.npy")
        cov_val_path = os.path.join(cov_dir, f"{subject_id}_val_cov.npy")

        if os.path.exists(cov_train_path) and os.path.exists(cov_val_path):
            X_train = np.load(cov_train_path)
            X_val = np.load(cov_val_path)
            print(f"Loaded precomputed covariances for {subject_id}")
        else:
            cov = Covariances(estimator='oas')
            X_train = cov.fit_transform(X_train_raw)
            X_val = cov.transform(X_val_raw)
            np.save(cov_train_path, X_train)
            np.save(cov_val_path, X_val)
            print(f"Saved computed covariances for {subject_id}")

        CHANNEL_DIM = X_train.shape[-1]
        best_costs = []
        best_cost = np.inf
        no_improve = 0

#objective는 fitness를 한번 찾음.
        def objective(particles):
            nonlocal best_cost, no_improve
            fitnesses = []

            for i in range(particles.shape[0]):
                particle = particles[i]
                mask = threshold_mask(particle, threshold=THRESHOLD)
                dr = int(np.sum(mask))
                if dr < 2:
                    fitnesses.append(1.0)
                    continue

                X_train_masked = apply_mask_to_spd(X_train, mask)
                X_val_masked = apply_mask_to_spd(X_val, mask)

                fitness = evaluate_fitness(
                    X_train_masked, y_train,
                    X_val_masked, y_val,
                    original_dim=CHANNEL_DIM,
                    selected_dim=dr,
                    alpha=ALPHA
                )
                fitnesses.append(-fitness)  # PSO는 최소화이므로 부호 반전

            current_best = min(fitnesses)
            best_costs.append(current_best)
            if current_best < best_cost:
                best_cost = current_best
                no_improve = 0
            else:
                no_improve += 1

            print(f"Iteration {len(best_costs)} - Best fitness: {-current_best:.4f}")

            if no_improve >= PATIENCE:
                raise StopIteration("No improvement in best fitness")

            return np.array(fitnesses)
################################################################################################################################################################################################################

        options = {'c1': 1.5, 'c2': 1.5, 'w': 0.73}
######################################################################
        # optimizer = GeneralOptimizerPSO(
        #     n_particles=N_PARTICLES,
        #     dimensions=CHANNEL_DIM,
        #     options=options,
        #     velocity_clamp=(-0.1, 0.1),
        #     topology=Star()
        # )

        optimizer = GlobalBestPSO(
            n_particles=N_PARTICLES,
            dimensions=CHANNEL_DIM,
            options=options,
            velocity_clamp=(-0.1, 0.1),
        )
        #################################################
        subject_id = f"sub{subject}" if meta is not None else dataset_name

        try:
            best_cost, best_position = optimizer.optimize(objective, iters=MAX_ITERS, verbose=False)
        except StopIteration:
            best_position = optimizer.swarm.best_pos
            print(f"[{dataset_name}][{subject_id}] early stopped after {len(best_costs)} iterations")

        best_mask = threshold_mask(best_position, threshold=THRESHOLD)
        np.save(os.path.join(mask_dir, f"best_mask_{subject_id}.npy"), best_mask)
        ##########################이 및으로는 손댈거 없음########################################################이 및으로는 손댈거 없음##############################




        plt.plot([-c for c in best_costs])
        plt.xlabel("Iteration")
        plt.ylabel("Best fitness")
        plt.title(f"{dataset_name} {subject_id} DSSR-{fitness_mode[0]}{classifier_mode[0]}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(mask_dir, f"fitness_curve_{subject_id}.png"))
        plt.close()
        X_train_final = apply_mask_to_spd(X_train, best_mask)
        X_val_final = apply_mask_to_spd(X_val, best_mask)


        clf = GLRSQ(n_classes=4, max_iter=50, learning_rate=0.1)
        clf.fit(X_train_final, y_train)
        y_pred = clf.predict(X_val_final)


        acc = accuracy_score(y_val, y_pred)
        kappa = cohen_kappa_score(y_val, y_pred)
        dssr_name = f"DSSR-{fitness_mode[0]}{classifier_mode[0]}"
        if verbose:
            print(f"[{dssr_name}][{subject}] acc: {acc:.4f}, kappa: {kappa:.4f}, dim: {np.mean(best_mask):.2f}")
        all_results.append({
            "subject": subject_id,
            "accuracy": acc,
            "kappa": kappa,
            "mask_ratio": np.mean(best_mask),
            "dssr": dssr_name
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"dssr_{save_suffix or fitness_mode[0]+classifier_mode[0]}_{dataset_name}.csv", index=False)

    # Summary print
    acc_mean, acc_std = results_df['accuracy'].mean(), results_df['accuracy'].std()
    kappa_mean, kappa_std = results_df['kappa'].mean(), results_df['kappa'].std()
    dim_mean, dim_std = results_df['mask_ratio'].mean(), results_df['mask_ratio'].std()
    print("\n--- Summary ---")
    print(f"{dssr_name} on {dataset_name}:")
    print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"Kappa:    {kappa_mean:.4f} ± {kappa_std:.4f}")
    print(f"MaskRatio:{dim_mean:.4f} ± {dim_std:.4f}")


if __name__ == "__main__":
    dataset = "BNCI2014_001"
    fitness_mode = "GLRSQ"
    classifier_mode = "GLRSQ"
    suffix = f"{fitness_mode[0]}{classifier_mode[0]}"

    print(f"Running {dataset} - DSSR-{suffix}...")
    run_dssr(
        dataset_name=dataset,
        fitness_mode=fitness_mode,
        classifier_mode=classifier_mode,
        save_suffix=suffix
    )