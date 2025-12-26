'''
Spatial-temporal benchmark using wrapped original scenarios
'''
import numpy as np
import sys
import time
from pathlib import Path

from funcs_st import STScenario1, STScenario2, STScenario3, STScenario4, STScenario5, STScenario6, STScenario7, STScenario8
from neural_sqerr import SqErrNetwork
from neural_model import QuantileNetwork
from visualize import heatmap_from_points
from logger import DualLogger

def run_st_benchmarks(demo=True, scenarios=None):
    N_trials = 1 if demo else 100
    N_test = 10000
    sample_sizes = [1000, 5000, 10000]
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    
    all_functions = [STScenario1(), STScenario2(), STScenario3(), STScenario4(), STScenario5(),
                     STScenario6(), STScenario7(), STScenario8()]
    
    # Filter scenarios if specified
    if scenarios:
        functions = [all_functions[i-1] for i in scenarios if 1 <= i <= len(all_functions)]
        scenario_indices = [i-1 for i in scenarios if 1 <= i <= len(all_functions)]
    else:
        functions = all_functions
        scenario_indices = list(range(len(all_functions)))
    
    models = [lambda: SqErrNetwork(),
              lambda: QuantileNetwork(quantiles=quantiles)]

    mse_results = np.full((N_trials, len(all_functions), len(models), len(sample_sizes), len(quantiles)), np.nan)
    print(f'Results shape: {mse_results.shape}')
    print(f'Running scenarios: {[i+1 for i in scenario_indices]}')

    Path('plots').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    for trial in range(N_trials):
        print(f'Trial {trial+1}/{N_trials}')
        for idx, (scenario_idx, func) in enumerate(zip(scenario_indices, functions)):
            print(f'  Scenario {scenario_idx+1}: {func.label}')
            if scenario_idx + 1 in [1, 2, 3, 4, 5]:
                X_test = np.random.random(size=(N_test, func.n_in))
            else:
                X_test = np.random.uniform(0, 1, (N_test, func.M, func.d))
            y_test = func.sample(X_test)
            y_quantiles = np.array([func.quantile(X_test, q) for q in quantiles]).T

            if demo and idx == 0:
                for qidx, q in enumerate((quantiles*100).astype(int)):
                    heatmap_from_points(f'plots/st_scenario{scenario+1}-quantile{q}-truth.pdf', 
                                      X_test[:,:2], y_quantiles[:,qidx], 
                                      vmin=y_quantiles.min(), vmax=y_quantiles.max())

            for nidx, N_train in enumerate(sample_sizes):
                print(f'    N={N_train}')
                if scenario_idx + 1 in [1, 2, 3, 4, 5]:
                    X_train = np.random.random(size=(N_train, func.n_in))
                else:
                    X_train = np.random.uniform(0, 1, (N_train, func.M, func.d))
                y_train = func.sample(X_train)

                for midx, model in enumerate([m() for m in models]):
                    print(f'      {model.label}')

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    mse_results[trial, scenario_idx, midx, nidx] = ((y_quantiles - preds)**2).mean(axis=0)

                    if demo and idx == 0 and nidx == 0:
                        for qidx, q in enumerate((quantiles*100).astype(int)):
                            heatmap_from_points(f'plots/st_scenario{scenario_idx+1}-quantile{q}-n{N_train}-{model.filename}.pdf',
                                              X_test[:,:2],
                                              preds[:,qidx] if preds.shape[1] > qidx else preds[:,-1],
                                              vmin=y_quantiles.min(), vmax=y_quantiles.max(),
                                              colorbar=midx == len(models)-1)

            print(f'  Results: {mse_results[trial, scenario_idx]}')

        if not demo:
            np.save('results/st_mse_results.npy', mse_results)

    print('\nFinal MSE Results (mean across trials):')
    mean_mse = np.nanmean(mse_results, axis=0)
    for scenario in range(len(functions)):
        print(f'\nScenario {scenario+1}:')
        for midx, model in enumerate([m() for m in models]):
            print(f'  {model.label}:')
            for nidx, N_train in enumerate(sample_sizes):
                print(f'    N={N_train}: {mean_mse[scenario, midx, nidx]}')

    return mse_results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ST Quantile regression benchmarks')
    parser.add_argument('--demo', action='store_true', default=True, help='Run in demo mode')
    parser.add_argument('--full', action='store_true', help='Run full experiment')
    parser.add_argument('--scenarios', type=int, nargs='+', help='Scenarios to run (e.g., --scenarios 6 7 8)')
    args = parser.parse_args()
    
    demo_mode = not args.full
    
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    np.set_printoptions(precision=3, suppress=True)
    
    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    mode_str = 'demo' if demo_mode else 'full'
    Path('logs').mkdir(exist_ok=True)
    log_file = Path('logs') / f'benchmark_st_wrapped_{mode_str}_{current_time}.log'
    
    logger = DualLogger(log_file)
    sys.stdout = logger
    
    print(f"{'='*80}")
    print(f"Spatial-Temporal Quantile Regression (Wrapped Scenarios)")
    print(f"{'='*80}")
    print(f"Mode: {'Demo (1 trial)' if demo_mode else 'Full (100 trials)'}")
    print(f"Time: {current_time}")
    print(f"Log file: {log_file}")
    print(f"{'='*80}\n")
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_time = time.time()
        run_st_benchmarks(demo=demo_mode, scenarios=args.scenarios)
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print('✅ Completed!')
        print(f'📊 Check plots/ directory')
        print(f'⏱️  Total time: {elapsed_time:.2f}s')
        print(f'📝 Log saved to: {log_file}')
        print(f"{'='*80}")
    
    logger.close()
