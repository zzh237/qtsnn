'''
Quantile regression benchmarks for spatial-temporal data using neural networks.
'''
import numpy as np
from spatial_temporal_scenarios import STScenario1, STScenario2, STScenario3
from neural_model import QuantileNetwork
from neural_sqerr import SqErrNetwork
from visualize import heatmap_from_points
import os

def run_st_benchmarks(demo=True):
    N_trials = 100 if not demo else 1
    N_test = 10000
    sample_sizes = [1000, 5000, 10000]
    quantiles = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    
    functions = [
        STScenario1(n=200, m_mult=1, d=2, tau=0.5),
        STScenario2(n=200, m_mult=1, d=2, tau=0.5),
        STScenario3(n=200, m_mult=1, d=2, tau=0.5)
    ]
    
    models = [
        lambda: SqErrNetwork(),
        lambda: QuantileNetwork(quantiles=quantiles)
    ]
    
    mse_results = np.full((N_trials, len(functions), len(models), len(sample_sizes), len(quantiles)), np.nan)
    print(f'Results shape: {mse_results.shape}')
    
    os.makedirs('plots', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    for trial in range(N_trials):
        print(f'Trial {trial+1}/{N_trials}')
        
        for scenario, func in enumerate(functions):
            print(f'  Scenario {scenario+1}: {func.label}')
            
            X_test = np.random.random(size=(N_test, func.n_in))
            y_test = func.sample(X_test)
            y_quantiles = np.array([func.quantile(X_test, q) for q in quantiles]).T
            
            if demo and scenario == 0:
                for qidx, q in enumerate((quantiles*100).astype(int)):
                    heatmap_from_points(
                        f'plots/st_scenario{scenario+1}-quantile{q}-truth.pdf',
                        X_test[:,:2], y_quantiles[:,qidx],
                        vmin=y_quantiles.min(), vmax=y_quantiles.max()
                    )
            
            for nidx, N_train in enumerate(sample_sizes):
                print(f'    N={N_train}')
                
                X_train = np.random.random(size=(N_train, func.n_in))
                y_train = func.sample(X_train)
                
                for midx, model in enumerate([m() for m in models]):
                    print(f'      {model.label}')
                    
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    
                    mse_results[trial, scenario, midx, nidx] = ((y_quantiles - preds)**2).mean(axis=0)
                    
                    if demo and scenario == 0 and nidx == 0:
                        for qidx, q in enumerate((quantiles*100).astype(int)):
                            heatmap_from_points(
                                f'plots/st_scenario{scenario+1}-quantile{q}-n{N_train}-{model.filename}.pdf',
                                X_test[:,:2],
                                preds[:,qidx] if preds.shape[1] > qidx else preds[:,-1],
                                vmin=y_quantiles.min(), vmax=y_quantiles.max(),
                                colorbar=midx == len(models)-1
                            )
            
            print(f'  Results: {mse_results[trial, scenario]}')
        
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
    
    parser = argparse.ArgumentParser(description='Quantile regression benchmarks for spatial-temporal data')
    parser.add_argument('--demo', action='store_true', default=True, help='Run in demo mode (1 trial, default: True)')
    parser.add_argument('--full', action='store_true', help='Run full experiment (100 trials)')
    args = parser.parse_args()
    
    # If --full is specified, override demo
    demo_mode = not args.full
    
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    np.set_printoptions(precision=3, suppress=True)
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = run_st_benchmarks(demo=demo_mode)
        if demo_mode:
            print('\nDemo completed! Check plots/ and results/ directories.')
        else:
            print('\nFull experiment completed! Check results/st_mse_results.npy')
