import yaml
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from brian2 import Hz, mV, ms
from model import SSN


n_run = 10
sugar_neurons = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]
slnc_neurons = []
flyid2name = {f: f'sugar_{i+1}' for i, f in enumerate(sugar_neurons)}

def run_experiment():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)
    print('config loaded.')
    
    output_dir = Path(config['general']['output-dir']) / config["general"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet'

    df_comp = pd.read_csv(config['general']['comp-path'], index_col=0)
    flyid2i = {int(j): i for i, j in enumerate(df_comp.index)}  # flywire id: biran ID
    i2flyid = {i: fly_id for fly_id, i in flyid2i.items()}      # brian ID: flywire ID

    missing_neurons = [n for n in sugar_neurons + slnc_neurons if n not in flyid2i]
    if missing_neurons:
        raise ValueError(f'These FlyWire IDs are missing from comp_df: {missing_neurons}')
    print('excitatory neurons checked.')
    
    snn = SSN(config_path='./config.yaml')
    print('model created.')
    
    exc_idx = [flyid2i[n] for n in sugar_neurons]
    slnc_idx = [flyid2i[n] for n in slnc_neurons]
    # exc2 = [flyid2i[n] for n in exc_idx2]
    
    snn.define_poisson(
            exc_idx=exc_idx,
            r_poi= 150 * Hz,
            w_poi= 68.47 * mV   # w_syn*f_poi
    )

    snn.store_baseline()
    results = []
    for _ in tqdm(range(n_run), desc='running trials.'):
        snn.silence_neuron(slnc_idx)
        res = snn()
        results.append(res)
        snn.restore()
    print('done.')

    rows = []
    for trial_idx, res in enumerate(results):
        for neuron_idx, spike_times in res.items():
            for t in spike_times:
                rows.append({
                    't': float(t),
                    'trial': trial_idx,
                    'flywire_id': i2flyid[int(neuron_idx)],
                    'exp_name': config['general']['name']
                })
    results_df = pd.DataFrame(rows, columns=['t', 'trial', 'flywire_id', 'exp_name'])
    results_df.to_parquet(output_path, compression='brotli', index=False)
    print(f'experiment saved to {output_path}')


if __name__ == "__main__":
    run_experiment()
