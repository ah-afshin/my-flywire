import pandas as pd
import yaml
from brian2 import NeuronGroup, Synapses, SpikeMonitor, Network, PoissonInput
from brian2 import mV, ms, Hz, store, restore


class SSN:
    def __init__(self, config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        self.config = config

        comp_df = pd.read_csv(config['general']['comp-path'], index_col=0)
        conn_df = pd.read_parquet(config['general']['conn-path'])
        
        self.G = NeuronGroup(
            name=config['general']['name'],
            N=len(comp_df),
            model=config['LIF-params']['eqs'],
            method=config['LIF-params']['method'],
            threshold=config['LIF-params']['threshold'],
            reset=config['LIF-params']['reset'],
            refractory='rfc',
            namespace={
                'v_0': config['LIF-params']['v_0_mV'] * mV,
                'v_reset': config['LIF-params']['v_reset_mV'] * mV,
                'v_threshold': config['LIF-params']['v_threshold_mV'] * mV,
                'tau_mbr': config['LIF-params']['tau_mbr_ms'] * ms,
                'tau_syn': config['LIF-params']['tau_syn_ms'] * ms
            }
        )
        self.G.v = config['LIF-params']['v_0_mV'] * mV
        self.G.g = 0 * mV
        self.G.rfc = config['LIF-params']['t_refractory_ms'] * ms

        self.S = Synapses(
            self.G,
            self.G,
            'w : volt',
            on_pre='g += w',
            delay=config['LIF-params']['t_delay_ms'] * ms,
            name='loaded_synapses'
        )
        idx_pre = conn_df.loc[:, 'Presynaptic_Index'].values
        idx_post = conn_df.loc[:, 'Postsynaptic_Index'].values
        self.S.connect(i=idx_pre, j=idx_post)
        self.S.w = (
            conn_df.loc[:,'Excitatory x Connectivity'].values
            * config['LIF-params']['w_syn_mV'] * mV
        )

        self.W_0 = self.S.w[:].copy()
        self.P = []
        self.trial_num = 0

        self.M = SpikeMonitor(self.G)
        self.N: Network = None

    def define_poisson(self, exc_idx:list[int], r_poi, w_poi):
        for i in exc_idx:
            p = PoissonInput(
                target=self.G[i], 
                target_var='v', 
                N=1, 
                rate=r_poi, 
                weight= w_poi # w_syn*f_poi
            )
            self.G[i].rfc = 0 * ms
            self.P.append(p)

    def silence_neuron(self, neu_idx):
        self.S.w = self.W_0

        for i in neu_idx:
            self.S.w[f'i == {i}'] = 0*mV

    def restore(self):
        self.N.restore(
            name=f'trial_{self.trial_num}',
            restore_random_state=False
        )
        self.S.w = self.W_0
        self.G.rfc = self.config['LIF-params']['t_refractory_ms'] * ms
        self.P = []
        self.trial_num += 1

    def __call__(self):
        self.N = Network(self.G, self.S, self.M, *self.P)
        self.N.store(f'trial_{self.trial_num}')
        self.N.run(duration=self.config['general']['t_run'])
        return {k: v for k, v in self.M.spike_trains().items() if len(v)}
