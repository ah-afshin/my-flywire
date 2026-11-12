# My FlyWire

An independent implementation of a FlyWire-based spiking neural network, developed to reproduce and explore the experiments described in:

**Shiu et al. (2024).** *A Drosophila computational brain model reveals sensorimotor processing.* Nature.

## Project status

**Discontinued.** The original goal was to understand and reproduce the computational model of the fruit fly brain, then connect it to a game environment and train it to learn and play the game.

That broader goal was never completed. The current repository contains an implementation of the model and a reproducible attempt to reproduce the paper's sugarR experiment.

## Implementation

The project is intentionally kept simple and consists of two main components:

- `model.py` - A reusable Brian2-based implementation of the FlyWire spiking neural network.
- `sugarr_experiment.py` - Runs the sugarR stimulation experiment described by Shiu et al.

Experiment-specific parameters, including the stimulated and silenced neuron IDs and Poisson input parameters, are kept in the experiment script to reflect the reference experiment. The model uses Brian2's conventional notation for its main network components.

Silencing a neuron means setting the weights of its outgoing synapses to zero, following the approach used in the reference implementation.

## Results

The results are analyzed in [`eval.ipynb`](eval.ipynb).

For one of the reported firing rate comparisons:

| | Firing rate |
|---|---|
| This implementation | 83.6 Hz |
| Reference result | 93.2 Hz|

The reproduced value is approximately 10.3% lower than the reference. This is a reasonably close preliminary result, but it should not be interpreted as a fully validated reproduction of the original experiment.

## Requirements and usage

Use Python 3.10 and install the dependencies:

```bash
pip install -r requirements.txt
```

Make sure the required FlyWire data files are available locally and that their paths are correctly configured in `config.yaml`. The data files are not included in this repository.

Run the experiment from the repository directory:

```bash
python sugarr_experiment.py config.yaml
```

The experiment results are saved as a Parquet file in the configured output directory.

## Reference

Shiu, P. K. et al. (2024). *A Drosophila computational brain model reveals sensorimotor processing.* Nature.

This project is an independently written implementation intended to reproduce and explore the published experiment. Its code was not copied or modified from the authors' implementation.

The original research and its associated resources can be found in the [original project repository](https://github.com/philshiu/Drosophila_brain_model).

## License

The code in this repository is released under the [MIT License](LICENSE).

---

This project was created for learning purposes. I’ve tried to write clear, reusable and well-documented code.
If you notice any issues or have suggestions, please let me know! 🌱