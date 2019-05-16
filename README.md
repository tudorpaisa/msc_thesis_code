# Repo for my Master's Thesis

WARNING: Code is somewhat messy and undocumented (it's on my to-do list). A good chunk of `data.py` is now "cruft code" (unused).

## How to

All the important script are in the `src` directory. To replicate the experiments in my thesis, they must be executed in the order shown below. However, you need to download the dataset first (run `download_data.sh`).

```bash
# Parse dataset for the baseline
python3 data.py

# Train the baseline (~10 days to train)
cp ../params/baseline_t3.json ../params/test.json
python3 run.py

# Train PerformanceRNN (~1/2 day)
cp ../params/performance_rnn_t1.json ../params/test.json
python3 run.py

# Train C-RNN-GAN (~1 day)
cp ../params/c_rnn_gan_t1.json ../params/test.json
python3 run.py

# Generate songs
python3 generate.py

# Evaluate generated songs
python3 evaluation.py

# Make pretty pictures of the NNs
#  (loss and neuron activations)
python3 plots.py
```

## Dataset

To download the dataset, just run the `download_data.sh` script.

```bash
chmod +x download_data.sh
./download_data.sh
```

## Dependencies

**Python**: `pytorch numpy mido pandas pretty_midi sklearn matplotlib`

**Shell**: `curl unzip`

## To-Do

- [ ] Document Code
