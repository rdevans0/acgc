## neurips_2021_acgc

# Setup
This code is tested on Ubuntu 20.04 with Python 3 and CUDA 10.1. 
Other cuda versions can be used by modifying the cupy version in [](requirements.txt), provided that CuDNN is installed.

```bash
# Set up environment
python3 -m venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# Running
Configurations are provided for CIFAR10/ResNet50 in the [](acgc/configs) folder. 

```bash
source venv/bin/activate
cd acgc
./configs/rn50_baseline.sh
```

# Code layout

Modifications to the training loop are in [](acgc/common/compression/compressed_momentum_sgd.py).

The AutoQuant implementation, and error bound calculation is in [](acgc/common/compression/autoquant.py).

Gradient and parameter estimation is performed in [](acgc/common/compression/grad_approx.py)

# Results

We have added example results for each configuration under [](acgc/results).
