# Code For AC-GC: Lossy Activation Compression with Guaranteed Convergence
This code is intended to be used as a supplemental material for submission to NeurIPS 2021.

**DO NOT DISTRIBUTE**

## Setup
This code is tested on Ubuntu 20.04 with Python 3 and CUDA 10.1. 
Other cuda versions can be used by modifying the cupy version in [requirements.txt](requirements.txt), provided that CuDNN is installed.

```bash
# Set up environment
python3 -m venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Training
Configurations are provided for CIFAR10/ResNet50 in the acgc/configs folder. 

```bash
source venv/bin/activate
cd acgc
./configs/rn50_baseline.sh
```

To replicate GridQuantZ results from the paper, you additionally need to:
* Run quantz with bitwidths of 2, 4, 6, 8, 10, 12, 14, and 16 bits, and run each 5 times
* Select the result with the lowest bitwidth and average accuracy no less than the baseline - 0.1%

## Evaluation
Evaluation with the CIFAR10 test dataset is run during training. 
The 'validation/main/accuracy' entry in the report.txt or log contains test accuracy throughout training.

## Pre-trained Models
You can download pre-trained snapshots for each config from acgc/configs.
* [rn50_baseline](https://drive.google.com/uc?export=download&id=1gYqiHqgowgNAekgA4InUbrD_ROvAU_Ku)
* [rn50_quant_8bit](https://drive.google.com/uc?export=download&id=1_7u6xvplYWW-34OSvgCuvApXbtDsVYmr)
* [rn50_quantz_8bit](https://drive.google.com/uc?export=download&id=1GOCfsca3qzEd-ICsTpgrFUNvTKJhmB3h)
* [rn50_autoquant](https://drive.google.com/uc?export=download&id=1P-FJgcUHvrVGsUzPMK2QgSX7-cYx9sm1)
* [rn50_autoquantz](https://drive.google.com/uc?export=download&id=1IqSkcNaEEt7ThpyypOuSHvQDmAWhiuHZ)

These snapshots can be run using

```bash 
python3 train_cifar_act_error.py ... --resume <snapshot_file>
```

## Results
We have added reports and logs for each configuration under acgc/results. The logs are associated with each snapshot, above.

A summarized output from these runs is:

| Configuration    | Best Val. Acc     | Average Bits | Epochs |
|------------------|:-----------------:|:------------:|:------:|
| rn50_baseline    |  95.16 %          |  N/A         | 300    |
| rn50_quant_8bit  |  94.90 %          |  8.000       | 300    |
| rn50_quantz_8bit |  94.82 %          |  7.426       | 300    |
| rn50_autoquant   |  94.73 %          |  7.305       | 300    |
| rn50_autoquantz  |  94.91 %          |  6.694       | 300    |

## Code Layout
Argument parsing and model initialization are handled in [acgc/cifar.py](acgc/cifar.py) and [acgc/train_cifar_act_error.py](acgc/train_cifar_act_error.py)

Modifications to the training loop are in [acgc/common/compression/compressed_momentum_sgd.py](acgc/common/compression/compressed_momentum_sgd.py).

The baseline fixpoint implementation is in [acgc/common/compression/quant.py](acgc/common/compression/quant.py).

The AutoQuant implementation, and error bound calculation are in [acgc/common/compression/autoquant.py](acgc/common/compression/autoquant.py).

Gradient and parameter estimation are performed in [acgc/common/compression/grad_approx.py](acgc/common/compression/grad_approx.py)


