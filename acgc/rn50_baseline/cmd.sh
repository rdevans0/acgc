pushd /home/rdevans/perforce/home/rdevans/papers/2021_neurips_autoquant/code/acgc
python3 train_cifar_act_error.py --model resnet50 -d 0 --dataset cifar10 --out rn50_baseline --grad_approx mean 100000 10
popd
