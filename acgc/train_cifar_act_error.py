import sys


import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import updaters
from chainer.links import Classifier


import cifar

from common.reporting import begin_output, FileReport
from common import compression as compr
from common.compression import CompressedMomentumSGD, GradApproximator, FilterArgumentParser

def main(argv=sys.argv[1:]):
    if type(argv) == str:
        argv = argv.split()
        
    parser = FilterArgumentParser(
        description='Activation Error injection into CIFAR')
    
    cifar.add_base_arguments(parser)
    compr.add_ae_arguments(parser)
    
    parser.add_argument('--grad_approx',type=str, nargs='+', 
                        help='Grad approximation to use. Options are exact, {max,mean,median} <interval> <num_obs>')
    
    parser.add_argument('--cz', action='store_true', default=False,
                        help='User zero correction with unsigned values')
    
    
    args = parser.parse_args(argv)
    
    if args.grad_approx is None:
        args.grad_approx = ['exact']
    
    # Other settings and derived arguments
    end_trigger = (args.epoch,'epoch')
    log_interval = (1, 'epoch') if not args.log_interval else (args.log_interval, 'epoch')
    report_entries = [
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr', 'elapsed_time'
    ]
    
    # Header, and output directory
    logprint, report = begin_output(args, argv, 'report.txt', file=__file__)
    
    
    ##
    # Set up model and dataset iterators
    rng, fixed_seeds = cifar.seed_rng(args.seed, args.device)
    c = 10 if args.dataset == 'cifar10' else 100
    model_ = lambda *a, **kw: Classifier(cifar.models[args.model](c, *a, **kw))
    model = cifar.init_model(model_, args.device)
    train_iters, val_iter = cifar.load_dataset(
        args.batchsize, args.dataset, augment=args.augment)
    
    # Scrambler configuration
    # A scrambler manages the uncompressed -> compressed -> uncompressed transform
    scrambler_map, filterspec_map = compr.parse_ae_arguments(
        parser, args, rng, model.predictor.var_names, printfn=logprint)
    
    #print_helper_summary(scrambler_map, filterspec_map, logprint)
    #print_helper_map(scrambler_map, model.predictor.var_names, printfn=logprint)
    
    tags = set(s.tag for s in scrambler_map.values() if hasattr(s, 'tag'))
    if tags:
        report_entries += ['main/bits'] + ['main/'+t+'/bits' for t in sorted(tags)]

    # Set up an optimizer
    optimizer = CompressedMomentumSGD(
        scrambler_map, grad_approx=GradApproximator(*args.grad_approx),
        lr=args.learnrate, momentum=args.momentum)
    optimizer.setup(model)

    # Set up a trainer
    updater = updaters.StandardUpdater(train_iters, optimizer, device=args.device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)
    
    # Decay
    if args.weight_decay:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))
    if args.learnrate_decay:
        trainer.extend(extensions.ExponentialShift('lr', 0.5),
                           trigger=(args.learnrate_decay, 'epoch'))

    # Extensions
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.device))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport(report_entries), trigger=log_interval)
    trainer.extend(FileReport(report_entries, out=report), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=args.update_interval))
    
    
    # Snapshots
    trainer.extend(extensions.snapshot(), trigger=end_trigger, name='snapshot_end')
    if args.snapshot_every:
        trainer.extend(extensions.snapshot(
                filename='snapshot_{0.updater.epoch}_iter_{0.updater.iteration}'), 
                trigger=(args.snapshot_every, 'epoch'),
                name='snapshot_every')
    
    # Load from snapshot
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer, strict=False)
    
    ##
    # Run
    try:
        trainer.run()
    finally:
        report.close()
    
    return trainer

if __name__ == '__main__':
    main()
