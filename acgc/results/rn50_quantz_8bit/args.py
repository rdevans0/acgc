{   'ae_autoquant'      : None,
   'ae_autoquantz'     : None,
   'ae_quant'          : [['8', 'BN', 'BR1234c']],
   'ae_quantz'         : [['8', 'CNV', 'B1ir,R1234rs']],
   'augment'           : 2,
   'batchsize'         : 128,
   'cz'                : False,
   'dataset'           : 'cifar10',
   'device'            : 0,
   'epoch'             : 300,
   'grad_approx'       : ['mean', '100000', '1'],
   'learnrate'         : 0.05,
   'learnrate_decay'   : 70,
   'log_interval'      : None,
   'model'             : 'resnet50',
   'momentum'          : 0.9,
   'out'               : 'rn50_quantz_8bit',
   'resume'            : '',
   'seed'              : None,
   'snapshot_every'    : None,
   'update_interval'   : 20,
   'weight_decay'      : 0.0005,
}