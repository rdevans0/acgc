
import numpy as np
from collections import defaultdict

import chainer
from chainer.optimizers import MomentumSGD

from common.compression.cg_tools import CGraph
from common.compression.grad_approx import ExtractOutGrads
        
class CompressedMomentumSGD(MomentumSGD):
    """ ScramblerUpdater which allows exact activation gradient information as an input
    
    Requires two backward passes
    """
    def __init__(self, scrambler_map={}, grad_approx=None, **kwargs):
        super(CompressedMomentumSGD, self).__init__(**kwargs)
        self.scrambler_init(scrambler_map, grad_approx)
    
    def scrambler_init(self, scrambler_map, grad_approx=None):
        self.scrambler_map = scrambler_map
        self.grad_approx = grad_approx
        
    def update(self, lossfun=None, *args, **kwds):
        ''' Modified from class GradientMethod(Optimizer) '''
        
        model = self.target
        predictor = model.predictor
        if hasattr(predictor,'retain'):
            predictor.retain(False) # Clear any old vars
        
        scramblers = list(set(s for s in self.scrambler_map.values() if s))
        
        
        
        if lossfun is not None:
            use_cleargrads = getattr(self, '_use_cleargrads', True)
            cleargrads = model.cleargrads if use_cleargrads else model.zerograds
            
            assert use_cleargrads, (
                'We use move semantics, so zeroing grads will cause saved '
                'gradients to be erased')
            
            need_grads = any(getattr(s, 'needs_next_grads', False) for s in scramblers)
            
            # Forward
            predictor.retain(True)
            loss = lossfun(*args, **kwds)
            
            # Convenience iters
            def nvs_iter():
                for name, var in predictor.namedvars():
                    if var.data is None:
                        continue
                    scrambler = self.scrambler_map[var.name]
                    if scrambler is None: 
                        continue
                    yield name, var, scrambler
                    
            def name_var_iter():
                for name, var, scrambler in nvs_iter():
                    if not getattr(scrambler, 'needs_next_grads', False):
                        continue
                    yield name, var
            
            self.grad_approx.setup_iter(self.t, list(name_var_iter()), loss, model)
            
            cleargrads()
            
            ##
            # Get retained grads
            # Might perform an extra backward pass
            if need_grads:
                layer_props = self.grad_approx.get_or_calc()
            
            ##
            # Scramble Activations
            for scrambler in scramblers:
                scrambler.scramble_setup()
            
            
            if self.grad_approx._is_statistic:
                scramble_stats = self.grad_approx.needs_extract(self.t)
            else:
                scramble_stats = True
            
                
            for name, var, scrambler in nvs_iter():
                if getattr(scrambler, 'needs_next_grads', False):
                    props = layer_props[name]
                    do_not_keep = ( all(label == '_ + _' for label in props['labels'])
                                   and any(name.endswith('-'+c) for c in 'cb') )
                else:
                    props = {}
                    do_not_keep = False
                    
                if do_not_keep:
                    var.data[...] = np.NaN
                else:
                    scrambler.scramble(var, props=props, stats=scramble_stats)
            
            # Backward Pass
            if self.grad_approx.needs_extract(self.t):
                
                cg = CGraph(model, loss)
                var_names, variables = zip(*cg.namedvars())
                param_names, params = zip(*model.namedparams())
                
                with ExtractOutGrads('BatchNormalization') as h:
                    all_grads = chainer.grad([loss], variables + params)
                    bn_gzs = h.grads
                
                grad_dict = dict(zip(var_names+param_names, all_grads))
                assert len(grad_dict) == len(var_names) + len(param_names), (
                    'There is overlap between variable and grad names')
                
                
                self.grad_approx.add(self.t, grad_dict, bn_gzs)
                
                del grad_dict, all_grads, bn_gzs, variables, params  # cleanup
                
                predictor.retain(False) # Delete old vars
                
            else:
                predictor.retain(False) # Delete old vars
                loss.backward()
                
            del loss
        
            
            if scramble_stats:
                # Average bit width statistic for tracking compression
                act_count = defaultdict(int)
                bit_count = defaultdict(int)
                intr_bit_count = defaultdict(int)
                
                for scrambler in set(self.scrambler_map.values()):
                    if scrambler is not None and hasattr(scrambler, 'tag'):
                        scrambler.add_counts(act_count, bit_count, intr_bit_count)
                
                keys = list(act_count.keys()).copy()
                for k in keys:
                    act_count['_'] += act_count[k]
                    bit_count['_'] += bit_count[k]
                    intr_bit_count['_'] += intr_bit_count[k]
            
                for k, N in act_count.items():
                    N = N+0.001
                    bits = (1.0*bit_count[k]) / N
                    intr_bits = (1.0*intr_bit_count[k]) / N
                    k2 = '' if k == '_' else k+'/'
                    chainer.report({k2+'bits':bits, k2+'intr_bits':intr_bits}, model)
        
        else:
            raise Exception('Scrambler updater was skipped (are you running on multiple GPUs?)')
        
        # Finish up
        self.reallocate_cleared_grads()
        self.call_hooks()
        
        self.t += 1
        
        for name, param in self.target.namedparams():
            assert param.grad is not None
            param.update()
