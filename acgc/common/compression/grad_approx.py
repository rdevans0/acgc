
from collections import defaultdict, deque

import chainer
from chainer.backends import cuda
from chainer.function_hook import FunctionHook

from common.compression.cg_tools import CGraph

class GradApproximator:
    
    raw_props = ('n_succs','labels','funcs')  # props which do not accumulate
    
    def __init__(self, method, interval=100, hist_len=10):
        assert method in ('exact','max','median','mean')
        self._method = method
        
        self.interval = int(interval)
        self._next_extract = -1
        
        self._is_statistic = ( method in ('max','median','mean') )
        
        
        xp = cuda.cupy
        def convert(q):
            if isinstance(q[0], xp.ndarray):
                return xp.stack(list(q), axis=0)
            else:
                return xp.array(list(q))
            
        if method == 'max':
            self.mfunc = lambda x: xp.max(convert(x), axis=0)
        elif method == 'median':
            self.mfunc = lambda x: xp.median(convert(x), axis=0)
        elif method == 'mean':
            self.mfunc = lambda x: xp.mean(convert(x), axis=0)
        
        if self._is_statistic:
            self.prop_history = dict()
            self.prop_summary = dict()
            self.hist_len = int(hist_len)
        self._prop_init = False
        
        print('using GradApproximator({}, interval={}, hist_len={})'.format(
            method, interval, hist_len))
    
    def setup_iter(self, t, name_var_iter, loss, model):
        if self._next_extract == -1:
            self._next_extract = t - 1 # Only on first iteration
        self._t = t
        self.name_var_iter = name_var_iter
        self.loss = loss
        self.model = model
        self.cg = CGraph(model, loss)
    
    def get_or_calc(self):
        if self._method == 'exact':
            grad_dict, bn_gzs = self.get_exact_grads()
            props = _compute_layer_properties(self.cg, self.name_var_iter, grad_dict, bn_gzs)
            
        elif self._is_statistic:
            if not self._prop_init:
                grad_dict, bn_gzs = self.get_exact_grads()
                props = _compute_layer_properties(self.cg, self.name_var_iter, grad_dict, bn_gzs)
                self.update_prop_history(props, init=True)
            
            props = self.prop_summary
            assert len(props)
            
        else:
            raise Exception('Unknown method {}'.format(self.method))
        
        # Remap the properties to be in a more usable format
        props_remapped = defaultdict(dict)
        for k,v in props.items():
            var_name, prop_name = k.split('@')
            props_remapped[var_name][prop_name] = v
        
        return props_remapped
    
    def needs_extract(self, t):
        if self._method == 'exact':
            return False
        else:
            return ( t >= self._next_extract )
            
    
    def add(self, t, grad_dict, bn_gzs):
        if self._method == 'exact' or ( t < self._next_extract ):
            return
        
        elif self._method in ('max','median','mean'):
            self._next_extract = self._next_extract + self.interval
            
            all_props = _compute_layer_properties(self.cg, self.name_var_iter, grad_dict, bn_gzs)
            self.update_prop_history(all_props, init=True)
            
    
    def update_prop_history(self, props, init=False):
        if not self._is_statistic:
            return
        
        hist = self.prop_history
        
        remaining_keys = set(hist.keys())
        
        for k,v in props.items():
            is_raw_prop = k.split('@')[-1] in GradApproximator.raw_props
            
            if k not in hist:
                if init==False:
                    raise KeyError(
                        '{} in props but not in prop_history'.format(k))
                
                if not is_raw_prop:
                    hist[k] = deque([], self.hist_len)
            
            if is_raw_prop:
                hist[k] = v  # Direct copy
            else:
                hist[k].append(v)
            
            if k in remaining_keys:
                remaining_keys.remove(k)
        
                
        if init:
            self._prop_init = True
            
        elif len(remaining_keys):
            raise Exception(
                'Keys {} were not found in props, '.format(remaining_keys)
                + 'but were in prop_history')
        
        # Summarize props, now that they have been updated
        def f(kv):
            k,v = kv
            if k.split('@')[-1] in GradApproximator.raw_props:
                return (k,v)
            else:
                fv = self.mfunc(v)
                if fv.size == 1:
                    return (k, float(fv))
                else:
                    return (k, fv)
        
        self.prop_summary = dict(map(f, self.prop_history.items()))
        
        
    
    def get_exact_grads(self):
        var_names, variables = zip(*self.cg.namedvars())
        param_names, params = zip(*self.model.namedparams())
        
        # Other options:
        #   chainer.backward -> not introduced until ~ v7
        #   loss.backward    -> Deletes variables after use 
        #                       (we are not done with them yet!)
        with ExtractOutGrads('BatchNormalization') as h:
            all_grads = chainer.grad([self.loss], variables + params)
            bn_gzs = h.grads
        
        grad_dict = dict(zip(var_names+param_names, all_grads))
        assert len(grad_dict) == len(var_names) + len(param_names), (
            'There is overlap between variable and grad names')
        
        # Cleargrad(s) uses 
        #     v.grad_var = None
        # Effectively we are using move semantics on v._grad_var.array (v.grad)
        self.model.cleargrads() # calls param.cleargrad()
        for name, var in self.model.predictor.namedvars():
            var.cleargrad()

        return grad_dict, bn_gzs

def _compute_layer_properties(cg, name_var_iter, grad_dict, bn_gzs):
        
    all_layer_props = {}
    
    for name, var in name_var_iter:
        succs = cg.successors(name)
        labels = [f.label for f in succs]
        
        def insert(k,v, all_layer_props=all_layer_props, prefix=name):
            k2 = name + '@' + k
            assert k2 not in all_layer_props, 'Cannot insert to %s, key exists'%(k2)
            all_layer_props[k2] = v
        
        
        # Ignore sum layers (No dependence on activation)
        if len(succs) > 1 and '_ + _' in labels:
            succs = [f for f in succs if f.label != '_ + _']
            labels = [f.label for f in succs]
        
        insert('n_succs', len(succs))
        insert('labels', [f.label for f in succs])
        insert('funcs', succs)
        if all(s=='Convolution2DFunction' for s in labels):
            
            for i,f in enumerate(succs):
                next_var = f.out_vars[0]
                gy = grad_dict[next_var.name].data
                sgy2 = float((gy**2).sum())
                insert('sgy2/%d'%i, sgy2)
                
                gW = grad_dict[f.path + '/W'].data
                sgW2 = float((gW**2).sum())
                insert('sgw2/%d'%i, sgW2)
                
                insert('w_shape/%d'%i, gW.shape)
        
        elif len(labels) == 1 and labels[0] == 'BatchNormalization':
            f = succs[0]
            assert len(f.out_vars) == 1, (
                'Assumed that nothing returns more than one value')
            
            insert('inv_std2', f.func.inv_std**2)
            insert('gamma2', f.in_params['gamma'].data**2)
            insert('ggamma2', grad_dict[f.path + '/gamma'].data**2)
            insert('gbeta2', grad_dict[f.path + '/beta'].data**2)
            
            gz = bn_gzs[f.func]
            nnz2_k = (gz != 0).sum(axis=(0,2,3)) ** 2
            gz2_max_k = (gz**2).max(axis=(0,2,3))
            
            insert('nnz2_k', nnz2_k)
            insert('gz2_max_k', gz2_max_k)
            
            gx = grad_dict[name].data
            sgx2 = float((gx**2).sum()) # E(||gW||^2) w.r.t. batch, but just use one as an estimate
            insert('sgx2', sgx2)
        
        elif all(label == '_ + _' for label in labels):
            if any(name.endswith('-'+c) for c in 'cb'):
                # This type of activation isn't used at all, normally would be discarded
                continue
        
            
        else: # if need_grads
            raise Exception(
                'Do not know how to handle successors {} for {}->{}'.format(
                    labels, name, succs))
    
    return all_layer_props


class ExtractOutGrads(FunctionHook):
    def __init__(self, target_label):
        self.grads = dict()
        self.target_label = target_label

    def backward_preprocess(self, func, in_data, out_grad_arrays):
        
        ctx = '{} at grad {}'.format(func, out_grad_arrays[0].shape)
        
        if func.label == self.target_label:
            #print('Extracting ' + ctx)
            assert len(out_grad_arrays) == 1, (
                'Assumes only one output grad per function, ' + ctx)
            assert func not in self.grads, (
                'func encountered more than once!, ' + ctx)
            self.grads[func] = out_grad_arrays[0]