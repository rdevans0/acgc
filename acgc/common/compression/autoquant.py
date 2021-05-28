
import numpy as np

from chainer.backends import cuda
from chainer.variable import Variable, VariableNode

from .scrambler import Scrambler

from .fixpoint import fixpoint_multi, fixpoint_multi_dec



class AutoQuant(Scrambler):
    """ Automatic Quantization Scrambler
    
    Derivation time:
        If the error on x needs to be <= dx, we want to find the number of bits, b
        
        First consider the case for an unsigned activation. 
        If the activation range is [0, 2^b]
            dx <= 1    (Max when x=2^b)
            dx <~ 1/2  (Under rounding, and x < 2^(b-0.5) )
        
        If the activation range is [0, 1]  (Multiply by 2^-b)
            dx <= 2^-b, dx <~ 2^(-b-1)
            
        If the activation range is [0, 1/scale] (Multiply by 2*-b / scale)
            dx <= 2^-b / scale, dx <~ 2^(-b-1) / scale
        
        Using this for unsigned, we solve for b
            log2(dx) <= -log2(scale) - b
                   b <= -log2(dx) - log2(scale)
            
        For unsigned:
            b <= -log2(dx) - log2(scale)
            b <~ -log2(dx) - log2(scale) - 1
        
        For signed, we need a sign bit added
            b <= -log2(dx) - log2(scale) + 1
            b <~ -log2(dx) - log2(scale)
        
        The <= is when at the end of the scaling range, but most of the time  
        this does not occur. We use the <~ instead, which holds for any scaled
        activation x < 2^(b-0.5)
    
    """
    
    needs_next_grads = True
    
    def __init__(self, eps_bits=None, tag='unk', base_bits=16, 
                 signed=True, rescale=1.0, correct_zeros=False, zvc=False):
        self.tag = tag
        self.rescale = rescale
        self.base_bits = base_bits
        self.flat_conv_approx = True
        self.correct_zeros = correct_zeros
        self.zvc = zvc
            
        self.auto_lambda = True
        eps_bits = float(eps_bits)
        self.eps_bits = eps_bits
        self.lammy = None
        
        kw = {}
        if correct_zeros:
            kw['correct_zeros'] = correct_zeros
        if zvc:
            kw['zvc'] = zvc
            
        self.init_detail('autoquant', eps_bits=eps_bits, tag=tag, **kw)
    
    def scramble(self, x_var, props={}, stats=True, var_name=None):
        """ Accepts an activation/variable and compresses then decompresses using AutoQuant """

        if isinstance(x_var, Variable) or isinstance(x_var, VariableNode):
            var_name = x_var.name if var_name is None else var_name
            x = x_var.data
        else:
            var_name = 'UNNAMED' if var_name is None else var_name
            x = x_var
            
        signed = self.get_signed(x, var_name)
            
        xp = cuda.get_array_module(x)
        
        if (not signed) and self.correct_zeros:
            if xp == cuda.cupy:
                get_nza = cuda.elementwise(
                    'T x', 'T nza',
                    ''' nza = (x < 0.0) ? -0.0001 :
                              (x > 0.0) ?  0.0001 : 0.0; ''', 
                    'get_nonzero_adjust_kernel')
                nonzero_adjust = get_nza(x)
            else:
                nonzero_adjust = (xp.sign(x)*0.0001).astype(x.dtype)

        assert x.ndim == 4, 'Only allows NCHW format tensors'
        if x.ndim == 4:
            axes = (0,2,3) # Channel-wise scaling
            exp = (None, Ellipsis, None, None) # Indices to expand array
        elif x.ndim == 2:
            axes = (0,)
            exp = (None, Ellipsis)
        else:
            raise Exception('Only allows NCHW or NC format tensors')
            
        ndim = x.ndim
        
        labels = props['labels']
        n_succs = props['n_succs']
        
        # Intrinsic bitwidth
        if all(label=='Convolution2DFunction' for label in labels):
            N,C,H,W = x.shape
            
            sgy2s = [props['sgy2/%d'%i] for i in range(n_succs)]
            sgw2s = [props['sgw2/%d'%i] for i in range(n_succs)]
            w_shapes = [props['w_shape/%d'%i] for i in range(n_succs)]
            funcs = props['funcs']
            
            
            if not (len(w_shapes) == len(funcs) and len(funcs) == len(sgy2s) and len(funcs) == len(sgw2s)):
                raise Exception('sizes of shape, funcs, gy do not match')
            if not self.flat_conv_approx:
                raise NotImplementedError('flat_conv_approx = False')
                
            u = [shape[2] * shape[3] / f.func.sx / f.func.sy for  # (R*S)/T^2
                 shape,f in zip(w_shapes, funcs)]
            
            tot_sy = sum(u_*sy_ for u_,sy_ in zip(u,sgy2s))
            tot_sy = max(float(tot_sy), 1e-16)
            tot_sgw = max(float(sum(sgw2s)), 1e-16)
            
            # We are using eps_bits = -log2(abs(e))
            e = 2**-self.eps_bits
            numer = e * tot_sgw
            denom = 2 * N*C*H*W * tot_sy
            
            exp2 = (None,)*ndim
            dx2 = xp.array(numer/denom, dtype='f')[exp2] # (1,1,...)
            
        
        elif labels[0] == 'BatchNormalization' and len(labels) == 1:
            N,K,H,W = x.shape
            
            inv_std2 = props['inv_std2']
            gamma2 = props['gamma2']
            ggamma2 = props['ggamma2']
            #gbeta2 = props['gbeta2']
            nnz2_k = props['nnz2_k']
            gz2_max_k = props['gz2_max_k']
            sgx2 = props['sgx2']
            
            max_dy = abs(x).max(axis=(0,2,3)) / 2  # TODO: Change, Dumb approx for now
            approx_dggamma = inv_std2 * nnz2_k * gz2_max_k * (max_dy**2)
            g2_k = approx_dggamma + ggamma2
            
            e = 2**-self.eps_bits
            numer = e * sgx2 * N*H*W
            denom_k = K * gamma2 * (inv_std2**2) * g2_k
            
            exp2 = (None, Ellipsis, None, None)
            
            dx2 = (numer/denom_k).astype('f')[exp2] # (1,K,1,1)
            
            # Sometimes gbeta == ggamma == 0, avoid issues by setting a lower limit
            dx2 = xp.maximum(dx2, 2**-16)
            
        elif len(labels) != 1:
            raise Exception(
                'Multi layer only allowed for Conv2D, not {}'.format(labels))
        else:
            raise ValueError(
                'Unrecognized labels {}'.format(labels))
            
        # Scaling calculation
        x_max = abs(x).max(axis=axes)  # shape is (C,)
        scale = (self.rescale / (x_max + 1e-6))[exp] # shape is (1,C,1,1)
        
        scale_bits = xp.log2(scale)
        intrinsic_bits = -0.5*xp.log2(dx2)
        
        f = intrinsic_bits - scale_bits - 1
        fbits = xp.clip(xp.trunc(f), 1.0, self.base_bits-1)
        bits = fbits + 1
        
        # Perform quantization
        x *= scale
        fixpoint_multi(x, bits, fbits, inplace=True, signed=signed, signed_1bit=False, zero_adjust=None)
        # Undo quantization
        fixpoint_multi_dec(x, bits, fbits, inplace=True, signed=signed, scale=scale)
        
        if (not signed) and self.correct_zeros:
            if xp == cuda.cupy:
                nza = cuda.elementwise(
                    'T nza', 'T x', 
                    'x = (x == 0) ? nza : x', 
                    'nonzero_adjust_kernel')
                nza(nonzero_adjust, x)
            else:
                xp.where(x==0, nonzero_adjust, x, out=x)
        
        
        # Save statistics
        if stats:
            self._act_count += np.prod(x.shape + (1,))
            if not self.zvc:
                self._bit_count += bcast_sum(bits, x.shape)
                self._intr_bit_count += bcast_sum(intrinsic_bits, x.shape)
                
            else:
                assert (bits.shape[0], bits.shape[2], bits.shape[3]) == (1,1,1), 'Ensure 1d bits'
                nz = xp.count_nonzero(x, axis=(0,2,3))[None, :, None, None]
                mask_bits = x.size
                nz_bits = (bits * nz).sum() 
                
                self._bit_count += mask_bits + nz_bits
                self._intr_bit_count += bcast_sum(bits, x.shape)

def bcast_sum(x, shape):
    """ Efficient implementation of x.broadcast_to(shape).sum() """
    ctx = '{} -> {}'.format(x.shape, shape)
    if x.ndim != len(shape):
        raise ValueError('Dimension mismatch, ' + ctx)
    if x.ndim > 10: # arbitrary
        raise ValueError('Function is meant for small dimensions ' + ctx)
    
    scale = 1
    for i in range(x.ndim):
        if x.shape[i] == 1:
            scale *= shape[i]
        elif x.shape[i] == shape[i]:
            pass
        else:
            raise Exception('Shapes are not broadcastable ' + ctx)
    
    return x.sum() * scale