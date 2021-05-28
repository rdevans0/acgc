
import numpy as np

from chainer.backends import cuda
from chainer.variable import Variable, VariableNode

from .scrambler import Scrambler

from .fixpoint import fixpoint, fixpoint_dec


class Quant(Scrambler):
    """ Scrambler for fixpoint with a constant bitwidth
    """
    
    needs_next_grads = False
    
    def __init__(self, bits=8, tag='unk', rescale=1.0, zvc=False):
        self.bits = int(bits)
        if self.bits > 31:
            raise Exception('Bits above 31 will cause overflow!')
        self.tag = tag
        self.rescale = rescale
        self.auto_lambda = False
        self.zvc = zvc
        
        kw = {}
        if zvc:
            kw['zvc'] = zvc
            
        self.init_detail('quant', bits=self.bits, tag=tag, **kw)
    
    def scramble(self, x_var, var_name=None, **kwargs):
        """ Accepts an activation/variable and compresses then decompresses using Quant """
        if isinstance(x_var, Variable) or isinstance(x_var, VariableNode):
            var_name = x_var.name if var_name is None else var_name
            x = x_var.data
        else:
            var_name = 'UNNAMED' if var_name is None else var_name
            x = x_var
            
        signed = self.get_signed(x, var_name)
        
        assert x.ndim == 4, 'Only allows NCHW format tensors'
        
        if x.ndim == 4:
            axes = (0,2,3) # Channel-wise scaling
            exp = (None, Ellipsis, None, None) # Indices to expand array
        elif x.ndim == 2:
            axes = (0,)
            exp = (None, Ellipsis)
        else:
            raise Exception('Only allows NCHW or NC format tensors')
        
        # Scaling calculation
        x_max = abs(x).max(axis=axes)  # shape is (N,)
        scale = (self.rescale / (x_max + 1e-6))[exp]
        
        # Perform quantization
        bits = self.bits
        fbits = bits-1
        x *= scale
        fixpoint(x, bits, fbits, inplace=True, signed=signed, signed_1bit=False)
        
        # Undo quantization
        fixpoint_dec(x, bits, fbits, inplace=True, signed=signed, scale=scale)
        
        # Save statistics
        size = np.prod(x.shape)
        self._act_count += size
        if self.zvc and any(var_name.endswith(s) for s in ('-d','-r','-p')):
            xp = cuda.get_array_module(x)
            nz = xp.count_nonzero(x)
            mask_bits = x.size
            nz_bits = (bits * nz).sum()
            
            self._bit_count += mask_bits + nz_bits
            self._intr_bit_count += bits * size
            
            
        else:
            self._bit_count += bits * size
            self._intr_bit_count += np.NaN