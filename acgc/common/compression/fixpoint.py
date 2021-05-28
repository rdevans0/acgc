#!/usr/bin/env python3

from chainer.backends import cuda
from chainer.backends.cuda import cupy
import numpy as np

import unittest

dprint = lambda *a, **kw: None
# dprint = print


def rp(v, s='5.3'):
    if isinstance(v, cuda.cupy.ndarray):
        v = cuda.to_cpu(v)
    fmt = '% {}f'.format(s)
    s = np.array2string(v, formatter={'float': lambda v: fmt%v})
    print(s)

def fixpoint(x, bits=8, fbits=5, bias=None, zero_adjust=None, signed=True, use_cupy=None, inplace=False,
             signed_1bit=False): # Must be false for compatibility
    if not inplace:
        y = x.copy() # Hack in non-inplace calls
        return fixpoint(y, bits, fbits, bias=bias, zero_adjust=zero_adjust, signed=signed, use_cupy=use_cupy, 
                        inplace=True, signed_1bit=signed_1bit)
        
    
    xp = cuda.get_array_module(x)
    if use_cupy is None:
        use_cupy = (xp is cuda.cupy)
    
    
    
    if signed is None:
        signed = x.min() < 0 # Note, this is an EXPENSIVE operation
    
    if signed_1bit is None:
        signed_1bit = signed
    
    bias = 0 if signed else -(1<<bits-1)
    
    if zero_adjust is None:
        zero_adjust = 0 if signed else -40
        
    clip_min = -(1<<bits-1)
    clip_max = (1<<bits-1)-1
    
    if use_cupy:
        assert x.dtype == np.float32, 'optimized cupy routines assume float32'
        # Use templating to create an optimized routine
        name = 'fixpoint_{}bits_{}fbits'.format(bits, fbits)
        code = ''' 
            float sh = ldexp(x, {fbits});
            float za = (sh == 0) ? sh + adj + bias : sh + bias; 
            int y = __float2int_rn(za);
            int min = {min};
            int max = {max};
            if ({bits} == 1) {{
                if ({signed_1bit}) 
                    x = (x >= 0) ? 1.0 : -1.0;  
                else
                    x = (float) ((y <= min) ? min :         // x = clip(y, min, max)
                            (y >=  max) ? max : y );
            }} else {{
                x = (float) ((y <= min) ? min :         // x = clip(y, min, max)
                            (y >=  max) ? max : y ); 
            }}
        '''.format(fbits=fbits, bits=bits, signed_1bit=int(signed_1bit), min=clip_min, max=clip_max)
        
        cuda.elementwise(
            'float32 bias, float32 adj',
            'float32 x', code, name
          ) (bias, zero_adjust, x)
     
    else:
        # dprint('SINGLE')
        # dprint(f'    signed: {signed}, signed_1bit: {signed_1bit}')
        # dprint(f'    clip_min:{clip_min} clip_max:{clip_max}')
        xp.ldexp(x, fbits, out=x)
        # dprint(f'    shifted: {x}')
        zeros = (x == 0)
        
        if signed_1bit:
            signs = xp.sign(x) + zeros # 1 if x >= 0 else -1
            
        x += bias + zeros*zero_adjust
        # dprint(f'    adjusted: {x}')
        xp.rint(x, out=x)
        # dprint(f'    rounded: {x}')
        
        if signed_1bit and bits == 1:
            # dprint(f'    signs: {signs}')
            x[...] = signs
        else:
            xp.clip(x, clip_min, clip_max, out=x)
        
    return x
    
def fixpoint_dec(x, bits=8, fbits=5, signed=True, use_cupy=None, 
                 inplace=False, scale=None):
    if not inplace:
        y = x.copy() # Hack in non-inplace calls
        return fixpoint_dec(y, bits, fbits, signed=signed, use_cupy=use_cupy, 
                              inplace=True)
    
    xp = cuda.get_array_module(x)
    if use_cupy is None:
        use_cupy = (xp is cuda.cupy)    
        
    if signed is None:
        signed = x.min() < 0 # Note, this is an EXPENSIVE operation
    
    bias = 0 if signed else -(1<<bits-1)
    
    if use_cupy:
        assert x.dtype == np.float32, 'optimized cupy routines assume float32'
        
        name = 'fixpoint_dec_{}bits_{}fbits{}'.format(bits, fbits, '_signed' if signed else '')
        
        if scale is None:
            code = ''' 
                float bias = ({signed}) ? 0 : -(1<<{bits}-1);
                x = (x - bias) / (1<<{fbits});
            '''.format(fbits=fbits, bits=bits, signed=int(signed))
            
            cuda.elementwise(
                '',
                'float32 x', code, name
              ) (x)
            
        else:
            scale = xp.broadcast_to(scale, x.shape)
            name += '_scale'
            code = ''' 
                float bias = ({signed}) ? 0 : -(1<<{bits}-1);
                x = (x - bias) / (1<<{fbits}) / scale; 
            '''.format(fbits=fbits, bits=bits, signed=int(signed))
            
            cuda.elementwise(
                'float32 scale',
                'float32 x', code, name
              ) (scale, x)
    
    else:
        x -= bias
        x /= (2**fbits) * scale
    
    
def fixpoint_multi(x, bits=8, fbits=5, signed=True, use_cupy=None, 
                   inplace=False, signed_1bit=None, zero_adjust=0):
    """ Fixpoint compression with a bitwidth array. Only fbits can be an array """
    
    if not inplace:
        y = x.copy() # Hack in non-inplace calls
        return fixpoint_multi(y, bits, fbits, signed=signed, use_cupy=use_cupy, 
                              inplace=True, signed_1bit=signed_1bit, zero_adjust=zero_adjust)
    
    bits = bits + 1 # Correct the bitwidth calculation for this function
    fbits = fbits + 1
    
    
    xp = cuda.get_array_module(x)
    if use_cupy is None:
        use_cupy = (xp is cuda.cupy)    
    
    if isinstance(bits, np.ndarray) or isinstance(fbits, cupy.ndarray):
        if bits.ndim != x.ndim:
            raise ValueError('bits must be int or array of same ndim as x')
        
    if isinstance(fbits, np.ndarray) or isinstance(fbits, cupy.ndarray):
        if fbits.ndim != x.ndim:
            raise ValueError('fbits must be int or array of same ndim as x')
            
    bits = xp.broadcast_to(bits, x.shape).astype('i')
    fbits = xp.broadcast_to(fbits, x.shape).astype('i')
    
    if signed is None:
        signed = x.min() < 0 # Note, this is an EXPENSIVE operation
    
    if signed_1bit is None:
        signed_1bit = signed
    
    bias = xp.zeros_like(bits) if signed else -(1<<bits-1)
    bias = bias.astype('f')
    
    if zero_adjust is None:
        zero_adjust = 0 if signed else -40
    
    
    if use_cupy:
        assert x.dtype == np.float32, 'cupy routines assume float32'
        
        name = 'fixpoint_multi' + ('_signed_1bit' if signed_1bit else '')
        code = ''' 
            float sh = ldexp(x, fbits);
            float za = (sh == 0) ? sh + adj + bias : sh + bias; 
            int y = __float2int_rn(za);
            int min = -(1<<bits-1);
            int max = (1<<bits-1)-1;
            if (bits == 1) {{
                if ({signed_1bit}) 
                    x = (x >= 0) ? 0.0 : -2.0;  // becomes 1,-1 when bias is added
                else
                    x = (float) ((y <= min) ? min :         // x = clip(y, min, max)
                            (y >=  max) ? max : y );         
            }} else {{
                x = (float) ((y <= min) ? min :         // x = clip(y, min, max)
                            (y >=  max) ? max : y ); 
            }}
            
        '''.format(signed_1bit=int(signed_1bit))
        
        
        cuda.elementwise(
            'float32 bias, float32 adj, int32 bits, int32 fbits',
            'float32 x', code, name
          ) (bias, zero_adjust, bits, fbits, x)
     
    else:
        # dprint('MULTI')
        # dprint(f'    signed: {signed}, signed_1bit: {signed_1bit}')
        clip_min = -(1<<bits-1)
        clip_max = (1<<bits-1)-1
        # dprint(f'    clip_min:{clip_min.mean()} clip_max:{clip_max.mean()}')
        
        xp.ldexp(x, fbits, out=x)
        # dprint(f'    shifted: {x}')
        zeros = (x == 0)
        
        if signed_1bit:
            signs = xp.sign(x) + zeros # 1 if x >= 0 else -1
        x += bias + zeros*zero_adjust
        
        # dprint(f'    adjusted: {x}')
        xp.rint(x, out=x)
        # dprint(f'    rounded: {x}')
        if signed_1bit:
            # dprint(f'    signs: {signs}')
            x[...] = xp.where(bits==1, signs, 
                         xp.clip(x, clip_min, clip_max))
        else:
            xp.clip(x, clip_min, clip_max, out=x)
        
    return x


def fixpoint_multi_dec(x, bits=8, fbits=5, signed=True, use_cupy=None, 
                 inplace=False, scale=None):
    if not inplace:
        y = x.copy() # Hack in non-inplace calls
        return fixpoint_multi_dec(y, bits, fbits, signed=signed, use_cupy=use_cupy, 
                              inplace=True, scale=scale)
    
    bits = bits + 1 # Correct the bitwidth calculation for this function
    fbits = fbits + 1
    
    xp = cuda.get_array_module(x)
    if use_cupy is None:
        use_cupy = (xp is cuda.cupy)    
        
    if signed is None:
        signed = x.min() < 0 # Note, this is an EXPENSIVE operation
    
    
    bits = xp.broadcast_to(bits, x.shape).astype('i')
    fbits = xp.broadcast_to(fbits, x.shape).astype('i')
    
    bias = xp.zeros_like(bits) if signed else -(1<<bits-1)
    bias = bias.astype('f')
    
    if use_cupy:
        assert x.dtype == np.float32, 'optimized cupy routines assume float32'
        
        name = 'fixpoint_multi_dec'
        
        if scale is None:
            code = ''' 
                x = (x - bias) / (1<<fbits);
            '''
            
            cuda.elementwise(
                'float32 bias, int32 fbits',
                'float32 x', code, name
              ) (bias, fbits, x)
            
        else:
            scale = xp.broadcast_to(scale, x.shape)
            name += '_scale'
            code = ''' 
                x = (x - bias) / (1<<fbits) / scale;
            '''
            
            cuda.elementwise(
                'float32 bias, int32 fbits, float32 scale',
                'float32 x', code, name
              ) (bias, fbits, scale, x)
    
    else:
        x -= bias
        if scale is None:
            x /= (2**fbits) * scale
        else:
            x /= (2**fbits)
    
    return x


##############################################################################
##############################################################################
# Unittests

class MultiSameTest(unittest.TestCase):
    def run_same(self, init_fn, bits, fbits, **kw):
        x = init_fn()
        xp = cuda.get_array_module(x)
        
        kw['inplace'] = False
        y_fix = cuda.to_cpu(fixpoint(x, bits, fbits, **kw))
        bits_a = xp.ones_like(x) * bits
        fbits_a = xp.ones_like(x) * fbits
        y_multi = cuda.to_cpu(fixpoint_multi(x, bits_a, fbits_a, **kw))
        
        msg = 'xp: {}'.format(xp)
        msg += '\nFormat is {}.{}'.format(bits, fbits)
        msg += '\nin: ' + np.array_repr(cuda.to_cpu(x), max_line_width=128, precision=3)
        np.testing.assert_array_equal(y_fix, y_multi, msg)

def _add_tests():
    seeds = [0xcafecafe, 0x12ef903]
    shape = (4,4)
    for seed_ind, seed in enumerate(seeds):
        #for xp, xp_name in (np, 'numpy'), (cuda.cupy, 'cupy'):
        for xp, xp_name in (np, 'numpy'),:
            for signed in ('signed','unsigned'):
                
                def init_signed():
                    v = xp.random.random(shape).astype('f')*8 - 4
                    nz = xp.random.random(shape) > 0.2
                    return v*nz
                
                def init_unsigned():
                    v = xp.random.random(shape).astype('f')*8
                    nz = xp.random.random(shape) > 0.2
                    return v*nz
                
                init_fn = init_signed if signed == 'signed' else init_unsigned
                
                for bits,fbits in ((8,4), (3,2), (3,1), (3,0), (2,1), (1,0)):
                    for s1b in (None, False, True,):
                        for signed_arg in (None, False, True):
                            kwargs = dict(zero_adjust=False, 
                                          signed=signed_arg, 
                                          signed_1bit=s1b)
                            
                            def test(self):
                                self.run_same(init_fn, bits, fbits, **kwargs)
                            
                            test_name = 'test_{}_{}_{}1b_{}S_{}_{}bits_{}'.format(
                                signed, xp_name,
                                s1b,
                                signed_arg,
                                bits,fbits, seed_ind)
                            
                            setattr(MultiSameTest, test_name, test)

_add_tests()


class ErrorTest(unittest.TestCase):
    
    def __init__(self, *a):
        self.num_tests = 100
        self.bit_tests = [2,3,4,5,6,7,8]
        self.shape = (1,100,1,20)
        self.xp = cuda.cupy
        self.seed = 0xcafecafe
        
        def pdb_(msg):
            print(msg)
            import pdb
            pdb.set_trace()
        
        def raise_(msg):
            raise Exception(msg)
            
        # What to do when an error is encountered, uncomment ONE
        #self.on_error = lambda msg: None  # Ignore
        # self.on_error = pdb_
        self.on_error = lambda msg: self.assertTrue(False, msg)
        # self.on_error = raise_
        
        self.seen = set()
        self.check_levels = False
        
    
    def check_error_scaling(self, x, error_bits, msg):
        xp = cuda.get_array_module(x)
        shape = x.shape
        
        signed = x.min() < 0.0
        
        x = x.astype('f')
        x_old = x.copy()
        
        x_max = abs(x).max(axis=(0,2,3))  # shape is (C,)
        scale = (1.0 / (x_max + 1e-6))[None, :, None, None]
        
        if not signed:
            b = error_bits - xp.log2(scale) - 1 + 1
        else:
            b = error_bits - xp.log2(scale) - 1 + 2  # +1 sign bit
        
        b = xp.maximum(xp.ceil(b), 1)
        bits = xp.broadcast_to(b, shape).astype('f')
        xs = x*scale
        
        # Why bits+1? Because the code doesn't know how to handle fbits == bits
        y = fixpoint_multi(xs, bits+1, bits, inplace=False, signed_1bit=False)
        x_new = fixpoint_multi_dec(y, bits+1, bits, inplace=False, scale=scale)
        
        if self.check_levels:
            import re
            k = re.sub('test \d+','', msg)
            if k not in self.seen:
                bl = int(xp.rint(y.min()) - 1)
                bh = int(xp.rint(y.max()) + 1)
                levels = sum([(y == v).any() for v in range(bl, bh+1)])
                print('{} {} [{:5.1f}, {:5.1f}] {} {}'.format(2**int(bits.max()), levels, bl+1, bh-1, signed, k))
                self.seen.add(k)
        
        dx2 = (x_old - x_new)**2
        
        # if True: # Per-channel comparison
        #     all_abe = -0.5*xp.log2(dx2)
        #     actual_bit_error = -0.5*xp.log2(dx2.max(axis=(0,2,3)))
            
        #     ind = (36, 18, 15, 4)
        #     print('x_old')
        #     rp(x_old[ind[:3]], '8.3')
        #     print('')
        #     print('x_new')
        #     rp(x_new[ind[:3]], '8.3')
        #     print('')
        #     print('abe')
        #     rp(all_abe[ind[:3]], '8.3')
        #     print('')
            
        #     if signed:
        #         actual_bit_error -= 1 # Remove the sign bit
            
        #     if (actual_bit_error < error_bits).any():
        #         reason = 'Error too high, {} is not more than {}. '.format(actual_bit_error, error_bits) + msg
        #         self.on_error(reason) # Might NOT exit!!
                
        #     elif x.shape[1] > 10 and (actual_bit_error >= (error_bits+1.0)).any():
        #         reason = 'Error too low, {} is not less than {}+1.0. '.format(actual_bit_error, error_bits) + msg
        #         self.on_error(reason) # Might NOT exit!!
        
        if True: # Comparison of ONLY the max
            actual_bit_error = float(-0.5*xp.log2(dx2.max()))
            if signed:
                actual_bit_error -= 1 # Remove the sign bit
            
            if actual_bit_error < error_bits:
                reason = 'Error too high, {} is not more than {}. '.format(actual_bit_error, error_bits) + msg
                self.on_error(reason) # Might NOT exit!!
                
            elif x.shape[1] > 10 and actual_bit_error >= (error_bits+1.0):
                reason = 'Error too low, {} is not less than {}+1.0. '.format(actual_bit_error, error_bits) + msg
                self.on_error(reason) # Might NOT exit!!
        
        if isinstance(actual_bit_error, float):
            return actual_bit_error
        else:
            return float(actual_bit_error.min())
    
    # Some small values always compressed to zero, so below ALWAYS fails
    # def test_extra_zeros(self, bits=5, msg=None):
    #     self.xp.random.seed(self.seed ^ 0xff991122)
    #     s = self.shape
    #     u = self.xp.random.rand(*s)
    #     v = self.xp.random.random((1,s[1],1,1)) * 0.5 + 0.5
    #     nzv = self.xp.random.rand(*s) > 0.1
        
    #     bits = self.xp.broadcast_to(self.xp.array([bits]), s).astype('f')
    #     xs = (abs(u*v) * nzv).astype('f')
    #     x_old = xs.copy()
    #     y = fixpoint_multi(xs, bits, bits-1, inplace=False, signed_1bit=False)
    #     x_new = fixpoint_multi_dec(y, bits, bits-1, inplace=False, scale=None)
        
    #     kwargs = {} if msg is None else {'msg':msg}
    #     np.testing.assert_array_equal(cuda.to_cpu(x_old==0)*1, cuda.to_cpu(x_new==0)*1, **kwargs)
        
    
    def test_error_unsigned_sparse_scaled(self, *a, **kw):
        xp = self.xp
        s = self.shape
        xp.random.seed(self.seed ^ 0x12345)
        
        for bits in self.bit_tests:
            actual_bit_errors = []
            for num in range(self.num_tests):
                u = xp.random.rand(*s)
                v = xp.random.random((1,s[1],1,1)) * 1.5
                nzv = xp.random.rand(*s) > 0.1
                x = abs(u*v) * nzv.astype('f')
                msg = 'Unsigned sparse {} bits test {}'.format(bits, num)
                actual_bit_error = self.check_error_scaling(x, bits, msg)
                actual_bit_errors.append(actual_bit_error)
            
            abe = np.array(actual_bit_errors)
            print('  {:2d} bits: [{:.2f}, {:.2f}]  mean {:.2f}'.format(bits, abe.min(), abe.max(), abe.mean()))
                
                
    def test_error_signed_sparse_scaled(self, *a, **kw):
        xp = self.xp
        s = self.shape
        xp.random.seed(self.seed ^ 0xc0ffee)
        
        for bits in self.bit_tests:
            actual_bit_errors = []
            
            for num in range(self.num_tests):
                u = xp.random.rand(*s)*2 - 1
                v = xp.random.random((1,s[1],1,1)) * 1.5
                x = u*v
                msg = 'Signed dense {} bits test {}'.format(bits, num)
                actual_bit_error = self.check_error_scaling(x, bits, msg)
                actual_bit_errors.append(actual_bit_error)
            
            abe = np.array(actual_bit_errors)
            print('  {:2d} bits: [{:.2f}, {:.2f}]  mean {:.2f}'.format(bits, abe.min(), abe.max(), abe.mean()))
                
                
    def test_error_unsigned_dense_scaled(self, *a, **kw):
        xp = self.xp
        s = self.shape
        xp.random.seed(self.seed ^ 0x52984)
        
        for bits in self.bit_tests:
            actual_bit_errors = []
            for num in range(self.num_tests):
                u = xp.random.rand(*s)
                v = xp.random.random((1,s[1],1,1)) * 1.5
                x = abs(u*v)
                msg = 'Unsigned dense {} bits test {}'.format(bits, num)
                actual_bit_error = self.check_error_scaling(x, bits, msg)
                actual_bit_errors.append(actual_bit_error)
            
            abe = np.array(actual_bit_errors)
            print('  {:2d} bits: [{:.2f}, {:.2f}]  mean {:.2f}'.format(bits, abe.min(), abe.max(), abe.mean()))
        

