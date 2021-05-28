

class Scrambler:
    def init_detail(self, name, **params):
        self.name = name
        self._params = params
        param_str = ', '.join('{}={!r}'.format(*it) for it in params.items())
        self.detail = '{}({})'.format(self.name, param_str)
        
        self._sign_cache = {}
    
    def get_signed(self, x, var_name):
        if var_name not in self._sign_cache:
            self._sign_cache[var_name] = (x.min() < 0)
        return self._sign_cache[var_name]
        
    def print_settings(self, printfn=print):
        for k,v in self._params.items():
            if k != 'tag':
                printfn('#   {:>8s}: {}'.format(k,v))
        
    def scramble_setup(self):
        self._act_count = 0
        self._bit_count = 0
        self._intr_bit_count = 0
    
    def add_counts(self, act_count, bit_count, intr_bit_count):
        t = self.tag
        act_count[t] += self._act_count
        bit_count[t] += self._bit_count
        intr_bit_count[t] += self._intr_bit_count
    
    def scrambler(self, x_var, var_name=None, **kwargs):
        raise NotImplementedError()