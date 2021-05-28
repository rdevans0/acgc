
from collections import OrderedDict
import chainer

class VarTracking:
    """ Handles common functionality for all resnet variable tracking """
    def __init__(self, retain=False):
        self.reg_cbs = []
        self.vars = OrderedDict()
        self._retain_vars = retain
    
    def clearvars(self):
        self.vars = OrderedDict()
    
    def retain(self, retain):
        self._retain_vars = retain
        if not self._retain_vars:
            self.vars = OrderedDict()
        
        for f in self.child_iter():
            f.retain(retain)
    
    def reg(self, name, var, override_retain=None):
        var.name = name
        
        if override_retain is None:
            if self._retain_vars:
                self.vars[name] = var
        else:
            if override_retain:
                self.vars[name] = var
                
        return var
    
    def finalize_reg(self):
        """ Discards variables which do not have retain_data set.
            Always keeps the last variable to return
        """
        
        if not self._retain_vars:
            return
        
        names = list(self.vars.keys())
        for name in names[:-1]:
            if self.vars[name].node.data is None:
                del self.vars[name]
        
    
    def child_iter(self):
        if isinstance(self, chainer.ChainList):
            return self.children()
        elif hasattr(self, 'layers'):
            return self.layers
        else:
            return []
    
    def namedvars(self):
        for n,v in self.vars.items():
            yield n,v
            
        for f in self.child_iter():
            for n,v in f.namedvars():
                yield n,v
    
    def custom_namedvars(self, order=['children']):
        assert 'children' in order
        
        for key in order:
            if key == 'children':
                for f in self.child_iter():
                    for n,v in f.namedvars():
                        yield n,v
            else:
                for n,v in self.vars.items():
                    if key in n:
                        yield n,v