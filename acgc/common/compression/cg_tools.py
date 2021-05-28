""" This file deals with manipulations of chainer computation graphs 

The chainer computation graph format is forward only
    
                              +--> Variable
                 weakrefs     |
    FunctionNode  --->  VariableNode --+--> FunctionNode
                                       |
                                       ---> FunctionNode 

FunctionNode:
    .inputs
        VariableNodes used in this function
    .outputs       
        WeakRefs to successive VariableNodes that are created

VariableNode
    .creator_node
        FunctionNode predecessor that created this
        
    .get_variable_or_none()
        Get real Variable (not Node)
         

Goals for this module:
    Iterate over all Variables in a cg
    Select a Variable from the cg by name
    Get the successor functions, and variables of a variable by name
                       
"""
from os.path import dirname
import heapq

from chainer import Variable, Parameter

from collections import namedtuple

# Convenience wrapper for commonly used function properties
FunctionProps = namedtuple('FunctionProps', 'func, label, path, in_params, out_vars')

class CGraph:
    """ A computation graph. 
    Note, modifies and saves the provided chainer graph! 
    """
    
    def __init__(self, model, final_node):
        self.var_nodes = self.func_nodes = None
        self.var_dict = self.succ_dict = None
        self.genetate_paths(model)
        self.load_graph(final_node)
    
    def genetate_paths(self, model):
        # Note these are ids of the PARAMETER not the NODE
        path_dict = dict((id(v), dirname(p)) for p, v in model.namedparams())
        self.path_dict = path_dict
        
    def load_graph(self, final_node):
        var_nodes, func_nodes = get_nodes(final_node, discard_params=True)
        
        # Lookup for variables by name
        var_dict = dict()
        for v in var_nodes:
            if v.name in var_dict:
                raise Exception('Name {} was seen twice, '.format(v.name)
                                + 'names are not unique!')
            var_dict[v.name] = v
        
        # Lookup for successors by name
        succ_dict = dict((v.name, []) for v in var_nodes)
        for f in func_nodes:
            outputs = [ref() for ref in f.outputs]
            tmp = [v.get_variable_or_none() for v in f.inputs]
            in_params = dict((p.name, p) for p in tmp if isinstance(p, Parameter))
            out_vars = [v for v in outputs if not isinstance(v, Parameter)]
            
            assert len(out_vars) == 1, (
                'Assert is here to notify me when we encounter a function '
                + 'with more than one output. Should be safe to remove.')
            
            if in_params:
                any_param = list(in_params.values())[0]
                path = self.path_dict[id(any_param)]
            else:
                path = None
                
            succ = FunctionProps(f, f.label, path, in_params, out_vars)
            
            for v in f.inputs:
                if v.name in succ_dict:
                    succ_dict[v.name] += [succ]
        
        for name, succs in succ_dict.items():
            if not succs:
                raise Exception('No successors found for {}'.format(name))
            
        
        self.var_nodes = var_nodes
        self.var_dict = var_dict
        self.succ_dict = succ_dict
        self.func_nodes = func_nodes
    
    
    def namedvars(self, ignore_missing_refs=True, return_nodes=False):
        num_missing = 0
        for node in self.var_nodes:
            if return_nodes:
                var = node
                is_missing = (node.data is None)
            else:
                var = node.get_variable_or_none()
                is_missing = (var is None)
                
            if is_missing:
                num_missing += 1
            else:
                yield (node.name, var)
        
        # Exception handling (no yields)
        if ignore_missing_refs and num_missing >= len(self.var_nodes)-1:
            raise Exception('Graph out of scope: Nearly all refs missing missing.')
            
        elif not ignore_missing_refs and num_missing:
            for node in self.var_nodes:
                var = node.get_variable_or_none()
                if var is None:
                    print('Variable ' + str(node.name)
                          + ' has a reference that is no longer valid')
            
            raise Exception('Exiting due to missing references, see above')
    
    
    def successors(self, var):
        name = var.name if isinstance(var, Variable) else var
        return self.succ_dict[name]

    


def get_nodes(final_node, discard_params=True):
    """ Reimplements backward_var_iter_nodup in a sane way """
    var_nodes = []
    func_nodes = []
    seen_ids = set()

    # We re-do backward var iter here, with some changes
    cand_funcs = []
    seen = set()
    
    def add_cand(cand):
        if cand not in seen:
            # Negate since heapq is min-heap
            heapq.heappush(cand_funcs, (-cand.rank, len(seen), cand))
            seen.add(cand)
            
    add_cand(final_node.creator_node)
    seen_ids.add(id(final_node))
    
    while cand_funcs:
        _,_,func = heapq.heappop(cand_funcs)
        inputs = func.inputs
        target_inputs = [x for x in inputs if x.requires_grad]
        if not target_inputs:
            continue
        
        if id(func) not in seen_ids:
            func_nodes += [func]
            seen_ids.add(id(func))
            
        for x in target_inputs:
            if discard_params and x.creator_node is None:
                if isinstance(x.get_variable_or_none(), Parameter):
                    continue # Discard param dead ends
            
            if id(x) not in seen_ids:
                var_nodes += [x]
                seen_ids.add(id(x))
                
            if x.creator_node is not None:
                add_cand(x.creator_node)
    
    return var_nodes, func_nodes

def backward_var_iter(start):
    ''' An iterator for going down the backprop chain '''
    cand_funcs = []
    seen = set()
    
    def add_cand(cand):
        if cand not in seen:
            # Negate since heapq is min-heap
            heapq.heappush(cand_funcs, (-cand.rank, len(seen), cand))
            seen.add(cand)
            
    add_cand(start.creator_node)
    
    while cand_funcs:
        rank, _, func = heapq.heappop(cand_funcs)
        inputs = func.inputs
        target_inputs = [x for x in inputs if x.requires_grad]
        if not target_inputs:
            continue
        for x in target_inputs:
            if x.creator_node is not None:
                yield (-rank, func, x)
                add_cand(x.creator_node)


def backward_var_iter_nodup(start):
    """ Same as backward_var_iter, with no duplicate variables (by ID) """
    seen = set()
    seen.add(id(start))
    
    for rank,func,var in backward_var_iter(start):
        if id(var) in seen:
            continue
        yield (rank,func,var)
        seen.add(id(var))

def backward_func_iter_nodup(start):
    ''' An iterator for going down the backprop chain '''
    cand_funcs = []
    seen = set()
    
    def add_cand(cand):
        if cand not in seen:
            # Negate since heapq is min-heap
            heapq.heappush(cand_funcs, (-cand.rank, len(seen), cand))
            seen.add(cand)
            
    add_cand(start.creator_node)
    
    while cand_funcs:
        rank, _, func = heapq.heappop(cand_funcs)
        inputs = func.inputs
        target_inputs = [x for x in inputs if x.requires_grad]
        
        for x in target_inputs:
            if x.creator_node is not None:
                add_cand(x.creator_node)
        
        yield(func)
        
        
def get_dep_funcs(start, var_name):
    """ Functions dependent on this variable in a computation graph"""
    func_nodes = []
    
    # Search for all uses of this variable
    for _,f,_ in backward_var_iter(start):
        if var_name in [v.get_variable().name for v in f.inputs]:
            func_nodes.append(f)
    
    if not len(func_nodes):
        raise ValueError('Could not find variable with name %s in graph'%var_name)
    
    return func_nodes