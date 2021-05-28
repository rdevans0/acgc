#!/usr/bin/env python3

import re
import textwrap
import argparse

from common.compression.quant import Quant
from common.compression.autoquant import AutoQuant

from functools import partial


def add_ae_arguments(parser):
    
    filter_help = ('Filters specifies which layers to apply to the method to '
                   + '(see <filters>).')
    
    def add_arg(name, metavars, help):
        metavars = tuple(metavars) + ('filters',)
        parser.add_argument('--ae_' + name, 
                            nargs=len(metavars),
                            metavar=metavars,
                            action='append',
                            help = help + filter_help)
            
            
    add_arg('quant', ['bits', 'tag'], 
            ('Fixpoint quantization of the range [-1, 1). '
             ))
    
            
    add_arg('quantz', ['bits', 'tag'], 
            ('Fixpoint quantization of the range [-1, 1) with zero value compression '
             ))
    
    add_arg('autoquant', ['bits', 'tag'], 
            ('Automatic fixpoint quantization with the target bits.'
             ))
    
    
    add_arg('autoquantz', ['bits', 'tag'], 
            ('Automatic fixpoint quantization with the target bits and zero value compression.'
             ))

def parse_ae_arguments(parser, args, rng, layers, printfn=print):
    scrambler_map, filterspec_map = _parse_ae_arguments(
        parser, args, rng, layers)
    
    print_helper_summary(scrambler_map, filterspec_map, printfn)
    print_helper_map(scrambler_map, layers, printfn=printfn)
    
    return scrambler_map, filterspec_map


def _parse_ae_arguments(parser, args, rng, layers):
    """ Decode the helpers for each argument and put them into a map of layer->helper or None"""
    
    scrambler_classes = {
        'quant': partial(Quant, zvc=False),
        'quantz': partial(Quant, zvc=True),
        'autoquant': partial(AutoQuant, zvc=False),
        'autoquantz': partial(AutoQuant, zvc=True),
    }
    
    scrambler_map = dict( (k,None) for k in layers)
    filterspec_map = {}
    
    for k,v in vars(args).items():
        if k.startswith('ae_') and v:
            k2 = k.replace('ae_','')
            if k2 not in scrambler_classes:
                raise Exception(
                    'Could not find {} in scrambler_classes'.format(k2))
    
    for name, Scrambler, in scrambler_classes.items():
        ae_args_all = getattr(args, 'ae_'+name)
        if ae_args_all is None:
            continue  # not set on command line
            
        for ae_args in ae_args_all:
            scrambler_args = ae_args[:-1]
            filters = ae_args[-1]
            
            s = Scrambler(*scrambler_args)
            
            filterspec_map[s] = filters
            filter_apply_wo(scrambler_map, filters, s)  # scrambler_map[<filter>] = s
            
    return scrambler_map, filterspec_map
       

def print_helper_summary(helper_map, filterspec_map, printfn=print):
    for helper, filterspec in filterspec_map.items():
        count = sum(v == helper for v in helper_map.values())
        printfn('# applied error: {0:<15}\t{1} ({2})'.format(helper.name, filterspec, count))
        helper.print_settings(printfn)

def print_helper_map(helper_map, sorted_layers, printfn=print):
    """ Nicely prints the helper_map using printfn. Order specified by sorted_layers """
    
    
    printfn('\n# Error Map')
              
    if all(v is None for v in helper_map.values()):
        printfn('#  No layers selected!')
        return
    
    # Cluster names according to prefix
    clustered = [ [name if name is not None else 'None-'] for name in reversed(sorted_layers)]
    head = 0
    get_prefix = lambda n: n.split('-')[0]
    get_postfix = lambda n: '-'.join(n.split('-')[1:])
    while head < len(clustered):
        prefix = get_prefix(clustered[head][0])
        tail = head + 1
        while tail < len(clustered):
            name = clustered[tail][0]
            if get_prefix(name) != prefix:
                break
            clustered[head] += [name]
            clustered.pop(tail)
        head = head+1
    
    for cluster in clustered:
        
        if cluster == ['None-']:
            cluster = [None]
        
        if len(cluster) == 1:
            layer = cluster[0]
            h = helper_map[layer]
            printfn('#   {0: <16} -> {1}'.format(str(layer), ' -- ' if h is None else h.detail))
            continue
        
        statuses = [' -- ' if helper_map[layer] is None else helper_map[layer].detail for layer in cluster]
        
        if all( status == statuses[0] for status in statuses ):
            prefix = get_prefix(cluster[0])
            postfix = '-[' + ''.join(get_postfix(layer) for layer in cluster) +']'
            printfn('#   {0: <16} -> {1}'.format(prefix + postfix, statuses[0]))
        else:
            for layer in cluster:
                h = helper_map[layer]
                printfn('#   {0: <16} -> {1}'.format(str(layer), ' -- ' if h is None else h.detail))
    

def filter_select(filters, layers, parser_error=lambda *a, **kw: None, 
                  write_once=False, empty_value=False, select_value=True):
    """ Takes filters and selects which layers they select 
    
    Parameters
    ----------
    filters : str
        Comma separated filters. Commas specify and AND operation. 
        E.g. R1,R2 is the same as R12
    
    layers : list(str)
        Available layers in the network.
    
    parser_error : callable(must exit)
        Called before issuing a parser error as a ValueError (can override 
        default error callback with parser.error)
    
    write_once : bool, default=False
        Forces an error if we attempt to set select_map to True when it is 
        already set. I.E. we can only select each layer ONCE.
    
    empty_value : object, default=False
        Initial value of selection_map, specifies an empty slot (ADVANCED!)
        
    select_value : object, default=False
        Value of selection_map once it is selected (ADVANCED!)
    
    Returns
    -------
    selection_map : map(str to bool)
        Map of True/False with Trues corresponding to the layers selected by filters.
    
    """
    
    # Selection map and setter
    selection_map = dict( (k,empty_value) for k in layers)
    
    _filter_apply_core(selection_map, filters,  
                       parser_error=parser_error, write_once=write_once, 
                       empty_value=empty_value, select_value=select_value)
    
    return selection_map
    

def filter_apply(selection_map, filters, value, parser_error=lambda *a, **kw: None, 
                 write_once=False, empty_value=None):
    
    _filter_apply_core(selection_map, filters,  
                       parser_error=parser_error, write_once=write_once, 
                       empty_value=empty_value, select_value=value)
    
    return selection_map

def filter_apply_wo(selection_map, filters, value, parser_error=lambda *a, **kw: None, ):
    
    _filter_apply_core(selection_map, filters,  
                       parser_error=parser_error, write_once=True, 
                       empty_value=None, select_value=value)
    
    return selection_map
        


def _filter_apply_core(selection_map, filters, parser_error=lambda *a, **kw: None, 
                        write_once=False, empty_value=False, select_value=True):
    
    # Some cleanup
    all_layers = sorted(selection_map.keys(), key=str)
    none_layers = list(filter(lambda l: l == None, all_layers))
    str_layers = list(filter(lambda l: l != None, all_layers))
    
    expanded = dict((v[0],v) for v in ['c','r','p','n','f','d','s','b','x','i'])
    
    def raise_parser_error(*a, **kw):
        parser_error(*a, **kw) # If it's defined, and exits, great!
        raise ValueError(*a, **kw) # Issue this if parser_error doesn't exit
    
    def select(layers, filt):
        for l in layers:
            if selection_map[l] == empty_value:
                selection_map[l] = select_value
            elif write_once:
                raise_parser_error('The filter {} overlaps with a previous filter at layer {}'.format(filt, l))
    
    # Do for each filter
    for filt in filters.split(','):
        
        m = re.findall('([ABFOWR]+)(\d+)?(?:_(\d+))?(?:_(\d+))?([crpnfdsbxi]*)', filt)
        if len(m) != 1:
            raise_parser_error('Filter "{}" is malformed at {}'.format(filters, filt))
        
        category, groups, subgroups, subsubgroups, layer_types = m[0]
        
        # Make all items unique and sort them
        category, groups, subgroups, subsubgroups, layer_types =\
                map(lambda x: sorted(set(x)), 
                    (category, groups, subgroups, subsubgroups, layer_types))
                
        # Closer to regex format
        groups = ''.join(groups)
        subgroups = ''.join(subgroups)
        subsubgroups = ''.join(subsubgroups)
        layer_types = '|'.join(expanded[t] for t in layer_types)
        
        if 'R' in category and 'W' in category:
            raise_parser_error('R and W (res and wide) cannot be used together. Specify blocks in Resnet and Widenet, respectively')
        
        # Do the selection
        if 'A' in category:
            if groups or subgroups or layer_types: 
                raise_parser_error('A (all) can only be specified on its own (e.g. A3f doesnt work)')
            select(all_layers, filt)
            
        if 'O' in category:
            select(none_layers, filt)
        
        if 'F' in category:
            expr = 'final'
            expr += '[%s]'%groups if groups else '\d'
            if layer_types:
                expr += '-(%s)'%layer_types
            select(filter(lambda l: re.match(expr, l), str_layers), filt)
        
        if 'B' in category:
            expr = 'block'
            expr += '[%s]'%groups if groups else '\d'
            expr += '_'
            expr += '[%s]'%subgroups if subgroups else '\d'
            if layer_types:
                expr += '-(%s)'%layer_types
            select(filter(lambda l: re.match(expr, l), str_layers), filt)
        
        if 'W' in category or 'R' in category:
            expr = 'wide' if 'W' in category else 'res'
            expr += '[%s]'%groups if groups else '\d'
            expr += '_'
            expr += '[%s]'%subgroups if subgroups else '\d+'
            expr += '_'
            expr += '[%s]'%subsubgroups if subsubgroups else '\d'
            if layer_types:
                expr += '-(%s)'%layer_types
            select(filter(lambda l: re.match(expr, l), str_layers), filt)
    
    
    return selection_map


filterspec_help = textwrap.dedent('''\
        Layer Filters <filters>:
        
            This is a single, or comma separated list of filters. 
            Each filter can specify one or more layers.
            
            The syntax is (in python regex format):
            
                [ABFO]+(\d+)?(_\d)?[crpnfd]*
                
                [ABFO]+     Group categories (all, block, final, other)
                (\d+)?      Group number(s) (omitting selects all groups)
                (_\d)?      Subgroup number (omitting this selects the entire group[s])
                [crpnfd]*   Layer types:
                                c - convolutional
                                r - relu
                                p - 2d max pooling
                                n - batch normalization
                                f - fully connected
                                d - dropout
            
            Examples (* is wildcard):
                
                A           *
                B3c         block3*-cv (convolutional)
                B123_1      block1_1-*, block2_1-*, and block3_1-*
                B_1r        block*_1-relu (if available)
                F1f         final1-fc (fully connected)
                F           final*
                B           block*
                B1rc        block1_*-relu and block1_*-cv
                Bc,B1_1r    block*-conv and block1_1-relu
             
                
        ''')

class _FilterSpecFormatter(argparse.RawTextHelpFormatter):
    def _split_lines(self, text, width):
        sp = text.splitlines()
        return sum((textwrap.wrap(t, width) if t else [' '] for t in sp), [])
        
FilterArgumentParser = partial(argparse.ArgumentParser,
                              formatter_class=_FilterSpecFormatter,
                              epilog=filterspec_help)
