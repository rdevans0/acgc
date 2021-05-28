
import sys
import os

from chainer.training.extensions import PrintReport
from chainer.training.extensions import log_report as log_report_module


def begin_output(args, argv, report=None, file='train.py', printfn=print, 
                 first=('model','epoch','batchsize','resume', 'out')):
    """ print_header, but also opens report.out """
    
    if not os.path.exists(args.out): 
        os.mkdir(args.out)
        
    if report is None:
        report_fid = open(os.devnull,'w')
    elif isinstance(report, str):
        report_file = os.path.join(args.out, 'report.txt')
        open(report_file,'w').close()
        report_fid = open(report_file, 'a')
    else:
        report_fid = report
    
    # Print header into report
    logprint = print_header(args, argv, log=report_fid, first=first, 
                            preamble=file)
    
    # Print commands into cmd.sh
    def cmd_join(*vals):
        ret = []
        for v in vals:
            ret += ['"%s"'%v if ' ' in vals else v]
        return ' '.join(ret)
    cmd_str = '\n'.join([
                cmd_join('pushd', os.getcwd()),
                cmd_join('python3', file, *argv),
                'popd']) + '\n'
    
    open(os.path.join(args.out, 'cmd.sh'), 'w').write(cmd_str)
    
    # Print arguments into args.py
    args_sorted = sorted(vars(args).items())
    arg_str = ('{'
               + '\n'.join('   {:20}: {},'.format("'%s'"%k, repr(v)) for k,v in args_sorted)
               + '\n}')
    
    open(os.path.join(args.out, 'args.py'), 'w').write(arg_str)
    
    return logprint, report_fid


def print_header(args, argv, preamble='ILSVRC2012', printfn=print, 
                 log=open(os.devnull,'w'), 
                 first=('model','epoch','batchsize','resume', 'out')):
    """ Prints the arguments and header, and returns a logging print function """
        
    if log is None:
        log = open(os.devnull,'w')
    
    def logprint(*args, file=log, **kwargs):
        if printfn:
            printfn(*args, **kwargs)
        print(*args, file=file, **kwargs)
        file.flush()
    
    vargs = vars(args)
    args_sorted = sorted(vargs.items())
    logprint('{' + ', '.join("'{}':{}".format(k,repr(v)) for k,v, in args_sorted) + '}')
    logprint(' '.join(argv))
    logprint('')
    logprint(preamble)
    logprint('')
    logprint('Arguments: ')
    
    def print_arg(arg):
        logprint('   {:20}: {},'.format("'%s'"%arg,repr(vargs[arg])))
    
    for arg in first:
        print_arg(arg)
    logprint('')
    for arg,_ in args_sorted:
        if arg in first:
            continue
        print_arg(arg)
    
    logprint('')
    logprint('Iterations / Epoch: {}'.format(1280000 // args.batchsize))
    logprint('')
    
    return logprint


class FileReport(PrintReport):

    """Removes weird control characters from PrintReport 
    
    Allows for writing a PrintReport to a nicely formatted file.
    """

    def __init__(self, entries, log_report='LogReport', out=sys.stdout):
        super(FileReport, self).__init__(entries, log_report=log_report, out=out)

    def __call__(self, trainer):
        out = self._out

        if self._header:
            out.write(self._header)
            self._header = None

        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            self._print(log[log_len])
            log_len += 1
            out.flush()
        self._log_len = log_len