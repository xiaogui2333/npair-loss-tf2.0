#!/usr/bin/env python
"""
Print-related utility functions.

History
  create  -  Feng Zhou (zhfe99@gmail.com), 2016-05
"""
import os
import sys
import logging
import time

# global variable
log_file = None
lPr = None
lMaPr = None
nmPrs = None
ticPrs = None
ticPr0s = None
nRepPrs = None
scaPrs = None


def init(lMa, log_path=None):
    """
    Set the promption level.

    Input
      lMa      -  maximum level, 0 | 1 | 2 | ...
      logPath  -  log file path, {None} | ...
    """
    global lPr, lMaPr, nmPrs, ticPrs, ticPr0s, nRepPrs, scaPrs, logFile

    # level
    lPr = 0
    lMaPr = lMa

    # list
    nMa = 10
    nmPrs = list(range(nMa))
    ticPrs = list(range(nMa))
    ticPr0s = list(range(nMa))
    nRepPrs = list(range(nMa))
    scaPrs = list(range(nMa))

    # log
    if log_path is not None:
        log_file = log_path
        log_set(log_path, haNew=True)


def pr(form, *args):
    """
    Prompt the information specified in the parameters.

    Input
      form   -  format
      *args  -  object list
    """
    # variables
    global lPr, lMaPr

    if lPr < lMaPr:
        for l in range(lPr + 1):
            sys.stdout.write('-')
        if len(args) == 0:
            print(form)
            sys.stdout.flush()
            if log_file is not None:
                logging.info(form)
        else:
            print(form % args)
            if log_file is not None:
                logging.info(form % args)


def in_func(nm, form="", *args):
    """
    Start a propmter for displaying information.

    Input
      nm        -  name
      form      -  format
      varargin  -  object list
    """
    # variables set in "init()"
    global lPr

    # init
    if not 'lPr' in globals():
        init(3)

    # print
    if form == "":
        pr('%s', nm)
    else:
        pr('%s: ' + form, nm, *args)

    # self add
    lPr = lPr + 1


def out_func():
    """
    Stop a propmter for function.
    """
    # variables set in "prSet.m"
    global lPr

    # delete
    lPr = lPr - 1


def in_out_func(nm, form="", *args):
    """
    Prompt the information specified in the parameters.

    Input
      form   -  format
      *args  -  object list
    """
    in_func(nm, form=form, *args)
    out_func()


def in_loop(nm, nRep, sca):
    """
    Start a propmter for displaying information about loop.

    Input
      nm    -  name
      nRep  -  #steps
      sca   -  scale of moving, (0, 1) | 1 | 2 | ...
    """
    # variables set in "prSet()"
    global lPr, nmPrs, ticPrs, ticPr0s, nRepPrs, scaPrs

    # insert
    nmPrs[lPr] = nm
    ticPrs[lPr] = time.time()
    ticPr0s[lPr] = ticPrs[lPr]
    nRepPrs[lPr] = nRep

    # scaling
    if sca < 1:
        sca = round(nRep * sca)
    if sca == 0:
        sca = 1
    scaPrs[lPr] = sca

    # print
    pr('%s: %d %d' %(nm, nRep, sca))

    lPr = lPr + 1


def out_loop(nRep):
    """
    Stop a propmter for counting.

    Input
      nRep  -  #steps
    """
    # variables defined in "prSet.m"
    global lPr, nmPrs, ticPrs, ticPr0s, nRepPrs, scaPrs

    lPr = lPr - 1

    # time
    t = time.time() - ticPr0s[lPr]

    # print
    pr('%s: %d iters, %.2f secs' %(nmPrs[lPr], nRep, t))


def loop(iRep):
    """
    Prompt information of a counter.

    Input
      iRep  -  current step

    Output
      is_pr  -  print flag
    """
    # variables defined in "prSet()"
    global lPr, nmPrs, ticPrs, nRepPrs, scaPrs

    is_pr = False
    lPr = lPr - 1
    if (iRep != 0 and iRep % scaPrs[lPr] == 0) or (iRep == nRepPrs[lPr]):
        is_pr = True

        # time
        t = time.time() - ticPrs[lPr]

        # print
        pr('%s: %d/%d, %.2f secs' %(nmPrs[lPr], iRep, nRepPrs[lPr], t))

        # re-start a timer
        ticPrs[lPr] = time.time()

    lPr = lPr + 1

    return is_pr


def log_set(log_path, level='info', renew=True, rec_date=True, rec_level=False, ha_new=False):
    """
    Set up logger.

    Input
      logPath    -  log path
      level      -  level, {'info'} |
      renew      -  flag of creating a new log, {True} | False
      rec_date   -  flag of recording time, {True} | False
      rec_level  -  flag of recording time, True | {False}
      ha_new     -  flag of creating all new handler, True | {False}
    """
    global log_file

    from . import file
    log_file = log_path

    # create the log folder if necessary
    log_fold = os.path.dirname(log_path)
    file.mkdir(log_fold)

    # delete old log
    if renew:
        file.rm(log_path)

    # remove onld handler
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # level
    if level == 'info':
        level = logging.INFO
    else:
        raise Exception('unknown level: {}'.format(level))

    # format
    format = '%(message)s'
    if rec_level:
        format = '%(levelname)s ' + format
    if rec_date:
        format = '%(asctime)s ' + format

    # set
    logging.basicConfig(level=level, filename=log_path,
                        format=format,
                        datefmt="%y-%m-%d %H:%M:%S")

    # record basic information
    log('$user {}, $host_name {}'.format(os.environ['USER'], os.uname()[1]))


def log(form, level='info', *args):
    """
    Output to log file.

    Input
      form   -  format
      level  -  level, {'info'} |
      *args  -  object list
    """
    if level == 'info':
        if len(args) == 0:
            logging.info(form)
        else:
            logging.info(form % args)

    elif level == 'error':
        if len(args) == 0:
            logging.error(form)
        else:
            logging.error(form % args)

    else:
        raise Exception('unknown level: {}'.format(level))
