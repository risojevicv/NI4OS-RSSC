#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Various utility functions.

Created on Tue Oct 06 12:12:55 2020

@author: vlado
"""

import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def is_valid_file(parser, arg):
    """
    Check if arg is a valid file/dir that already exists on the file system.

    Parameters
    ----------
    parser : argparse object
    arg : str

    Returns
    -------
    arg
    """
    arg = os.path.abspath(arg)
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg

def get_parser():
    """Get parser object."""

    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data",
                        dest="data",
                        type=lambda x: is_valid_file(parser, x),
                        help="Path to the dataset.",
                        default="/home/vlado/dl/data/NWPU-RESISC45",
                        required=False)
    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        type=int,
                        help="Batch size",
                        default=10,
                        required=False)
    parser.add_argument("-a", "--arch",
                        dest="arch",
                        help="Convnet architecture.",
                        default="resnet50",
                        required=False)
    parser.add_argument("-w", "--weights",
                        dest="weights",
                        help="Path to the weights file.",
                        default=None,
                        required=False)
    trainable = parser.add_mutually_exclusive_group(required=False)
    trainable.add_argument("--trainable",
                        dest="trainable",
                        help="Use trainable model.",
                        default=False,
                        action="store_true")
    trainable.add_argument("--no-trainable",
                        dest="trainable",
                        help="Use frozen model.",
                        default=True,
                        action="store_false")
    parser.add_argument("-l", "--lr",
                        dest="lr",
                        type=float,
                        help="Learning rate.",
                        default=1e-3,
                        required=False)
    return parser
