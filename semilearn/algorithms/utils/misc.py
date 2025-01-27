# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import torch.nn as nn 


class SSL_Argument(object):
    """
    Algrithm specific argument
    """
    def __init__(self, name, type, default, help=''):
        """
        Model specific arguments should be added via this class.
        """
        self.name = name
        self.type = type
        self.default = default
        self.help = help


def str2bool(v):
    """
    str to bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(delimiter=';', dtype=int):
    """
    str to list of specified type
    """
    def parser(v):
        list_of_strings = v.split(delimiter)
        return list(map(dtype, list_of_strings))

    return parser
