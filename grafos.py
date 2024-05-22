# -*- coding: utf-8 -*-
"""
Grafos

@author: hector@bith.net
@date:   04/2023
"""
import copy
import enum
import random
import logging
import math
import argparse
import numpy as np
import networkx as nx
import sys
import os
import matplotlib.pyplot as plt


def grafo(nodos):
    g = nx.barabasi_albert_graph(nodos, 3)
    pos = nx.spring_layout(g)
    nx.draw(g,pos,with_labels=True)
    # plt.show()
    return g



