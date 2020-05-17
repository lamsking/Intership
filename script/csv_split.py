#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 01:03:09 2019

@author: inra-cirad
"""


import pandas as pd
import numpy as np

df = pd.read_csv('/home/inra-cirad/Bureau/apprentissage/gregory_folder/FichierAll_intersection.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]