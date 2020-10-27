#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 16:45:13 2020

@author: isakh
"""





import yaml

def get_params(model: str):
    with open("config.yml", "r") as ymlfile:
        cfg = yaml.load(ymlfile)
    return cfg[model]

