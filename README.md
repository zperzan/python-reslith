# python-reslith

This repository contains code and examples for implementing a resistivity-lithology transform in python. In particular, we use the bootstrapping approach of [Knight et al. (2018)](https://doi.org/10.1111/gwat.12656) to transform a realization of resistivity derived from a towed transient electromagnetic (tTEM) survey to a value referred to as "coarse fraction".  The `input_files` directory contains example input files used to build the resistivity-lithology transform, including cone penetrometer test (CPT) logs and a subset of the tTEM data.

The jupyter notebook `ResistivityLithologyTransform.ipynb` walks through the example application and contains markdown cells that describe each step in the process. 

## Python dependencies
- `numpy`
- `matplotlib`
- `scipy`
- `tqdm`

