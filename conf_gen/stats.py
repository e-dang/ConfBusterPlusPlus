"""
MIT License

Copyright (c) 2019 e-dang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

github - https://github.com/e-dang
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set(style='ticks', color_codes=True)


def plot_stats(filepaths):
    """
    Plots the run statistics for separate runs on the same macrocycle at different energy cutoffs. The plot contains 5
    graphs in 4 quadrants. The top left quadrant contains two bar graphs showing the runtimes and number of conformers.
    The top right quadrant is a violin plot showing the distributions of energies. The bottom left quadrant is a violin
    plot showing the distributions of RMSDs. The bottom right quadrant is a violin plot of the distributions of ring
    RMSDs.

    Args:
        filepaths (list): A list of strings that specify the different run statistics json files to load data from.
    """

    stats = load_data(filepaths)

    current_palette = sns.color_palette()
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 4, 1)
    ax2 = fig.add_subplot(2, 4, 2)
    ax3 = fig.add_subplot(2, 2, 2)
    ax4 = fig.add_subplot(2, 2, 3)
    ax5 = fig.add_subplot(2, 2, 4)

    # Number of Conformers
    num_confs = np.array([data['num_confs'] for data in stats.values()]).flatten()
    ax1.bar(stats.keys(), num_confs, color=current_palette)
    ax1.set_xlabel('Energy Cutoff (kcal/mol)')
    ax1.set_ylabel('Number of Conformers')

    # Runtimes
    runtimes = np.array([data['time'] for data in stats.values()]).flatten()
    ax2.bar(stats.keys(), runtimes, color=current_palette)
    ax2.set_xlabel('Energy Cutoff (kcal/mol)')
    ax2.set_ylabel('Runtime (sec)')

    # Energy
    cutoffs = np.array([])
    for key, value in stats.items():
        cutoffs = np.append(cutoffs, np.ones(len(value['energies'])) * int(key))

    energies = np.concatenate([np.array(data['energies']) for data in stats.values()])
    df_energy = pd.DataFrame({'Energy Cutoff (kcal/mol)': cutoffs, 'Energy (kcal/mol)': energies})
    sns.catplot(x='Energy Cutoff (kcal/mol)', y='Energy (kcal/mol)',
                kind='violin', inner='stick', cut=0, data=df_energy, ax=ax3)
    sns.catplot(x='Energy Cutoff (kcal/mol)', y='Energy (kcal/mol)',
                kind='violin', inner='box', cut=0, data=df_energy, ax=ax3)

    # RMSD
    cutoffs = np.array([])
    for key, value in stats.items():
        cutoffs = np.append(cutoffs, np.ones(len(value['rmsd'])) * int(key))

    rmsd = np.concatenate([data['rmsd'] for data in stats.values()])
    df_rmsd = pd.DataFrame({'Energy Cutoff (kcal/mol)': cutoffs, 'RMSD (Å)': rmsd})

    sns.catplot(x='Energy Cutoff (kcal/mol)', y='RMSD (Å)',
                kind='violin', inner='stick', cut=0, data=df_rmsd, ax=ax4)
    sns.catplot(x='Energy Cutoff (kcal/mol)', y='RMSD (Å)',
                kind='violin', inner='box', cut=0, data=df_rmsd, ax=ax4)
    ax4.set_ylim(0)

    # Ring RMSD
    cutoffs = np.array([])
    for key, value in stats.items():
        cutoffs = np.append(cutoffs, np.ones(len(value['ring_rmsd'])) * int(key))

    ring_rmsd = np.concatenate([data['ring_rmsd'] for data in stats.values()])
    df_ring_rmsd = pd.DataFrame({'Energy Cutoff (kcal/mol)': cutoffs, 'Ring RMSD (Å)': ring_rmsd})

    sns.catplot(x='Energy Cutoff (kcal/mol)', y='Ring RMSD (Å)',
                kind='violin', inner='stick', cut=0, data=df_ring_rmsd, ax=ax5)
    sns.catplot(x='Energy Cutoff (kcal/mol)', y='Ring RMSD (Å)',
                kind='violin', inner='box', cut=0, data=df_ring_rmsd, ax=ax5)
    ax5.set_ylim(0)

    fig.tight_layout(w_pad=0.5)

    return fig


def load_data(filepaths):
    """
    Loads the json run statistics files specified by filepaths and inserts their data into a dictionary that is indexed
    by the energy cutoff used in the run. This function is not meant to load data from runs on different macrocycles.

    Args:
        filepaths (list): A list of strings that specify the run statistics files to load.

    Returns:
        dict: The run statistics indexed by energy cutoffs.
    """

    stats = {}
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            data = json.load(file)
            stats[str(int(data['parameters']['energy_diff']))] = data

    return stats


filepaths = [os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'examples', 'macrocycle_2_5_0.json'),
             os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'examples', 'macrocycle_2_10_0.json')]
fig = plot_stats(filepaths)
fig.savefig('macrocycle_2.png')
