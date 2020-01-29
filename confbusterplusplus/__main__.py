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

import argparse

from confbusterplusplus.runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Perform conformational sampling on the given macrocycle. Macrocycles '
                                     'can be input via SMILES strings or .sdf files. The conformers are output to a '
                                     '.pdb file specified by the --out option and the run statistics are output in a '
                                     'separate json file with the same base filename as the .pdb file. The input '
                                     'macrocycles and output filepath must be specified, all other command line '
                                     'arguments are optional. Program exits with code 0 upon a successful run, code 1 '
                                     'when there an invalid argument is given to an I/O option, and code 2 when there '
                                     'is an invalid given to any other type of option. Written by Eric Dang, github - '
                                     'https://github.com/e-dang')

    parser.add_argument('--smiles', type=str, nargs='*', help='The SMILES string(s) of the macrocycle(s) separated by a'
                        ' single space.')
    parser.add_argument('--sdf', type=str, help='The filepath to the .sdf input file containing the macrocycles.')
    parser.add_argument('--repeats_per_cut', '-r', type=int, help='The number of times the linear oligomer is subjected'
                        ' to random embedding, the genetic algorithm, and subsequent rotamer search. Defaults to 5.')
    parser.add_argument('--num_confs_genetic', '-N', type=int, help='The number of conformers to generate using the '
                        'genetic algorithm. Defaults to 50.')
    parser.add_argument('--num_confs_rotamer_search', '-n', type=int, help='The maximum number of conformers to accept '
                        'during the rotamer search. Defaults to 5.')
    parser.add_argument('--force_field', '--ff', type=str, choices=['MMFF94', 'MMFF94s'], help='The force field to use '
                        'for energy minimizations. Defaults to MMFF94s.')
    parser.add_argument('--dielectric', '--eps', type=float, help='The dielectric constant to use during energy '
                        'minimizations. Defaults to 1.0.')
    parser.add_argument('--score', '-s', type=str, choices=['energy', 'rmsd'], help='The score to use for the genetic '
                        'algorithm. Defaults to energy.')
    parser.add_argument('--min_rmsd', '--rmsd', type=float, help='The minimum RMSD that two conformers must be apart in '
                        'order for both to be kept. Defaults to 0.5.')
    parser.add_argument('--energy_diff', '-e', type=float, help='The maximum energy difference between the lowest energy '
                        'conformer and the highest energy conformer in the final set of conformers. Defaults to 10.')
    parser.add_argument('--small_angle_gran', type=int, help='The granularity with which dihedral angles are rotated '
                        'during the fine grained portion of rotamer optimization. Defaults to 5.')
    parser.add_argument('--large_angle_gran', type=int, help='The granularity with which dihedral angles are rotated '
                        'during the coarse grained portion of rotamer optimization. Defaults to 15.')
    parser.add_argument('--clash_threshold', '-c', type=float, help='The threshold used to identify clashing atoms. '
                        'Defaults to 0.9.')
    parser.add_argument('--distance_interval', '-d', type=float, nargs=2, help='The range of distances that the two atoms'
                        ' of the cleaved bond must be brought to during rotamer optimization in order for that'
                        ' conformer to be accepted. Defaults to 1.0 - 2.5')
    parser.add_argument('--num_threads', '-t', type=int, help='The number of threads to use when embedding and doing '
                        'global energy minimizations. If set to 0, the maximum threads supported by the computer is '
                        'used. Defaults to 0.')
    parser.add_argument('--max_iters', '-i', type=int, help='The maximum number of iterations used for alignments and '
                        'energy minimizations, however more iterations are performed in the case of energy minimization'
                        ' if convergence is not reached by the end of these iterations (see optimize_confs() for '
                        'details). Defaults to 1000.')
    parser.add_argument('--num_embed_tries', type=int, help='The number of tries to perform embedding with. Defaults '
                        'to 5.')
    parser.add_argument('--out', '-o', type=str, help='The .pdb output filepath. If file exists, numbers will be '
                        'appended to the file name to produce a unique filepath.')

    args = parser.parse_args()

    Runner(args)


if __name__ == "__main__":
    main()
