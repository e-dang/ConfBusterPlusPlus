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

from runner import Runner


def main():
    parser = argparse.ArgumentParser(description='Perform conformational sampling on the given macrocycle.')

    parser.add_argument('--smiles', type=str, help='The SMILES string of the macrocycle.')
    parser.add_argument('--sdf', type=str, help='The filepath to the input file containing the macrocycles.')
    parser.add_argument('--out', '-o', type=str, help='The .pdb output file path. If file exists, numbers will be '
                        'appended to the file name to produce a unique file path.')

    args = parser.parse_args()

    Runner(args)


if __name__ == "__main__":
    main()
