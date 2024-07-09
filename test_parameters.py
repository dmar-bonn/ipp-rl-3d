"""
MIT License from https://github.com/marmotlab/CAtNIPP/

Copyright (c) 2022 MARMot Lab @ NUS-ME

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
"""

INPUT_DIM = 6 # 7 if including binary
EMBEDDING_DIM = 128
USE_GPU = False
USE_GPU_GLOBAL = False # True if GPU available
NUM_GPU = 1
NUM_META_AGENT = 1
GAMMA = 1
FOLDER_NAME = 'k20_novec_final'
model_path = f'model/{FOLDER_NAME}'
result_path = f'result/{FOLDER_NAME}'

SEED = 1
NUM_TEST = 1
TRAJECTORY_SAMPLING = True
PLAN_STEP = 7
NUM_STEP = 3
NUM_SAMPLE_TEST = 4 # do not exceed 99
SAVE_IMG_GAP = 1
SAVE_CSV_RESULT = False
SAVE_TRAJECTORY_HISTORY = False
SAVE_TIME_RESULT = False

BUDGET_RANGE = (9.99999, 10)
SAMPLE_SIZE = 350
K_SIZE = 20
SAMPLE_LENGTH = 0.2 # 0/None: sample at nodes

TEST_TYPE = 'random'
FACING_ACTIONS = ['F', 'B', 'L', 'R']
TREE_BINARY = False
DIST_SAMPLE = True
logs_path = f'result/{FOLDER_NAME}'
csv_path = f'CSVs/{TEST_TYPE}'
