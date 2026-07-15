# SPDX-License-Identifier: Apache-2.0
"""Five-entry runtime cache layout for the PD decode host-offload path.

  [0] resident K block view
  [1] resident V block view
  [2] separately composed real-indexer cache (or an empty sentinel)
  [3] resident K row view
  [4] resident V row view
"""

OFFLOAD_MAIN_K = 0
OFFLOAD_MAIN_V = 1
OFFLOAD_INDEXER_K = 2
OFFLOAD_RESIDENT_K = 3
OFFLOAD_RESIDENT_V = 4
OFFLOAD_TUPLE_LEN = 5
