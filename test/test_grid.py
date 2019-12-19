from itertools import product

import pygrid.grid_cpp
import pytest
from .utils import dtypes, devices, tensor

tests = [{
    'pos': [2, 6],
    'size': [5],
    'cluster': [0, 0],
}, {
    'pos': [2, 6],
    'size': [5],
    'start': [0],
    'cluster': [0, 1],
}, {
    'pos': [[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]],
    'size': [5, 5],
    'cluster': [0, 5, 3, 0, 1],
}, {
    'pos': [[0, 0], [11, 9], [2, 8], [2, 2], [8, 3]],
    'size': [5, 5],
    'end': [19, 19],
    'cluster': [0, 6, 4, 0, 1],
}]


@pytest.mark.parametrize('test,dtype,device', product(tests, dtypes, devices))
def test_grid_cluster(test, dtype, device):
    pos = tensor(test['pos'], dtype, device)
    size = tensor(test['size'], dtype, device)
    start = tensor(test.get('start'), dtype, device)
    end = tensor(test.get('end'), dtype, device)
    
    pos = pos.unsqueeze(-1) if pos.dim() == 1 else pos
    start = pos.t().min(dim=1)[0] if start is None else start
    end = pos.t().max(dim=1)[0] if end is None else end

    op = pygrid.grid_cpp

    cluster = op.grid(pos, size, start, end)
    #cluster = grid_cluster(pos, size, start, end)
    assert cluster.tolist() == test['cluster']
