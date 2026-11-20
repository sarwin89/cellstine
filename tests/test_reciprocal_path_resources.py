"""Ray scale, cache ownership and exact band-path resource regressions."""

from dataclasses import replace
import math

import numpy as np
import pytest

from cellstine.core import brillouin, kpath, strata


@pytest.fixture(scope='module')
def cube():
    return brillouin.wigner_seitz_cell(np.eye(3))


@pytest.mark.parametrize('magnitude',[1e-300,1e-10,1.,1e10,1e300])
@pytest.mark.parametrize('direction',[[1.,0.,0.],[1.,-2.,.5]])
def test_boundary_geometry_depends_on_direction_not_magnitude(cube,magnitude,direction):
    ray=np.array(direction)
    expected=ray/(2.*np.max(np.abs(ray)))
    assert np.allclose(cube.boundary_point(ray*magnitude),expected,rtol=1e-14,atol=0.)
    factor=cube.boundary_scale(ray*magnitude)
    assert math.isfinite(factor) and factor>0.
    assert np.allclose((ray*magnitude)*factor,expected,rtol=1e-14,atol=0.)
    expected_distance=float(np.linalg.norm(expected*2.*np.pi))
    assert brillouin.zone_boundary_distance(np.eye(3),ray*magnitude)==pytest.approx(expected_distance,rel=1e-14)


@pytest.mark.parametrize('ray',[[0.,0.,0.],[float('nan'),0.,0.],[float('inf'),0.,0.]])
def test_boundary_rejects_zero_and_nonfinite_directions(cube,ray):
    for function in (cube.boundary_scale,cube.boundary_point):
        with pytest.raises(ValueError,match='nonzero|finite'):
            function(ray)
    with pytest.raises(ValueError,match='nonzero|finite'):
        brillouin.zone_boundary_distance(np.eye(3),ray)


@pytest.mark.parametrize('tolerance',[float('nan'),float('inf'),-float('inf'),0.,-1.])
def test_wigner_tolerance_is_finite_positive(tolerance):
    with pytest.raises(ValueError,match='finite|positive'):
        brillouin.wigner_seitz_cell(np.eye(3),tolerance=tolerance)


def test_cached_direction_list_and_arrays_are_owned_by_each_call():
    saved=dict(strata._STRATA_CACHE)
    strata._STRATA_CACHE.clear()
    try:
        operations=np.array([np.eye(3,dtype=np.int64),np.diag([-1,-1,1])])
        first=strata.grid_strata(operations)
        index=next(i for i,value in enumerate(first[2]) if value is not None)
        original=first[2][index].copy()
        assert first[2][index].flags.writeable
        first[2][index][0]+=7
        second=strata.grid_strata(operations)
        assert np.array_equal(second[2][index],original)
        first[2][index]=None
        third=strata.grid_strata(operations)
        assert np.array_equal(third[2][index],original)
        assert isinstance(third[2],list) and third[2][index].flags.writeable
        assert not third[0].flags.writeable and not third[1].flags.writeable
        assert not np.shares_memory(second[2][index],third[2][index])
    finally:
        strata._STRATA_CACHE.clear()
        strata._STRATA_CACHE.update(saved)


@pytest.fixture(scope='module')
def path():
    return kpath.band_path(np.eye(3))


def _scalar_samples(path,divisions):
    points,distances,labels=[],[],[]
    travelled=0.
    for run,run_points in zip(path.walk,path.walk_points):
        coordinates=np.asarray(run_points,dtype=float)
        for position in range(len(run)-1):
            start,end=coordinates[position:position+2]
            span=float(np.linalg.norm((end-start)@path.reciprocal))
            first=0 if position==0 else 1
            for step in range(first,divisions):
                fraction=step/(divisions-1)
                points.append(start+fraction*(end-start))
                distances.append(travelled+fraction*span)
                labels.append(run[position] if step==0 else run[position+1] if step==divisions-1 else '')
            travelled+=span
    return np.asarray(points,dtype=float),np.asarray(distances,dtype=float),tuple(labels)


@pytest.mark.parametrize('divisions',[2,3,9,127])
@pytest.mark.parametrize('walk',[(('A','B','C'),('D','E')),(('A','A'),),(('A',),),()])
def test_sample_preserves_scalar_points_labels_breaks_and_empty_shape(path,divisions,walk):
    rng=np.random.default_rng(901)
    run_points=tuple(rng.random((len(run),3)) for run in walk)
    modified=replace(path,walk=walk,walk_points=run_points,reciprocal=np.array([[2.,.3,.1],[.1,3.,-.2],[0.,.4,4.]]))
    expected=_scalar_samples(modified,divisions)
    actual=modified.sample(divisions)
    assert np.array_equal(actual[0],expected[0]) and actual[0].shape==expected[0].shape
    assert np.array_equal(actual[1],expected[1]) and actual[2]==expected[2]


def test_unaddressable_sample_count_rejected_before_segment_work(path,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('unaddressable sample request reached per-segment work')
    monkeypatch.setattr(kpath.np.linalg,'norm',forbidden)
    with pytest.raises(ValueError,match='too large|address'):
        path.sample(2**62)


def test_fixing_masks_are_blocked_and_match_periodic_reference(monkeypatch):
    rng=np.random.default_rng(321)
    points=rng.random((10000,3))
    operations=kpath.kspace_operations(np.array([np.eye(3,dtype=int),np.diag([-1,-1,1])]))
    points[:4]=[[0.,0.,0.],[.5,0.,0.],[1e-8,0.,0.],[np.nextafter(1e-8,np.inf),0.,0.]]
    expected=[]
    for point in points:
        images=np.einsum('j,kjl->kl',point,operations.astype(float))
        residual=images-point
        expected.append(np.all(np.abs(residual-np.rint(residual))<=1e-8,axis=1))
    einsum=np.einsum
    def bounded(expression,*args,**kwargs):
        if expression=='pj,kjl->pkl':
            assert len(args[0])<=4096,'whole point table temporary'
        return einsum(expression,*args,**kwargs)
    monkeypatch.setattr(strata.np,'einsum',bounded)
    assert np.array_equal(strata._fixing_masks(points,operations),np.array(expected))
    assert strata._fixing_masks(np.empty((0,3)),operations).shape==(0,len(operations))


def _sorted_dijkstra(neighbours,source):
    count=len(neighbours)
    distances=[math.inf]*count
    previous=[-1]*count
    distances[source]=0.
    queue=[(0.,source)]
    seen=set()
    while queue:
        queue.sort(reverse=True)
        distance,node=queue.pop()
        if node in seen:
            continue
        seen.add(node)
        for target,weight in neighbours[node]:
            candidate=distance+weight
            if candidate<distances[target]-1e-12:
                distances[target]=candidate
                previous[target]=node
                queue.append((candidate,target))
    return distances,previous


def test_heap_graph_distances_and_ties_preserve_sorted_reference():
    rng=np.random.default_rng(773)
    for count in (1,9,51):
        graph=[[(int(j),float(rng.integers(0,4))) for j in rng.choice(count,size=min(count,5),replace=False)] for _ in range(count)]
        for source in (0,count-1):
            assert kpath._dijkstra(graph,source,count)==_sorted_dijkstra(graph,source)
    graph=[[(1,1.),(2,1.)],[(3,1.)],[(3,1.-5e-13)],[]]
    assert kpath._dijkstra(graph,0,4)==_sorted_dijkstra(graph,0)


def test_integer_grid_congruence_does_not_overflow_before_modulo(monkeypatch):
    saved=dict(strata._STRATA_CACHE)
    strata._STRATA_CACHE.clear()
    try:
        k=2_000_000_000
        matrix=np.array([[k-1,-k*k+k-1,0],[1,-k,0],[0,0,1]],dtype=np.int64)
        squared=-matrix-np.eye(3,dtype=np.int64)
        squared[2,2]=1  # The invariant z axis is not part of the order-three 2D block.
        operations=np.array([np.eye(3,dtype=np.int64),matrix,squared])
        assert np.array_equal(matrix.astype(object)@squared.astype(object),np.eye(3,dtype=object))
        def classify(matrices):
            # Isolate exact integer congruence; huge-coefficient float SVD is an existing separate limitation.
            return (1,None) if len(matrices)==3 else (3,None)
        monkeypatch.setattr(strata,'_fixed_space_dimension',classify)
        grid,dimensions,_=strata.grid_strata(operations)
        expected=np.all((grid@(matrix%48)-grid)%48==0,axis=1)
        assert np.array_equal(dimensions,np.where(expected,1,3))
    finally:
        strata._STRATA_CACHE.clear()
        strata._STRATA_CACHE.update(saved)
