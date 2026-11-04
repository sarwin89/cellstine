"""Symmetry tolerance contracts, complete ordered searches and bounded work."""

import itertools
import math

import numpy as np
import pytest

from cellstine.core import pointgroup3d, symmetry2d, symmetry3d


def _centering_reference(translations,centering,tolerance=1e-6):
    vectors=np.mod(np.asarray(translations,dtype=float).reshape(-1,3),1.)
    sites=np.mod(np.asarray(centering,dtype=float).reshape(-1,3),1.)
    differences=vectors[:,None,:]-sites[None,:,:]
    differences-=np.rint(differences)
    return bool(np.all(np.any(np.all(np.abs(differences)<=tolerance,axis=2),axis=1)))


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('origin',[.5,0.])
def test_centering_accepts_cube_match_even_when_nearest_sphere_match_fails(axis,origin):
    query=np.array([.5,.5,.5])
    query[axis]=origin
    outside=query.copy()
    outside[axis]+=1.1e-6
    inside=query.copy()
    inside[axis]+=.9e-6
    inside[(axis+1)%3]+=.9e-6
    centers=np.vstack((np.zeros(3),np.mod(outside,1.),np.mod(inside,1.)))
    assert _centering_reference([query],centers)
    assert symmetry3d._translations_are_centering(np.array([query]),centers)


def test_centering_near_boundary_matches_direct_componentwise_rule():
    rng=np.random.default_rng(5317)
    for _ in range(40):
        sites=rng.random((9,3))
        query=sites+rng.uniform(-1.5e-6,1.5e-6,(9,3))
        assert symmetry3d._translations_are_centering(query,sites)==_centering_reference(query,sites)
    assert symmetry3d._translations_are_centering(np.empty((0,3)),np.empty((0,3)))
    assert not symmetry3d._translations_are_centering(np.zeros((1,3)),np.empty((0,3)))


@pytest.mark.parametrize('query',[[1e-6]*3,[1e-6,1e-6,0.],[1e-6,0.,0.]])
def test_centering_cube_boundary_is_not_lost_to_spherical_rounding(query):
    sites=np.zeros((1,3))
    assert _centering_reference([query],sites)
    assert symmetry3d._translations_are_centering(np.array([query]),sites)


@pytest.mark.parametrize('function',[symmetry3d.translation_group,symmetry3d.planar_translation_basis,
                                   symmetry3d.translation_lattice_basis,symmetry3d.planar_primitive_layer])
def test_empty_structure_rejects_nonempty_species(function):
    with pytest.raises(ValueError,match='one species label per atom'):
        function(np.eye(3),np.empty((0,3)),['C'])


def test_valid_empty_primitive_layer_keeps_empty_shape():
    lattice,positions,species,index=symmetry3d.planar_primitive_layer(np.eye(3),np.empty((0,3)),[])
    assert np.array_equal(lattice,np.eye(3))
    assert positions.shape==(0,3) and species==[] and index==1


@pytest.mark.parametrize('function',[symmetry3d.site_permutations,symmetry3d.equivalent_atom_map])
def test_provided_operations_reuse_one_index_with_same_permutations(monkeypatch,function):
    positions=np.array([[0.,0.,0.],[.5,.5,.5]])
    rotations=np.repeat(np.eye(3,dtype=np.int64)[None,:,:],5,axis=0)
    shifts=np.array([[0.,0.,0.],[.5,.5,.5],[.25,0.,0.],[0.,0.,0.],[.5,.5,.5]])
    original=symmetry3d.PeriodicSiteIndex
    constructions=[]
    class Counted(original):
        def __init__(self,*args,**kwargs):
            constructions.append(1)
            super().__init__(*args,**kwargs)
    monkeypatch.setattr(symmetry3d,'PeriodicSiteIndex',Counted)
    actual=function(np.eye(3)*3.,positions,['C','C'],rotations,shifts)
    assert len(constructions)<=1
    if function is symmetry3d.site_permutations:
        assert np.array_equal(actual,np.array([[0,1],[1,0],[0,1],[0,1],[1,0]]))
    else:
        assert np.array_equal(actual,np.zeros(2,dtype=np.int64))
    assert np.array_equal(positions,np.array([[0.,0.,0.],[.5,.5,.5]]))


@pytest.mark.parametrize('function',[symmetry3d.site_permutations,symmetry3d.equivalent_atom_map])
def test_no_provided_operations_do_not_build_an_index(monkeypatch,function):
    def forbidden(*args,**kwargs):
        pytest.fail('empty operation set built a spatial index')
    monkeypatch.setattr(symmetry3d,'PeriodicSiteIndex',forbidden)
    actual=function(np.eye(3),np.zeros((1,3)),['C'],np.empty((0,3,3),dtype=int),np.empty((0,3)))
    assert np.array_equal(actual,np.array([[0]]) if function is symmetry3d.site_permutations else np.array([0]))


@pytest.mark.parametrize('bounds',[(1,2),(2,1,3),(0,0),(0,)])
@pytest.mark.parametrize('block',[1,3,7,64])
def test_bounded_boxes_preserve_meshgrid_order(bounds,block):
    from cellstine.core._integer_grid import _integer_box_blocks
    pieces=list(_integer_box_blocks(bounds,block))
    assert all(len(piece)<=block and piece.dtype==np.dtype(np.int64) for piece in pieces)
    expected=np.array(list(itertools.product(*(range(-n,n+1) for n in bounds))),dtype=np.int64)
    assert np.array_equal(np.concatenate(pieces),expected)


def test_box_streams_without_full_index_or_axis_arrays():
    from cellstine.core._integer_grid import _integer_box_blocks
    pieces=list(itertools.islice(_integer_box_blocks((2**40,2**40),16),2))
    assert len(pieces)==2 and all(piece.shape==(16,2) for piece in pieces)
    assert np.array_equal(pieces[0][:,0],np.full(16,-2**40,dtype=np.int64))
    assert np.array_equal(pieces[1][:,1],np.arange(16,dtype=np.int64)-2**40+16)


@pytest.mark.parametrize('bounds,block,error',[
    ((),64,ValueError),((-1,),64,ValueError),((2**63,),64,ValueError),
    ((1,),0,ValueError),((1.5,),64,TypeError),
])
def test_box_rejects_invalid_coordinate_bounds(bounds,block,error):
    from cellstine.core._integer_grid import _integer_box_blocks
    with pytest.raises(error):
        next(_integer_box_blocks(bounds,block))


@pytest.mark.parametrize('radius',[-1.,0.,1.,3.,np.nextafter(3.,0.),np.nextafter(3.,np.inf)])
def test_three_dimensional_vectors_have_exact_order_and_norms(radius):
    vectors,norms=pointgroup3d._integer_vectors_within(np.diag([1.,2.,3.]),radius)
    expected=[row for row in itertools.product(range(-2,3),repeat=3)
              if row[0]**2+2*row[1]**2+3*row[2]**2<=radius]
    assert np.array_equal(vectors,np.asarray(expected,dtype=np.int64).reshape(-1,3))
    assert np.array_equal(norms,np.asarray([x*x+2*y*y+3*z*z for x,y,z in expected],dtype=float))


def test_large_skew_plane_search_is_complete_without_a_full_meshgrid(monkeypatch):
    meshgrid=np.meshgrid
    def bounded(*axes,**kwargs):
        assert math.prod(len(axis) for axis in axes)<=65536,'full search-box allocation'
        return meshgrid(*axes,**kwargs)
    monkeypatch.setattr(symmetry2d.np,'meshgrid',bounded)
    expected=[]
    for n in range(-100,101):
        bound=math.isqrt(10001-n*n)
        expected.extend((k-100*n,n) for k in range(-bound,bound+1) if k or n)
    actual=symmetry2d._short_vectors(np.array([[1.,100.],[100.,10001.]]),10001.)
    assert np.array_equal(actual,np.asarray(sorted(expected),dtype=np.int64))


def test_large_3d_search_preserves_integer_ellipsoid_and_norms(monkeypatch):
    meshgrid=np.meshgrid
    def bounded(*axes,**kwargs):
        assert math.prod(len(axis) for axis in axes)<=65536,'full search-box allocation'
        return meshgrid(*axes,**kwargs)
    monkeypatch.setattr(pointgroup3d.np,'meshgrid',bounded)
    expected=[(i,j,k) for i,j,k in itertools.product(range(-100,101),range(-70,71),range(-1,2))
              if i*i+2*j*j+10000*k*k<=10000]
    actual,norms=pointgroup3d._integer_vectors_within(np.diag([1.,2.,10000.]),10000.)
    assert np.array_equal(actual,np.asarray(expected,dtype=np.int64))
    assert np.array_equal(norms,np.asarray([i*i+2*j*j+10000*k*k for i,j,k in expected],dtype=float))


def test_decorated_match_uses_bounded_pairwise_blocks(monkeypatch):
    planar=np.array(list(itertools.product(np.arange(32)/32.,repeat=2)))
    wrap=symmetry2d._wrap
    shapes=[]
    def bounded(values):
        shapes.append(values.shape)
        assert values.shape[0]*values.shape[1]<=65536,'full pairwise-site allocation'
        return wrap(values)
    monkeypatch.setattr(symmetry2d,'_wrap',bounded)
    assert symmetry2d._accepts_translation(planar,planar,np.zeros(len(planar),dtype=int),
                                         np.zeros(len(planar)),1e-4,1e-3)
    assert len(shapes)>1


def test_decorated_match_preserves_row_and_column_rule():
    rng=np.random.default_rng(8319)
    for _ in range(16):
        planar=rng.random((13,2))
        labels=rng.integers(0,2,13)
        heights=rng.integers(0,2,13).astype(float)
        for element in (np.eye(2),-np.eye(2),np.array([[0.,1.],[1.,0.]])):
            mapped=planar@element.T
            expected=False
            for partner in np.flatnonzero((labels==labels[0])&(np.abs(heights-heights[0])<=1e-3)):
                shifted=mapped+(planar[partner]-mapped[0])
                difference=shifted[:,None,:]-planar[None,:,:]
                difference-=np.rint(difference)
                close=np.all(np.abs(difference)<=1e-4,axis=2)
                close&=labels[:,None]==labels[None,:]
                close&=np.abs(heights[:,None]-heights[None,:])<=1e-3
                expected|=bool(np.all(close.any(axis=1)) and np.all(close.any(axis=0)))
            assert symmetry2d._accepts_translation(mapped,planar,labels,heights,1e-4,1e-3)==expected


@pytest.mark.parametrize('mode',['identity','last-row-missing','last-column-missing'])
def test_decorated_match_aggregates_all_blocks_and_columns(mode):
    rng=np.random.default_rng(912)
    planar=rng.random((270,2))
    mapped=planar.copy()
    labels=np.ones(270,dtype=int)
    labels[0]=0 # only one reference partner, so no alternative translation masks a failure
    if mode=='last-row-missing':
        mapped[-1]+=.1
    elif mode=='last-column-missing':
        mapped[-1]=mapped[-2] # every row matches, but the final target column has no image
    assert symmetry2d._accepts_translation(mapped,planar,labels,np.zeros(270),1e-4,1e-3)==(mode=='identity')
