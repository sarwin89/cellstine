"""Exact ordered sublattices and bounded temporary enumeration."""
from itertools import product
import tracemalloc
import numpy as np
import pytest
from cellstine.defect import supercell as sc
from cellstine.defect.records import DefectAnalysis,DefectSite
from cellstine.defect.reporting import DefectReportingMixin
from cellstine.core._integer_grid import _integer_box_blocks


@pytest.mark.parametrize('cells',[1,2,6,12,25,32])
@pytest.mark.parametrize('plane',[False,True])
def test_parameter_arrays_match_independent_nested_integer_enumeration(cells,plane):
    expected=[]
    for first in range(1,cells+1):
        if cells%first:continue
        if plane:
            expected.extend((first,upper,cells//first) for upper in range(cells//first))
        else:
            rest=cells//first
            for second in range(1,rest+1):
                if rest%second:continue
                third=rest//second
                expected.extend((first,ab,ac,second,bc,third)
                    for ab,ac,bc in product(range(second),range(third),range(third)))
    actual=(sc._hermite_parameters_2d if plane else sc._hermite_parameters_3d)(cells)
    assert actual.dtype==np.int64 and actual.flags.c_contiguous
    assert np.array_equal(actual,np.asarray(expected,dtype=np.int64))


@pytest.mark.parametrize('plane,cells,budget',[(False,256,12*1024*1024),(True,1024,80*1024)])
def test_parameter_output_does_not_require_a_second_full_copy(plane,cells,budget):
    tracemalloc.start()
    try:
        array=(sc._hermite_parameters_2d if plane else sc._hermite_parameters_3d)(cells)
        peak=tracemalloc.get_traced_memory()[1]
    finally:tracemalloc.stop()
    assert len(array)==sc.hermite_normal_form_count(cells,plane=plane)
    assert peak<budget


@pytest.mark.parametrize('plane',[False,True])
def test_host_vectors_use_bounded_real_integer_blocks(monkeypatch,plane):
    seen=[]
    def checked_blocks(bounds):
        for block in _integer_box_blocks(bounds):
            assert len(block)<=65536
            seen.append(len(block))
            yield block
    def no_mesh(*args,**kwargs):
        pytest.fail('host vector search materialized a meshgrid')
    monkeypatch.setattr(sc.np,'meshgrid',no_mesh)
    monkeypatch.setattr(sc,'_integer_box_blocks',checked_blocks,raising=False)
    cell=np.array([[1.,0.,0.],[10.,.5,0.],[0.,10.,.5]])
    points,norms=sc.host_lattice_vectors(cell,1.25,plane=plane)
    assert len(points)==len(norms)>0
    assert np.all(norms<=1.25+1e-9)
    assert seen


@pytest.mark.parametrize('plane',[False,True])
@pytest.mark.parametrize('radius',[0.,-.5,1.,np.sqrt(2.)-0.5e-9,np.sqrt(2.)-2e-9])
def test_host_vectors_match_scalar_integer_reference(plane,radius):
    cell=np.eye(3);dimensions=2 if plane else 3
    expected=[]
    for point in product(range(-3,4),repeat=dimensions):
        norm=float(np.linalg.norm(np.asarray(point,dtype=float)))
        if any(point) and norm<=radius+1e-9:expected.append((point,norm))
    points,norms=sc.host_lattice_vectors(cell,radius,plane=plane)
    assert {tuple(point) for point in points}=={p for p,n in expected}
    assert np.all(np.diff(norms)>=0.)
    assert np.array_equal(norms,np.array([np.linalg.norm(point) for point in points]))


def test_skew_host_vector_temporaries_fit_small_budget():
    cell=np.array([[1.,0.,0.],[10.,.5,0.],[0.,10.,.5]])
    tracemalloc.start()
    try:
        points,norms=sc.host_lattice_vectors(cell,1.25,plane=False)
        peak=tracemalloc.get_traced_memory()[1]
    finally:tracemalloc.stop()
    assert len(points)>0 and len(points)==len(norms)
    assert peak<16*1024*1024


@pytest.mark.parametrize('radius',[1e20,-1e20,float('inf'),float('nan')])
def test_unrepresentable_search_is_rejected_instead_of_reported_empty(radius):
    with pytest.raises(ValueError,match='finite|int64|coordinate'):
        sc.host_lattice_vectors(np.eye(3),radius,plane=False)


def test_report_limit_only_changes_presentation():
    sites=[DefectSite(site_id=f'atom_{i:03d}',species='Al',layer_id=1,direct=(0.,0.,0.),
        cartesian=(0.,0.,0.),representative_index=i) for i in range(1,5)]
    analysis=DefectAnalysis(structure_path='host.vasp',structure_kind='bulk',backend='native',
        atom_count=4,species=['Al'],counts=[4],layers=[],sites=sites)
    text=DefectReportingMixin.format_analysis(analysis,limit=2)
    assert 'atom_001' in text and 'atom_002' in text and 'atom_003' not in text
    assert '2 more site(s)' in text and analysis.sites is sites and len(sites)==4


@pytest.mark.parametrize('limit',[0,-2])
def test_zero_presentation_limit_does_not_claim_discovery_was_empty(limit):
    sites=[DefectSite(site_id='atom_001',species='Al',layer_id=1,direct=(0.,0.,0.),cartesian=(0.,0.,0.))]
    analysis=DefectAnalysis(structure_path='host.vasp',structure_kind='bulk',backend='native',
        atom_count=1,species=['Al'],counts=[1],layers=[],sites=sites)
    text=DefectReportingMixin.format_analysis(analysis,limit=limit)
    assert 'No defect sites were detected' not in text
    assert '1 more site(s) not shown' in text and analysis.sites is sites
