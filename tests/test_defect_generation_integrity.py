"""Saved analyses and written defect structures retain their physical meaning."""
import json
import numpy as np
import pytest
from cellstine.defect.records import DefectAnalysis, DefectSite
from cellstine.defect.workflow import Defect
from cellstine.io.models import StructureRecord
from cellstine.io.vasp import VaspIO


def _saved(tmp_path, cell=None, *, kind='atom', index=1, pair=(), multiplicity=1, equivalent=(1,), side='top'):
    cell = np.diag([4.,4.,20.]) if cell is None else np.asarray(cell, dtype=float)
    direct = np.array([[0.,0.,.3],[.5,.5,.7]])
    source = tmp_path/'host.vasp'
    record = StructureRecord(comment='defect integrity',lattice=cell,species=['Al'],counts=[2],
        positions_direct=direct,positions_cartesian=direct@cell,selective_dynamics=True,
        selective_flags=[('F','F','F'),('T','F','T')])
    VaspIO().write(record,str(source),positions_are_cartesian=False,wrap_positions=False)
    position = direct[1 if side == 'top' else 0] if kind == 'adatom' else direct[0]
    site = DefectSite(site_id='site_001',species='Al',layer_id=None if kind=='adatom' else 1,
        direct=tuple(position),cartesian=tuple(position@cell),site_kind=kind,representative_index=index,
        pair_indices=list(pair),multiplicity=multiplicity,equivalent_indices=list(equivalent))
    analysis = DefectAnalysis(structure_path=str(source),structure_kind='slab',backend='native',
        atom_count=2,species=['Al'],counts=[2],layers=[{'layer_id':i+1,'projection':6.+8*i,'atom_indices':[i+1],'atom_count':1} for i in range(2)],sites=[site])
    saved = tmp_path/'analysis.json'; saved.write_text(json.dumps(analysis.to_dict()),encoding='utf-8')
    workflow = Defect(runs_root=str(tmp_path/'runs'),output_root=str(tmp_path/'output'))
    return workflow,source,saved


@pytest.mark.parametrize('options',[{'layers':'1'},{'generate':'all'}])
def test_missing_multi_member_orbit_requires_reanalysis(tmp_path, options):
    workflow,source,saved = _saved(tmp_path,multiplicity=2,equivalent=(1,2))
    before = source.read_bytes(),saved.read_bytes()
    with pytest.raises(ValueError,match='orbit members.*[Rr]eanal'):
        workflow.generate(str(saved),'vacancy',backend='native',**options)
    assert (source.read_bytes(),saved.read_bytes()) == before
    assert not list((tmp_path/'output').glob('*.vasp'))


def test_known_single_representative_can_be_selected_without_guessing_an_orbit(tmp_path):
    workflow,source,saved = _saved(tmp_path)
    before = source.read_bytes(),saved.read_bytes()
    result = workflow.generate(str(saved),'vacancy',layers='1',backend='native')
    written = workflow.converter.read(result.payload['generated'][0]['output_path'],canonicalize=False)
    assert written.natoms == 1
    assert np.allclose(written.positions_direct,[[.5,.5,.7]])
    assert written.selective_flags == [('T','F','T')]
    assert (source.read_bytes(),saved.read_bytes()) == before


def test_legacy_multi_member_analysis_still_supports_unfiltered_inequivalent_generation(tmp_path):
    workflow,source,saved = _saved(tmp_path,multiplicity=2,equivalent=(1,2))
    before = source.read_bytes(),saved.read_bytes()
    result = workflow.generate(str(saved),'vacancy',backend='native')
    assert len(result.payload['generated']) == 1
    written = workflow.converter.read(result.payload['generated'][0]['output_path'],canonicalize=False)
    assert written.natoms == 1 and np.allclose(written.positions_direct,[[.5,.5,.7]])
    assert (source.read_bytes(),saved.read_bytes()) == before


def test_known_legacy_singleton_all_mode_still_accepts_its_original_site_id(tmp_path):
    workflow,source,saved = _saved(tmp_path)
    before = source.read_bytes(),saved.read_bytes()
    result = workflow.generate(str(saved),'vacancy',generate='all',site_ids=['site_001'],backend='native')
    assert result.payload['generated'][0]['site_id'] == 'site_001'
    written = workflow.converter.read(result.payload['generated'][0]['output_path'],canonicalize=False)
    assert written.natoms == 1 and np.allclose(written.positions_direct,[[.5,.5,.7]])
    assert (source.read_bytes(),saved.read_bytes()) == before


@pytest.mark.parametrize('missing_equivalence',[False,True])
@pytest.mark.parametrize('options',[
    {'layers':'1','site_ids':['site_001']},
    {'layers':'1','site_ids':['site_001_L01']},
    {'layers':'1','site_ids':['1']},
    {'generate':'all','site_ids':['site_001_M001']},
    {'generate':'all','site_ids':['1']},
])
def test_unselected_missing_orbit_does_not_block_a_complete_selected_site(tmp_path,options,missing_equivalence):
    workflow,source,saved = _saved(tmp_path)
    record = workflow.converter.read(str(source),canonicalize=False)
    record.counts = [3]
    record.positions_direct = np.vstack([record.positions_direct,[.25,.25,.5]])
    record.positions_cartesian = record.positions_direct@record.lattice
    record.selective_flags.append(('T','T','F'))
    VaspIO().write(record,str(source),positions_are_cartesian=False,wrap_positions=False)
    payload = json.loads(saved.read_text(encoding='utf-8'))
    payload.update(atom_count=3,counts=[3],layers=[
        {'layer_id':1,'projection':6.,'atom_indices':[1],'atom_count':1},
        {'layer_id':2,'projection':10.,'atom_indices':[3],'atom_count':1},
        {'layer_id':3,'projection':14.,'atom_indices':[2],'atom_count':1}])
    payload['sites'][0]['members'] = [{'indices':[1],'direct':record.positions_direct[0].tolist(),
        'cartesian':record.positions_cartesian[0].tolist(),'layer_ids':[1]}]
    other = dict(payload['sites'][0])
    other.update(site_id='other_002',layer_id=3,representative_index=2,equivalent_indices=[] if missing_equivalence else [2,3],
        multiplicity=2,members=[],direct=record.positions_direct[1].tolist(),cartesian=record.positions_cartesian[1].tolist())
    payload['sites'].append(other)
    saved.write_text(json.dumps(payload),encoding='utf-8')
    before = source.read_bytes(),saved.read_bytes()
    result = workflow.generate(str(saved),'vacancy',backend='native',**options)
    assert len(result.payload['generated']) == 1
    written = workflow.converter.read(result.payload['generated'][0]['output_path'],canonicalize=False)
    assert written.natoms == 2 and np.allclose(written.positions_direct,record.positions_direct[1:])
    assert written.selective_flags == record.selective_flags[1:]
    assert (source.read_bytes(),saved.read_bytes()) == before


@pytest.mark.parametrize('equivalent',[[],[2]])
def test_hidden_numeric_request_cannot_be_dropped_when_legacy_equivalence_is_incomplete(tmp_path,equivalent):
    workflow,source,saved = _saved(tmp_path)
    record = workflow.converter.read(str(source),canonicalize=False)
    record.counts = [3]
    record.positions_direct = np.vstack([record.positions_direct,[.25,.25,.5]])
    record.positions_cartesian = record.positions_direct@record.lattice
    record.selective_flags.append(('T','T','F'))
    VaspIO().write(record,str(source),positions_are_cartesian=False,wrap_positions=False)
    payload = json.loads(saved.read_text(encoding='utf-8'))
    payload.update(atom_count=3,counts=[3])
    payload['sites'][0]['members'] = [{'indices':[1],'direct':record.positions_direct[0].tolist(),
        'cartesian':record.positions_cartesian[0].tolist(),'layer_ids':[1]}]
    other = dict(payload['sites'][0])
    other.update(site_id='other_002',representative_index=2,equivalent_indices=equivalent,
        multiplicity=2,members=[],direct=record.positions_direct[1].tolist(),cartesian=record.positions_cartesian[1].tolist())
    payload['sites'].append(other)
    saved.write_text(json.dumps(payload),encoding='utf-8')
    before = source.read_bytes(),saved.read_bytes()
    with pytest.raises(ValueError,match='orbit members.*[Rr]eanal'):
        workflow.generate(str(saved),'vacancy',generate='all',site_ids=['site_001_M001','3'],backend='native')
    assert (source.read_bytes(),saved.read_bytes()) == before
    assert not list((tmp_path/'output').glob('*.vasp'))


@pytest.mark.parametrize('defect',['antisite','AnTiSiTe'])
def test_automatic_antisite_discovery_uses_the_normalized_defect_type(tmp_path,defect):
    workflow,source,saved = _saved(tmp_path)
    record = workflow.converter.read(str(source),canonicalize=False)
    record.species,record.counts = ['Al','He'],[1,1]
    VaspIO().write(record,str(source),positions_are_cartesian=False,wrap_positions=False)
    payload = json.loads(saved.read_text(encoding='utf-8'))
    payload['species'],payload['counts'] = ['Al','He'],[1,1]
    saved.write_text(json.dumps(payload),encoding='utf-8')
    before = source.read_bytes(),saved.read_bytes()
    result = workflow.generate(str(saved),defect,backend='native')
    written = workflow.converter.read(result.payload['generated'][0]['output_path'],canonicalize=False)
    assert written.natoms == 2
    assert all(symbol == 'He' for symbol,count in zip(written.species,written.counts) if count)
    assert np.allclose(written.positions_direct,record.positions_direct)
    assert written.selective_flags == record.selective_flags
    assert (source.read_bytes(),saved.read_bytes()) == before


@pytest.mark.parametrize('defect,index,pair',[('vacancy',0,()),('vacancy',3,()),
    ('substitution',0,()),('substitution',3,()),('divacancy',1,(1,1)),('divacancy',1,(1,3))])
def test_invalid_saved_atom_indices_never_write_wrong_cardinality(tmp_path, defect, index, pair):
    workflow,source,saved = _saved(tmp_path,kind='divacancy' if pair else 'atom',index=index,pair=pair)
    before = source.read_bytes(),saved.read_bytes()
    with pytest.raises(ValueError,match='index|indices|distinct'):
        workflow.generate(str(saved),defect,species='He',backend='native')
    assert (source.read_bytes(),saved.read_bytes()) == before
    assert not list((tmp_path/'output').glob('*.vasp'))


CELLS = [np.diag([4.,4.,20.]),np.diag([4.,4.,-20.]),
    np.array([[4.,0.,0.],[0.,0.,4.],[1.,-20.,0.]])]


@pytest.mark.parametrize('cell',CELLS)
@pytest.mark.parametrize('side',['top','bottom'])
@pytest.mark.parametrize('preserve',[False,True])
def test_written_adatom_height_follows_physical_normal(tmp_path,cell,side,preserve):
    workflow,source,saved = _saved(tmp_path,cell,kind='adatom',index=None,equivalent=(),side=side)
    before = source.read_bytes(),saved.read_bytes()
    result = workflow.generate(str(saved),'adatom',species='He',surface_side=side,height=2.,
        preserve_vacuum=preserve,backend='native')
    written = workflow.converter.read(result.payload['generated'][0]['output_path'],canonicalize=False)
    normal = np.cross(cell[0],cell[1]);normal/=np.linalg.norm(normal)
    if normal@cell[2]<0: normal=-normal
    host_index = 1 if side == 'top' else 0
    delta = written.positions_cartesian[-1]-written.positions_cartesian[host_index]
    assert np.allclose(delta,(2. if side=='top' else -2.)*normal,rtol=0.,atol=1e-8)
    assert written.selective_flags == [('F','F','F'),('T','F','T'),('T','T','T')]
    assert (source.read_bytes(),saved.read_bytes()) == before


@pytest.mark.parametrize('side',['top','bottom'])
def test_rotated_adatom_cannot_escape_normal_cell_bound_without_vacuum_fit(tmp_path,side):
    workflow,source,saved = _saved(tmp_path,CELLS[2],kind='adatom',index=None,equivalent=(),side=side)
    before = source.read_bytes(),saved.read_bytes()
    with pytest.raises(ValueError,match='outside the current cell'):
        workflow.generate(str(saved),'adatom',species='He',surface_side=side,height=25.,
            preserve_vacuum=False,backend='native')
    assert (source.read_bytes(),saved.read_bytes()) == before


@pytest.mark.parametrize('side',['top','bottom'])
@pytest.mark.parametrize('extra,accepted',[(.5e-6,True),(2.e-6,False)])
def test_normal_cell_bound_retains_its_existing_slack(tmp_path,side,extra,accepted):
    workflow,source,saved = _saved(tmp_path,CELLS[2],kind='adatom',index=None,equivalent=(),side=side)
    before = source.read_bytes(),saved.read_bytes()
    if accepted:
        result = workflow.generate(str(saved),'adatom',species='He',surface_side=side,height=12.+extra,
            preserve_vacuum=False,backend='native')
        assert len(result.payload['generated']) == 1
    else:
        with pytest.raises(ValueError,match='outside the current cell'):
            workflow.generate(str(saved),'adatom',species='He',surface_side=side,height=12.+extra,
                preserve_vacuum=False,backend='native')
    assert (source.read_bytes(),saved.read_bytes()) == before
