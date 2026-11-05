"""Exact builder geometry, bounded cosets and safe batch failure semantics."""

import copy
import importlib
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from cellstine.io import native as io_mod
from cellstine.core.layers import identify_top_layer, resolve_shift_vectors
from cellstine.moire.builder import generator
from cellstine.moire.builder import nlayer as stack_builder
from cellstine.moire.search.find import run_find
from cellstine.moire.search.nlayer import run_findn
from cellstine.moire.search.results import read_results, write_results

bilayer=importlib.import_module('cellstine.moire.builder.make')


@pytest.fixture
def layer(tmp_path):
    path=tmp_path/'layer.vasp'
    path.write_text('carbon\n1\n2 0 0\n0 2 0\n0 0 12\nC\n1\nSelective dynamics\nDirect\n0 0 0.5 F T F\n',encoding='utf-8')
    return path


@pytest.fixture
def bilayer_document(tmp_path,layer):
    run=run_find(top_poscar=str(layer),bottom_poscar=str(layer),max_length=2.1,
                 top_strain=0.,bottom_strain=0.,reduce_layers=False,
                 output_root=str(tmp_path/'search'))
    document=read_results(run.result_path)
    document['candidates']=[dict(document['candidates'][0],index=i) for i in (1,2)]
    document['metadata'].update(candidates_found=2,candidates_recorded=2)
    return document


@pytest.mark.parametrize('matrix,expected',[
    ([[2**32,0],[0,2**32]],(2**32,0,2**32)),
    ([[-2**32,0],[0,2**32]],(2**32,0,2**32)),
    ([[2**32,2**32-1],[2**32+1,2**32]],(1,0,1)),
])
def test_hnf_determinant_uses_exact_integers_without_overflow(matrix,expected):
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        assert generator._column_hermite_normal_form(np.asarray(matrix,dtype=np.int64))==expected


def test_impossible_coset_buffer_rejected_before_allocation(monkeypatch):
    matrix=np.diag([2**31,2**31]).astype(np.int64)
    def forbidden(*args,**kwargs):
        pytest.fail('impossible coset enumeration attempted allocation')
    for name in ('arange','empty','meshgrid'):
        monkeypatch.setattr(generator.np,name,forbidden)
    with pytest.raises(ValueError,match='too large|address'):
        generator._coset_representatives(matrix)


@pytest.mark.parametrize('matrix',[
    [[3,1],[1,4]],[[-6,2],[3,4]],[[0,5],[7,0]],[[37,0],[0,53]],
])
def test_coset_order_matches_exact_ij_box(matrix):
    matrix=np.asarray(matrix,dtype=np.int64)
    h11,_,h22=generator._column_hermite_normal_form(matrix.T)
    expected=np.asarray([(x,y) for x in range(h11) for y in range(h22)],dtype=np.int64)
    actual=generator._coset_representatives(matrix)
    assert actual.dtype==np.dtype(np.int64)
    assert actual.flags.c_contiguous and np.array_equal(actual,expected)


@pytest.mark.parametrize('kind',['json','npz','npz-compressed'])
@pytest.mark.parametrize('workers',[1,2])
@pytest.mark.parametrize('parallel_backend',['process','thread'])
def test_batch_reads_once_preserves_duplicates_and_exact_output(tmp_path,monkeypatch,bilayer_document,kind,workers,parallel_backend):
    suffix='json' if kind=='json' else 'npz'
    path=write_results(tmp_path/f'results.{suffix}',bilayer_document,compressed=kind=='npz-compressed')
    original=copy.deepcopy(bilayer_document)
    expected=[bilayer.generate_from_results(str(path),index=i,interlayer_distance=3.35,
                output_dir=str(tmp_path/'single'),top_c_repeat=2,vacuum=15.) for i in (2,1,2)]
    calls=[]
    parse=generator.parse_results
    def counted(filename):
        calls.append(filename)
        return parse(filename)
    monkeypatch.setattr(generator,'parse_results',counted)
    actual=bilayer.generate_many_from_results(str(path),indexes=[2,1,2],interlayer_distance=3.35,
                output_dir=str(tmp_path/'batch'),top_c_repeat=2,vacuum=15.,workers=workers,parallel_backend=parallel_backend)
    assert calls==[str(path)]
    assert [r.selected_index for r in actual]==[2,1,2]
    assert len({r.output_path for r in actual})==3
    for before,after in zip(expected,actual):
        assert before.output_path.read_bytes()==after.output_path.read_bytes()
        assert before.total_atoms==after.total_atoms and before.notes==after.notes
        assert before.contact_distance==after.contact_distance
        assert before.structure_contact_distance==after.structure_contact_distance
        result=io_mod.read_poscar(str(after.output_path))
        assert result.selective_flags==[('F','T','F')]*result.natoms
    assert bilayer_document==original


def test_empty_batch_does_not_read_or_create_outputs(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('empty selection read a results document')
    monkeypatch.setattr(generator,'parse_results',forbidden)
    monkeypatch.setattr(stack_builder,'read_nlayer_results',forbidden)
    assert bilayer.generate_many_from_results('missing.json',indexes=[],interlayer_distance=3.,output_dir=str(tmp_path/'out'),workers=2)==[]
    assert stack_builder.generate_many_from_results('missing.json',indexes=[],interlayers=[3.],output_dir=str(tmp_path/'out'))==[]
    assert not (tmp_path/'out').exists()


def test_late_invalid_index_keeps_prior_output(tmp_path,bilayer_document):
    path=write_results(tmp_path/'results.json',bilayer_document)
    with pytest.raises(ValueError,match='index 999 not found'):
        bilayer.generate_many_from_results(str(path),indexes=[1,999],interlayer_distance=3.35,output_dir=str(tmp_path/'out'))
    files=list((tmp_path/'out').glob('*.vasp'))
    assert len(files)==1 and io_mod.read_poscar(str(files[0])).natoms==2


class _InlineExecutor:
    def __init__(self,**kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self,*args):
        pass
    def submit(self,worker,task):
        from concurrent.futures import Future
        future=Future()
        try:
            future.set_result(worker(task))
        except BaseException as error:
            future.set_exception(error)
        return future


def test_worker_write_error_propagates_without_retrying_completed_outputs(tmp_path,monkeypatch,bilayer_document):
    path=write_results(tmp_path/'results.json',bilayer_document)
    monkeypatch.setattr(bilayer,'ProcessPoolExecutor',_InlineExecutor)
    write=generator.write_supercell_poscar
    attempts=[]
    def fail_second(filename,*args,**kwargs):
        attempts.append(filename)
        if len(attempts)==2:
            raise OSError('candidate write failed')
        return write(filename,*args,**kwargs)
    monkeypatch.setattr(generator,'write_supercell_poscar',fail_second)
    with pytest.raises(OSError,match='candidate write failed'):
        bilayer.generate_many_from_results(str(path),indexes=[1,2,1],interlayer_distance=3.35,
                    output_dir=str(tmp_path/'out'),workers=2)
    # Later tasks may start before the ordered failure is observed.
    # All three requested writes run once; no completed output is replayed.
    assert len(attempts)==3
    files=list((tmp_path/'out').glob('*.vasp'))
    assert len(files)==3 # failed default destination remains reserved
    assert sum(bool(p.stat().st_size) for p in files)==2


def test_pool_construction_failure_safely_falls_back(tmp_path,monkeypatch,bilayer_document):
    path=write_results(tmp_path/'results.json',bilayer_document)
    def unavailable(**kwargs):
        raise OSError('pool unavailable before dispatch')
    monkeypatch.setattr(bilayer,'ProcessPoolExecutor',unavailable)
    runs=bilayer.generate_many_from_results(str(path),indexes=[2,1],interlayer_distance=3.35,
                output_dir=str(tmp_path/'out'),workers=2)
    assert [r.selected_index for r in runs]==[2,1]
    assert all(io_mod.read_poscar(str(r.output_path)).natoms==2 for r in runs)


def test_pool_dispatch_error_is_not_assumed_safe_to_retry(tmp_path,monkeypatch,bilayer_document):
    path=write_results(tmp_path/'results.json',bilayer_document)
    class DispatchError(_InlineExecutor):
        def submit(self,*args):
            raise OSError('dispatch failure after pool creation')
    monkeypatch.setattr(bilayer,'ProcessPoolExecutor',DispatchError)
    with pytest.raises(OSError,match='dispatch failure'):
        bilayer.generate_many_from_results(str(path),indexes=[1,2],interlayer_distance=3.35,
                    output_dir=str(tmp_path/'out'),workers=2)
    assert not (tmp_path/'out').exists()


def test_nlayer_batch_reads_once_and_preserves_output(tmp_path,monkeypatch,layer):
    run=run_findn(base_poscar=str(layer),upper_poscars=[str(layer),str(layer)],max_length=2.1,
                  layer_strains=0.,reduce_layers=False,output_root=str(tmp_path/'search'))
    document=json.loads(run.result_path.read_text(encoding='utf-8'))
    document['candidates']=[dict(document['candidates'][0],index=i) for i in (1,2)]
    path=tmp_path/'stack.json'
    path.write_text(json.dumps(document),encoding='utf-8')
    expected=[stack_builder.generate_from_results(str(path),index=i,interlayers=[3.35,4.],
                output_dir=str(tmp_path/'single'),vacuum=15.) for i in (2,1,2)]
    calls=[]
    read=stack_builder.read_nlayer_results
    def counted(filename):
        calls.append(filename)
        return read(filename)
    monkeypatch.setattr(stack_builder,'read_nlayer_results',counted)
    actual=stack_builder.generate_many_from_results(str(path),indexes=[2,1,2],interlayers=[3.35,4.],
                output_dir=str(tmp_path/'batch'),vacuum=15.)
    assert calls==[str(path)]
    assert len({r.output_path for r in actual})==3
    for before,after in zip(expected,actual):
        assert before.output_path.read_bytes()==after.output_path.read_bytes()
        assert before.layer_counts==after.layer_counts==(1,1,1)


@pytest.mark.parametrize('heights,cutoff,message',[
    ([],None,'does not contain any atoms'),
    ([1.],None,'at least two atoms'),
    ([0.,.2],None,'largest internal z gap'),
    ([0.,2.],2.,'no atoms were found above'),
    ([0.,2.],-1.,'all atoms are above'),
])
def test_layer_selection_rejects_unseparated_or_empty_sides(heights,cutoff,message):
    cartesian=np.zeros((len(heights),3))
    cartesian[:,2]=heights
    lattice=np.diag([3.,3.,12.])
    structure=io_mod.PoscarData(comment='layers',lattice=lattice,species=['C'],counts=[len(heights)],
        positions_direct=cartesian@np.linalg.inv(lattice),positions_cartesian=cartesian,
        coordinate_mode='Direct',selective_dynamics=False,selective_flags=None)
    with pytest.raises(ValueError,match=message):
        identify_top_layer(structure,z_cutoff=cutoff)


def test_layer_ties_choose_first_gap_and_cutoff_is_strict():
    lattice=np.diag([3.,3.,12.])
    cartesian=np.array([[0.,0.,0.],[0.,0.,2.],[0.,0.,4.]])
    structure=io_mod.PoscarData(comment='layers',lattice=lattice,species=['C'],counts=[3],
        positions_direct=cartesian@np.linalg.inv(lattice),positions_cartesian=cartesian,
        coordinate_mode='Direct',selective_dynamics=False,selective_flags=None)
    selected=identify_top_layer(structure,min_gap=2.)
    assert selected.top_indices==(1,2) and selected.bottom_indices==(0,)
    assert selected.top_atom_count==2 and selected.bottom_atom_count==1
    explicit=identify_top_layer(structure,z_cutoff=2.)
    assert explicit.top_indices==(2,) and explicit.bottom_indices==(0,1)


@pytest.mark.parametrize('direct',[[.2,.3],[.2,.3,.4]])
def test_shift_vector_modes_agree_on_skew_lattice(direct):
    lattice=np.array([[3.,0.,0.],[1.,4.,0.],[.5,.25,12.]])
    cartesian,resolved=resolve_shift_vectors(lattice,None,direct)
    restored,restored_direct=resolve_shift_vectors(lattice,cartesian,None)
    assert np.array_equal(restored,cartesian)
    assert np.allclose(restored_direct,resolved,rtol=0.,atol=1e-15)


@pytest.mark.parametrize('cartesian,direct,message',[
    ([1.,2.],[1.,2.],'either shift_cartesian or shift_direct'),
    ([1.],None,'shift_cartesian must contain'),
    ([1.]*4,None,'shift_cartesian must contain'),
    (None,[1.],'shift_direct must contain'),
    (None,[1.]*4,'shift_direct must contain'),
])
def test_invalid_shift_vectors_have_clear_errors(cartesian,direct,message):
    with pytest.raises(ValueError,match=message):
        resolve_shift_vectors(np.eye(3),cartesian,direct)


def test_absent_shift_is_zero_in_both_coordinate_modes():
    cartesian,direct=resolve_shift_vectors(np.eye(3),None,None)
    assert np.array_equal(cartesian,np.zeros(3)) and np.array_equal(direct,np.zeros(3))


@pytest.mark.parametrize('height,threshold,expected',[
    (1.,None,None),(1.,2.,('F',)*3),(2.,2.,('T',)*3),
    (3.,-2.,('F',)*3),(2.,-2.,('T',)*3),(1.,0.,('F',)*3),(0.,0.,('T',)*3),
])
def test_relaxation_flags_keep_threshold_sign_and_boundaries(height,threshold,expected):
    assert generator._relax_flags(height,threshold)==expected


def test_affine_transform_preserves_height_flags_and_inputs():
    position=np.array([1.,2.,3.])
    affine=np.array([[2.,1.],[-1.,3.]])
    flags=('F','T','F')
    actual=generator._transform_layer_atoms([('C',position,flags)],affine)
    assert actual[0][0]=='C' and actual[0][2]==flags
    assert np.array_equal(actual[0][1],np.array([4.,5.,3.]))
    assert np.array_equal(position,np.array([1.,2.,3.]))
    assert generator._transform_layer_atoms([],object())==[]


def test_handedness_correction_preserves_cartesian_geometry():
    lattice=np.array([[1.,0.,0.],[0.,-2.,0.],[0.,0.,10.]])
    direct=np.array([[.2,.3,.4]])
    corrected,positions=generator._swap_if_left_handed(lattice,direct)
    assert np.linalg.det(corrected)>0.
    assert np.array_equal(positions@corrected,direct@lattice)
    assert np.array_equal(lattice,np.diag([1.,-2.,10.]))


@pytest.mark.parametrize('vacuum',[-1.,float('nan'),float('inf')])
def test_invalid_vacuum_rejected_before_building_lattice(vacuum):
    with pytest.raises(ValueError,match='finite nonnegative'):
        generator._build_final_lattice(np.array([1.,0.,0.]),np.array([0.,1.,0.]),
            np.array([0.,0.,12.]),[('C',np.array([0.,0.,2.]),None)],1e-4,vacuum)


@pytest.mark.parametrize('backend',['threads','invalid',''])
def test_invalid_parallel_backend_rejected_before_reading(tmp_path,monkeypatch,backend):
    def forbidden(*args,**kwargs):
        pytest.fail('invalid backend accessed input or created artifacts')
    monkeypatch.setattr(generator,'parse_results',forbidden)
    with pytest.raises(ValueError,match='parallel_backend'):
        bilayer.generate_many_from_results('missing.npz',indexes=[1],interlayer_distance=3.,
                output_dir=str(tmp_path/'out'),workers=2,parallel_backend=backend)
    assert not (tmp_path/'out').exists()


def test_thread_backend_does_not_run_process_initializer(tmp_path,monkeypatch,bilayer_document):
    path=write_results(tmp_path/'results.npz',bilayer_document)
    def forbidden():
        pytest.fail('thread backend ran process environment initializer')
    monkeypatch.setattr(bilayer,'_limit_worker_threads',forbidden)
    actual=bilayer.generate_many_from_results(str(path),indexes=[2,1],interlayer_distance=3.35,
                output_dir=str(tmp_path/'out'),workers=2,parallel_backend='thread')
    assert [r.selected_index for r in actual]==[2,1]


def test_parallel_submission_is_bounded_and_preserves_completion_order():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    release=Event()
    completed=[]
    consumed=[]
    def tasks():
        for n in range(40):
            consumed.append(n)
            yield n
    def worker(n):
        if n==0:
            assert release.wait(10), 'later work never ran concurrently'
        else:
            completed.append(n)
            release.set()
        return n
    with ThreadPoolExecutor(max_workers=2) as executor:
        ordered=bilayer._ordered_parallel_results(executor,worker,tasks(),max_pending=4)
        assert next(ordered)==0
        assert completed and len(consumed)<=4
        assert [0,*ordered]==list(range(40))


def test_parallel_error_cancels_pending_work_without_replay():
    from concurrent.futures import Future
    submitted=[]
    class Executor:
        def submit(self,worker,task):
            future=Future()
            if task==0:
                future.set_exception(OSError('write failed'))
            submitted.append((task,future))
            return future
    with pytest.raises(OSError,match='write failed'):
        list(bilayer._ordered_parallel_results(Executor(),lambda x:x,range(1000),max_pending=4))
    assert [n for n,_ in submitted]==[0,1,2,3]
    assert all(f.cancelled() for _,f in submitted[1:])


def test_effective_thread_workers_capped_to_selection_size(tmp_path,monkeypatch,bilayer_document):
    from concurrent.futures import ThreadPoolExecutor
    observed=[]
    def executor(**kwargs):
        observed.append(kwargs)
        return ThreadPoolExecutor(**kwargs)
    monkeypatch.setattr(bilayer,'ThreadPoolExecutor',executor)
    path=write_results(tmp_path/'results.npz',bilayer_document)
    runs=bilayer.generate_many_from_results(str(path),indexes=[2,1],interlayer_distance=3.35,
                output_dir=str(tmp_path/'out'),workers=10000,parallel_backend='thread')
    assert len(runs)==2 and observed==[{'max_workers':2}]


@pytest.mark.parametrize('backend',['process','thread'])
def test_cli_forwards_parallel_backend(monkeypatch,backend):
    from cellstine.cli.parsers import build_parser
    main=importlib.import_module('cellstine.cli.main')
    calls=[]
    class Workflow:
        def make(self,**kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(main,'_workflow',lambda _:Workflow)
    args=build_parser().parse_args(['moire','build','results.npz','--indexes','2,1',
                     '--workers','4','--parallel-backend',backend])
    main.execute_namespace(args)
    assert calls[0]['parallel_backend']==backend and calls[0]['workers']==4


def test_workflow_records_thread_backend_and_rejects_invalid_before_run(tmp_path,bilayer_document):
    from cellstine.moire.moire import Moire
    from cellstine.core.manifests import RunManifest
    path=write_results(tmp_path/'results.npz',bilayer_document)
    runs_root=tmp_path/'runs'
    workflow=Moire(backend='native',runs_root=runs_root,output_root=tmp_path/'output')
    before=set(runs_root.rglob('*'))
    with pytest.raises(ValueError,match='parallel_backend'):
        workflow.make(results_file=str(path),indexes=[1],interlayer_distance=3.35,parallel_backend='invalid')
    assert set(runs_root.rglob('*'))==before
    result=workflow.make(results_file=str(path),indexes=[2,1],interlayer_distance=3.35,
                         workers=2,parallel_backend='thread')
    manifest=RunManifest.load(result.manifest_path)
    assert manifest.parameters['parallel_backend']=='thread'
    assert manifest.summary['generated_count']==2
