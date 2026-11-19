"""Plane requests remain ordinal, ordered and bounded by actual available planes."""
from copy import deepcopy
import builtins
import pytest
from cellstine.defect import layers
from cellstine.defect.records import DefectSite


@pytest.mark.parametrize('spec,bad', [('1-1000000000000',6), ('-1000000000000--1',-1000000000000),
    ('1000000000000..1',6), ('-2..1000000000000',0)])
def test_unavailable_ranges_fail_before_enumerating_them(monkeypatch, spec, bad):
    class GuardedRange:
        def __init__(self, *args):
            self.value = builtins.range(*args)
            self.start, self.stop = self.value.start, self.value.stop

        def __iter__(self):
            assert len(self.value) <= 10, 'unavailable plane range was expanded'
            return iter(self.value)

    monkeypatch.setattr(layers, 'range', GuardedRange, raising=False)
    message = 'plane 0 does not exist' if bad == 0 else f'plane {bad} does not exist'
    with pytest.raises(ValueError, match=message):
        layers.resolve_layer_ids([{'layer_id':i+1} for i in range(5)], spec)


@pytest.mark.parametrize('spec,expected', [('1,-1,2,1',(10,30)), ('-1--3',(10,30)),
    ('3..1',(10,30)), ('all',(10,10,30)), ('middle',(10,)), ('surface',(10,30))])
def test_ordinals_and_duplicates_are_preserved(spec, expected):
    assert layers.resolve_layer_ids([{'layer_id':30},{'layer_id':10},{'layer_id':10}], spec) == expected


def test_layer_index_retains_site_request_and_member_order():
    members = [
        {'indices':[3,4], 'direct':[.3,0,0], 'cartesian':[3.,0,0], 'layer_ids':[2,1,2]},
        {'indices':[1,2], 'direct':[.1,0,0], 'cartesian':[1.,0,0], 'layer_ids':[1]},
        {'indices':[5,6], 'direct':[.5,0,0], 'cartesian':[5.,0,0], 'layer_ids':[2]},
    ]
    site = DefectSite(site_id='pair_001',species='Al-Al',layer_id=1,direct=(.3,0,0),cartesian=(3.,0,0),
        site_kind='divacancy',pair_indices=[3,4],multiplicity=3,members=members)
    saved = deepcopy(site)
    out, notes = layers.sites_by_layer([site], [2,1,2])
    assert [s.site_id for s in out] == ['pair_001_L02','pair_001_L01','pair_001_L02']
    assert [s.pair_indices for s in out] == [[3,4]]*3
    assert [[m['indices'] for m in s.members] for s in out] == [[[3,4],[5,6]],[[3,4],[1,2]],[[3,4],[5,6]]]
    assert [s.multiplicity for s in out] == [2,2,2]
    assert not notes and site == saved
    assert out[0].members[0] is not site.members[0]


def test_unplaced_only_output_does_not_claim_member_expansion():
    site = DefectSite(site_id='adatom_001',species=None,layer_id=None,direct=(0.,0.,.8),
        cartesian=(0.,0.,8.),site_kind='adatom',multiplicity=3)
    out, notes = layers.sites_by_member([site])
    assert out == [site] and out[0] is site
    assert not any('Every member' in note for note in notes)


@pytest.mark.parametrize('requested',[1,2,3])
def test_missing_pair_plane_membership_is_not_inferred_from_a_scalar(requested):
    site = DefectSite(site_id='pair_001',species='Al-Al',layer_id=None,direct=(0.,0.,.5),
        cartesian=(0.,0.,10.),site_kind='divacancy',pair_indices=[1,2],multiplicity=1)
    with pytest.raises(ValueError,match='orbit members.*[Rr]eanal'):
        layers.sites_by_layer([site],[requested])


@pytest.mark.parametrize('kind',['atom','divacancy'])
def test_known_legacy_singleton_retains_its_original_identity_for_all_mode(kind):
    site = DefectSite(site_id='legacy_001',species='Al',layer_id=1 if kind=='atom' else None,
        direct=(0.,0.,.5),cartesian=(0.,0.,10.),site_kind=kind,representative_index=1,
        pair_indices=[1,2] if kind=='divacancy' else [],equivalent_indices=[1] if kind=='atom' else [])
    out,notes = layers.sites_by_member([site])
    assert out == [site] and out[0] is site
    assert not any('Every member' in note for note in notes)
