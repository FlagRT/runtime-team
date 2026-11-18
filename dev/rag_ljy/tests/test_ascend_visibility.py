import pytest
from rag_engine.ascend_visibility import check_pinned_visibility, embedding_visible_device


def test_default_embedding_index_preserves_the_pinned_host_die():
    env={'ASCEND_RT_VISIBLE_DEVICES':'7','RAG_ASCEND_PINNED_DIE':'7'}
    assert embedding_visible_device(0,env)=='7'
    assert env['ASCEND_RT_VISIBLE_DEVICES']=='7'
    with pytest.raises(ValueError,match='outside'):
        embedding_visible_device(1,env)


def test_unpinned_legacy_and_inherited_device_lists():
    assert embedding_visible_device(0,{})=='0'
    assert embedding_visible_device(3,{})=='3'
    assert embedding_visible_device(1,{'ASCEND_RT_VISIBLE_DEVICES':'4,7'})=='7'


@pytest.mark.parametrize('selection',[None,'0','6','0,7'])
def test_changed_selection_fails_before_device_initialization(selection):
    env={'RAG_ASCEND_PINNED_DIE':'7'}
    if selection is not None:
        env['ASCEND_RT_VISIBLE_DEVICES']=selection
    with pytest.raises(RuntimeError,match='refusing'):
        check_pinned_visibility(env)


@pytest.mark.parametrize('selection',['-1','7,7','0-7',''])
def test_invalid_masks_do_not_fall_back_to_host_zero(selection):
    env={'ASCEND_RT_VISIBLE_DEVICES':selection}
    if not selection:
        env['RAG_ASCEND_PINNED_DIE']='7'
    with pytest.raises((ValueError,RuntimeError)):
        embedding_visible_device(0,env)
