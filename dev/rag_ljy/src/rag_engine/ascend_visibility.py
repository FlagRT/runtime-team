"""Keep the selected host die consistent across separately launched processes.

ASCEND_RT_VISIBLE_DEVICES is an application-level selection, not Docker device
isolation. In a privileged container all host device nodes may remain accessible.
"""
import os


def check_pinned_visibility(environ=None):
    env=os.environ if environ is None else environ
    pinned=env.get('RAG_ASCEND_PINNED_DIE')
    if pinned is None:
        return
    if not pinned.isdecimal() or env.get('ASCEND_RT_VISIBLE_DEVICES')!=pinned:
        raise RuntimeError('Ascend selection differs from RAG_ASCEND_PINNED_DIE; refusing to use another host die')


def embedding_visible_device(index, environ=None):
    """Select a logical index within an inherited visible-device list.

    Previously this launcher always overwrote an inherited physical selection
    with '0'. That would select host die 0 in a privileged all-device container.
    Without an inherited list, preserve the existing index behavior.
    """
    env=os.environ if environ is None else environ
    check_pinned_visibility(env)
    if isinstance(index,bool) or not isinstance(index,int) or index<0:
        raise ValueError('Device index must be a nonnegative integer')
    value=env.get('ASCEND_RT_VISIBLE_DEVICES','').strip()
    if not value:
        return str(index)
    devices=[v.strip() for v in value.split(',')]
    if any(not v.isdecimal() for v in devices) or len(set(devices))!=len(devices):
        raise ValueError('ASCEND_RT_VISIBLE_DEVICES must be a comma-separated list of device numbers')
    if index>=len(devices):
        raise ValueError('Device index is outside the inherited ASCEND_RT_VISIBLE_DEVICES selection')
    return devices[index]
