"""Torch-free vendor CUDA driver probe. Uses one primary context; never reset."""
import argparse
import ctypes as C
import faulthandler
import json
import os
from pathlib import Path
import time

p=argparse.ArgumentParser()
p.add_argument('--library',required=True)
p.add_argument('--pci',required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
if args.output.exists(): raise FileExistsError(args.output)
result={'pid':os.getpid(),'library':args.library,'requested_pci':args.pci,
        'scope':'Vendor driver API only; no torch, FlagCX or device prototype',
        'steps':[],'verdict':'ERROR'}
def save(): args.output.write_text(json.dumps(result,indent=2)+'\n')
def call(name,types,values):
    result['stage']=name+'_begin';save();print(result['stage'],flush=True)
    function=getattr(library,name);function.argtypes=types;function.restype=C.c_int
    start=time.monotonic();code=function(*values)
    result['steps'].append({'function':name,'code':code,'seconds':time.monotonic()-start})
    save();print(name,'return',code,flush=True)
    if code:raise RuntimeError(f'{name}: {code}')
faulthandler.enable();faulthandler.dump_traceback_later(15,repeat=True)
library=C.CDLL(args.library)
call('cuInit',[C.c_uint],[0])
count=C.c_int()
call('cuDeviceGetCount',[C.POINTER(C.c_int)],[C.byref(count)])
identities=[]
devices=[]
for index in range(count.value):
    device=C.c_int();bus=C.create_string_buffer(64)
    call('cuDeviceGet',[C.POINTER(C.c_int),C.c_int],[C.byref(device),index])
    call('cuDeviceGetPCIBusId',[C.c_char_p,C.c_int,C.c_int],[bus,64,device.value])
    identities.append(bus.value.decode());devices.append(device.value)
result['pci_order']=identities;save()
normalize=lambda value:value.lower().replace('00000000:','0000:')
chosen=[index for index,pci in enumerate(identities) if normalize(pci)==normalize(args.pci)]
if len(chosen)!=1:raise ValueError('Exactly one matching PCI identity required before context creation')
device=devices[chosen[0]]
context=C.c_void_p()
call('cuDevicePrimaryCtxRetain',[C.POINTER(C.c_void_p),C.c_int],[C.byref(context),device])
call('cuCtxSetCurrent',[C.c_void_p],[context])
pointer=C.c_uint64()
call('cuMemAlloc_v2',[C.POINTER(C.c_uint64),C.c_size_t],[C.byref(pointer),16])
call('cuMemsetD32_v2',[C.c_uint64,C.c_uint,C.c_size_t],[pointer.value,0x3f800000,4])
call('cuCtxSynchronize',[],[])
host=(C.c_uint32*4)()
call('cuMemcpyDtoH_v2',[C.c_void_p,C.c_uint64,C.c_size_t],[host,pointer.value,16])
result['readback']=list(host)
if result['readback']!=[0x3f800000]*4:raise AssertionError(result['readback'])
call('cuMemFree_v2',[C.c_uint64],[pointer.value])
call('cuDevicePrimaryCtxRelease',[C.c_int],[device])
result.update(verdict='PASS',stage='complete');save();print('PASS',flush=True)
