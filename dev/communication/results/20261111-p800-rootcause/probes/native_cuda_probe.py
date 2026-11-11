"""Torch-free CUDA-compatibility runtime probe; no reset or collectives."""
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
        'scope':'Native CUDA-compatible allocation/memset/copy; no torch, FlagCX or runtime prototype',
        'steps':[],'verdict':'ERROR'}
def save(): args.output.write_text(json.dumps(result,indent=2)+'\n')
def call(name,types,values):
    result['stage']=name+'_begin';save();print(result['stage'],flush=True)
    function=getattr(library,name)
    function.argtypes=types;function.restype=C.c_int
    start=time.monotonic();code=function(*values)
    result['steps'].append({'function':name,'code':code,'seconds':time.monotonic()-start})
    save();print(name,'return',code,flush=True)
    if code:raise RuntimeError(f'{name}: {code}')
faulthandler.enable()
faulthandler.dump_traceback_later(15,repeat=True)
library=C.CDLL(args.library)
count=C.c_int()
call('cudaGetDeviceCount',[C.POINTER(C.c_int)],[C.byref(count)])
identities=[]
for index in range(count.value):
    bus=C.create_string_buffer(64)
    call('cudaDeviceGetPCIBusId',[C.c_char_p,C.c_int,C.c_int],[bus,64,index])
    identities.append(bus.value.decode())
result['pci_order']=identities;save()
normalize=lambda value: value.lower().replace('00000000:','0000:')
chosen=[index for index,pci in enumerate(identities) if normalize(pci)==normalize(args.pci)]
if len(chosen)!=1:raise ValueError('Exactly one matching PCI identity required before selecting device')
device=chosen[0]
call('cudaSetDevice',[C.c_int],[device])
pointer=C.c_void_p()
call('cudaMalloc',[C.POINTER(C.c_void_p),C.c_size_t],[C.byref(pointer),16])
call('cudaMemset',[C.c_void_p,C.c_int,C.c_size_t],[pointer,1,16])
call('cudaDeviceSynchronize',[],[])
host=(C.c_uint32*4)()
call('cudaMemcpy',[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int],[host,pointer,16,2])
result['readback']=list(host)
if result['readback']!=[16843009]*4:raise AssertionError(result['readback'])
call('cudaFree',[C.c_void_p],[pointer])
result.update(verdict='PASS',stage='complete');save()
print('PASS',flush=True)
