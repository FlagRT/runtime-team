"""PPU idle observation; a point-in-time check, never a reservation."""
import re

def require_idle(smi,card):
    lines=smi.splitlines()
    i=next(i for i,s in enumerate(lines) if re.search(r'\|\s*'+str(card)+r'\s+PPU-',s))
    m=re.search(r'\|\s*(\d+)MiB\s*/\s*\d+MiB\s*\|\s*(\d+)%',lines[i+1])
    if not m or int(m[1])>64 or int(m[2])!=0:
        raise RuntimeError('Card not idle at start')
    if 'Processes:' not in smi:
        raise RuntimeError('Missing process inventory')
    if re.search(r'\|\s*'+str(card)+r'\s+N/A\s+N/A\s+\d+',smi.split('Processes:',1)[1]):
        raise RuntimeError('Card has a process, including zero-MiB processes')
    return dict(memory_mib=int(m[1]),utilization=0,processes=0)
