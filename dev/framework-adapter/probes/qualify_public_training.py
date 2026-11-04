"""Isolated model qualification, NOT a production admission switch.

Runs the measured, unchanged public entry with an installed wheel. Candidate
qualification uses only already-reviewed operator records, never creates model
approval. A human must inspect these results before making a model review.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback


class TensorAudit:
    def __init__(self, root, reference, rank, target=False):
        self.root, self.reference = Path(root), Path(reference) if reference else None
        self.rank, self.target = rank, target
        self.rows=[]
        (self.root/f'tensors/rank{rank}').mkdir(parents=True)

    def tensor(self, name, tensor):
        import torch
        value=tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise AssertionError('Nonfinite '+name)
        rel=f'tensors/rank{self.rank}/{name}.pt'
        row=dict(name=name,shape=list(value.shape),dtype=str(value.dtype),status='saved')
        self.rows.append(row)
        if self.reference is None:
            torch.save(value,self.root/rel)
            return
        expected=torch.load(self.reference/rel,map_location='cpu',weights_only=True)
        if value.shape!=expected.shape or value.dtype!=expected.dtype:
            row['status']='FAIL'; raise AssertionError('Tensor metadata mismatch '+name)
        row['max_abs_diff']=float((value-expected).abs().max()) if value.numel() else 0.0
        try:
            torch.testing.assert_close(value,expected,atol=1e-5 if self.target else 0,
                                       rtol=1e-4 if self.target else 0)
        except AssertionError:
            row['status']='FAIL'
            raise
        row['status']='PASS'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True)
    p.add_argument('--mode',choices=('native','repeat','candidate'),required=True)
    p.add_argument('--reference',type=Path)
    args=p.parse_args()
    if (args.mode=='native') != (args.reference is None): p.error('Only native has no reference')
    import torch
    import flagrt_adapter.training as training
    from flagrt_adapter import train_cli
    from flagrt_adapter.public_entry import ENTRY_SHA,file_hash
    entry=Path(os.environ['DC_ROOT'])/'runtime/proto/proto_train_leg.py'
    if file_hash(entry)!=ENTRY_SHA: p.error('Not the reviewed public entry')
    rank=int(os.environ['RANK']); output=Path(os.environ['DC_OUT_DIR'])
    output.mkdir(parents=True,exist_ok=True)
    audit=TensorAudit(output,args.reference,rank,target=args.mode=='candidate')
    report=dict(status='FAIL',scope='independent qualification, no model approval',
                mode=args.mode,rank=rank,optimizer_steps=0,compared=audit.rows,
                atol=1e-5 if args.mode=='candidate' else 0,rtol=1e-4 if args.mode=='candidate' else 0)
    step=0
    old_step=torch.optim.AdamW.step
    old_backward=torch.Tensor.backward
    old_execute=training.execute_training
    in_backward=False

    def backward(tensor,*a,**kw):
        nonlocal in_backward
        if in_backward: return old_backward(tensor,*a,**kw)
        in_backward=True
        try:
            audit.tensor(f's{step}-loss',tensor)
            return old_backward(tensor,*a,**kw)
        finally: in_backward=False

    def optimizer_step(optimizer,*a,**kw):
        nonlocal step
        parameters=[p for group in optimizer.param_groups for p in group['params']]
        for index,param in enumerate(parameters):
            if param.grad is None: raise AssertionError('Missing gradient '+str(index))
            audit.tensor(f's{step}-grad-{index}',param.grad)
        value=old_step(optimizer,*a,**kw)
        report['optimizer_steps']+=1
        for index,param in enumerate(parameters): audit.tensor(f's{step}-param-{index}',param)
        step+=1
        return value

    def qualify(entry,config,records,actual,runtime,gems,path,**kwargs):
        # Test harness only: no claimed passed model and no production bypass flag.
        if config['schema']!=3 or config.get('workload_reviews') is not None:
            raise ValueError('Qualification must not consume or generate model approvals')
        expected=args.mode=='candidate'
        if config['enabled']!=expected: raise ValueError('Qualification switch mismatch')
        if kwargs.get('workload') is None: raise ValueError('Measured workload missing')
        from dataclasses import asdict
        report['workload']=asdict(kwargs['workload']); report['environment']=asdict(actual)
        report['provider_identity']=kwargs.get('identity')
        probe=dict(config,schema=2,require_target=expected)
        probe.pop('workload_reviews')
        return old_execute(entry,probe,records,actual,runtime,gems,path,**kwargs)

    torch.Tensor.backward=backward; torch.optim.AdamW.step=optimizer_step
    training.execute_training=qualify
    try:
        sys.argv=[sys.argv[0],'--config',args.config]
        train_cli.main()
        if step!=int(os.environ['MAX_STEPS']): raise AssertionError('Incomplete qualification')
        report['status']='PASS'
    except BaseException:
        report['error']=traceback.format_exc()
        raise
    finally:
        torch.Tensor.backward=old_backward; torch.optim.AdamW.step=old_step
        training.execute_training=old_execute
        with (output/f'qualification_rank{rank}.json').open('x') as f: json.dump(report,f,indent=2)


if __name__=='__main__': main()
