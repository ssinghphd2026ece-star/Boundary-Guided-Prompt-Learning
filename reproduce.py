"""Independent SAR-BAP-SAM implementation; see README for limitations."""
import argparse, copy, json, math, random
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SIZE=1024; LOW=256; NQ=100

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def write_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f: json.dump(obj,f,indent=2)

def split_coco(args):
    with open(args.dev_json) as f: data=json.load(f)
    ids=sorted(int(x['id']) for x in data['images'])
    if len(ids)!=928: raise ValueError(f'Expected 928 development images; found {len(ids)}')
    rng=np.random.default_rng(args.seed); rng.shuffle(ids)
    for name, subset in [('train',ids[:789]),('val',ids[789:])]:
        chosen=set(subset)
        out={**data,'images':[x for x in data['images'] if x['id'] in chosen],
             'annotations':[x for x in data['annotations'] if x['image_id'] in chosen]}
        write_json(out,Path(args.out)/f'{name}.json')
    write_json({'seed':args.seed,'train_ids':ids[:789],'val_ids':ids[789:]},Path(args.out)/'manifest.json')
    print('Generated a NEW split; manuscript original IDs are unavailable.')

def resolve_image(root, info):
    p = Path(info['file_name'])
    if p.is_file():
        return p
    root = Path(root)
    # Official BBox_RBox_PSeg_SSDD COCO JSON stores bare filenames;
    # the archive separates train and test image directories.
    candidates = [root / p, root / 'train' / p, root / 'test' / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'Image {p} missing under {root} (checked root/train/test)')

def get_masks(coco,info):
    anns=coco.loadAnns(coco.getAnnIds(imgIds=[info['id']]))
    masks=[]
    for a in anns:
        if a.get('iscrowd',0): continue
        m=coco.annToMask(a).astype(np.uint8)
        if m.any(): masks.append(m)
    return masks

class SSDD(Dataset):
    def __init__(self, json_path, root, train=False):
        self.coco=COCO(str(json_path)); self.ids=sorted(self.coco.getImgIds()); self.root=root; self.train=train
    def __len__(self): return len(self.ids)
    def __getitem__(self,idx):
        info=self.coco.loadImgs([self.ids[idx]])[0]
        img=cv2.imread(str(resolve_image(self.root,info)),cv2.IMREAD_GRAYSCALE)
        if img is None: raise FileNotFoundError(info['file_name'])
        masks=get_masks(self.coco,info); h,w=img.shape
        # Manuscript does not publish the exact training percentile statistics.
        # Explicit per-image 1st/99th percentile scaling is a documented approximation.
        lo,hi=np.percentile(img,[1,99]); x=np.clip((img.astype(np.float32)-lo)/max(hi-lo,1),0,1)
        x=cv2.resize(x,(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)
        mm=np.stack([cv2.resize(m,(LOW,LOW),interpolation=cv2.INTER_NEAREST) for m in masks]) if masks else np.zeros((0,LOW,LOW),np.uint8)
        if self.train and random.random()<0.5:
            x=np.ascontiguousarray(x[:,::-1]); mm=np.ascontiguousarray(mm[:,:,::-1])
        return {'image':torch.from_numpy(x.copy()).float().unsqueeze(0),
                'masks':torch.from_numpy(mm.copy()).float(), 'id':int(info['id']),
                'orig_hw':(h,w),'info':info}

def collate(batch): return batch

def metrics(pred,gt):
    p=pred.astype(bool); g=gt.astype(bool)
    tp=np.logical_and(p,g).sum(); fp=np.logical_and(p,~g).sum(); fn=np.logical_and(~p,g).sum()
    e=1e-9
    return {'iou':tp/(tp+fp+fn+e),'dice':2*tp/(2*tp+fp+fn+e),
            'precision':tp/(tp+fp+e),'recall':tp/(tp+fn+e)}

def classical(img):
    _,otsu=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur=cv2.GaussianBlur(img,(5,5),1)
    _,go=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(img)
    _,co=cv2.threshold(clahe,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    adaptive=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    top=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    _,to=cv2.threshold(top,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    opened=cv2.morphologyEx(go,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    return {'Otsu':otsu,'Gaussian-Otsu':go,'CLAHE-Otsu':co,'Adaptive Gaussian':adaptive,
            'Top-hat-Otsu':to,'Gaussian-Otsu + morphology':opened}

def baseline(args):
    ds=SSDD(args.json,args.images); rows=[]
    scene_lookup = {}
    if args.scene_json:
        with open(args.scene_json) as f:
            scene_lookup = {int(k): v for k, v in json.load(f).items()}
    for info in tqdm([ds.coco.loadImgs([i])[0] for i in ds.ids],desc='Baselines'):
        img=cv2.imread(str(resolve_image(args.images,info)),0)
        gt=np.zeros(img.shape,np.uint8)
        for m in get_masks(ds.coco,info): gt|=m
        scene=scene_lookup.get(int(info['id']), 'unknown')
        for method,p in classical(img).items():
            rows.append({'image_id':info['id'],'scene':scene,'method':method,'gt_area':int(gt.sum()),**metrics(p>0,gt>0)})
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/'per_image.csv',index=False)
    summary=df.groupby('method')[['iou','dice','precision','recall']].mean()
    rng=np.random.default_rng(2026); intervals=[]
    for name,sub in df.groupby('method',sort=False):
        values=sub.iou.to_numpy(); means=np.mean(rng.choice(values,(2000,len(values)),replace=True),axis=1)
        intervals.append({'method':name,'ci_low':np.percentile(means,2.5),'ci_high':np.percentile(means,97.5),'iou_sd':np.std(values,ddof=1)})
    summary=summary.merge(pd.DataFrame(intervals).set_index('method'),left_index=True,right_index=True)
    summary.to_csv(out/'summary.csv'); print(summary.to_string())
    df.groupby(['method','scene'])[['iou','dice','precision','recall']].mean().to_csv(out/'scene_summary.csv')
    ax=summary[['iou','dice','precision','recall']].plot.bar(figsize=(12,5));ax.set_ylim(0,1);plt.tight_layout();plt.savefig(out/'baseline_metrics.png',dpi=220);plt.close()

class ResidualAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.path=nn.Sequential(nn.Conv2d(1,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),
                                nn.Conv2d(64,256,3,padding=1),nn.BatchNorm2d(256),nn.GELU())
        self.skip=nn.Conv2d(1,256,1)
    def forward(self,x):
        x=F.interpolate(x,size=(64,64),mode='bilinear',align_corners=False)
        return self.path(x)+self.skip(x)

class GatedFusion(nn.Module):
    def __init__(self,n=5):
        super().__init__();self.gate=nn.Sequential(nn.Linear(256*n,256),nn.GELU(),nn.Linear(256,n))
    def forward(self,feats):
        weights=self.gate(torch.cat([f.mean((2,3)) for f in feats],1)).softmax(-1)
        return sum(f*weights[:,i,None,None,None] for i,f in enumerate(feats)),weights

class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,256,3,padding=1),nn.BatchNorm2d(256),nn.GELU(),
                               nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.GELU(),
                               nn.Conv2d(256,1,1))
        self.edge=nn.Conv2d(256,1,1)
    def forward(self,logits,gray,local):
        # logits: B,Q,256,256; memory-conscious per-query refinement
        b,q,h,w=logits.shape
        low=F.interpolate(local,size=(h,w),mode='bilinear',align_corners=False)
        gray=F.interpolate(gray,size=(h,w),mode='bilinear',align_corners=False)
        sx=gray.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)
        sy=sx.transpose(2,3)
        sobel=torch.sqrt(F.conv2d(gray,sx,padding=1).square()+F.conv2d(gray,sy,padding=1).square()+1e-6)
        edge=self.edge(low)
        result=[]
        for k in range(q):
            inp=torch.cat([logits[:,k:k+1],sobel,edge],dim=1)
            result.append(logits[:,k:k+1]+self.net(inp))
        return torch.cat(result,dim=1)

class SARBAPSAM(nn.Module):
    def __init__(self,sam_checkpoint,refine=True,queries=100,chunk=10):
        super().__init__()
        from segment_anything import sam_model_registry
        self.sam=sam_model_registry['vit_b'](checkpoint=str(sam_checkpoint))
        for p in self.sam.parameters():p.requires_grad_(False)
        self.sam.eval();self.refine=refine;self.queries=queries;self.chunk=chunk
        self.adapter=ResidualAdapter();self.proj=nn.ModuleList([nn.Conv2d(768,256,1) for _ in range(4)])
        self.fusion=GatedFusion(5)
        layer=nn.TransformerDecoderLayer(d_model=256,nhead=8,dim_feedforward=1024,batch_first=True,dropout=0.1)
        self.decoder=nn.TransformerDecoder(layer,num_layers=3)
        self.query=nn.Embedding(queries,256);self.cls=nn.Linear(256,2);self.prompt=nn.Linear(256,256)
        self.boundary=BoundaryHead() if refine else None
    def train(self,mode=True):
        super().train(mode);self.sam.eval();return self
    def forward(self,x):
        b=x.shape[0];saved=[]
        def hook(module,inputs,output):saved.append(output)
        handles=[self.sam.image_encoder.blocks[i].register_forward_hook(hook) for i in (2,5,8,11)]
        try:
            rgb=x.repeat(1,3,1,1)*255.
            rgb=(rgb-self.sam.pixel_mean)/self.sam.pixel_std
            # no_grad would prevent trainable projections from receiving features? detached outputs are fine.
            with torch.no_grad(): base=self.sam.image_encoder(rgb)
        finally:
            for h in handles:h.remove()
        feats=[]
        for f,p in zip(saved,self.proj):
            # SAM ViT-B blocks return B,H,W,768
            feats.append(F.interpolate(p(f.detach().permute(0,3,1,2)),size=(64,64),mode='bilinear',align_corners=False))
        local=self.adapter(x);feats.append(local)
        fused,gates=self.fusion(feats)
        memory=fused.flatten(2).transpose(1,2)
        z=self.decoder(self.query.weight[None].expand(b,-1,-1),memory)
        scores=self.cls(z);sparse=self.prompt(z)
        masks=[]
        # Frozen SAM mask decoder remains differentiable with respect to fused image and sparse prompts.
        dense_pe=self.sam.prompt_encoder.get_dense_pe()
        dense_prompt=self.sam.prompt_encoder.no_mask_embed.weight.reshape(1,-1,1,1).expand(b,-1,64,64)
        for j in range(0,self.queries,self.chunk):
            block=[]
            for k in range(j,min(j+self.chunk,self.queries)):
                m,_=self.sam.mask_decoder(image_embeddings=fused,image_pe=dense_pe.expand(b,-1,-1,-1),
                    sparse_prompt_embeddings=sparse[:,k:k+1],dense_prompt_embeddings=dense_prompt,
                    multimask_output=False)
                block.append(m)
            masks.append(torch.cat(block,1))
        masks=torch.cat(masks,1)
        if self.boundary is not None: masks=self.boundary(masks,x,local)
        return {'logits':masks,'classes':scores,'gates':gates}

def dice_loss(logits,target):
    p=logits.sigmoid().flatten(1);t=target.flatten(1)
    return 1-(2*(p*t).sum(1)+1)/((p+t).sum(1)+1)

def boundary_band(target):
    dil=F.max_pool2d(target,7,stride=1,padding=3)
    ero=-F.max_pool2d(-target,7,stride=1,padding=3)
    return (dil-ero).clamp(0,1)

def hungarian(logits,classes,gt):
    n=gt.shape[0]
    if n==0:return np.array([],dtype=int),np.array([],dtype=int)
    # matching cost is computed without gradient; mask BCE cost via efficient pixel subsampling
    with torch.no_grad():
        prob=logits.sigmoid().flatten(1)[:,::16]
        target=gt.flatten(1)[:,::16]
        bce=-(target[None]*prob[:,None].clamp(1e-6,1-1e-6).log()+
              (1-target[None])*(1-prob[:,None]).clamp(1e-6,1-1e-6).log()).mean(-1)
        inter=prob@target.T
        dice=1-(2*inter+1)/(prob.sum(-1)[:,None]+target.sum(-1)[None,:]+1)
        cost=2*(-classes.softmax(-1)[:,1,None])+5*bce+5*dice
        return linear_sum_assignment(cost.cpu().numpy())

def supervised_loss(output,targets):
    total=output['logits'].new_zeros(());b=output['logits'].shape[0]
    for i,gt in enumerate(targets):
        logits=output['logits'][i];cls=output['classes'][i]
        rows,cols=hungarian(logits,cls,gt)
        labels=torch.zeros(cls.shape[0],device=cls.device,dtype=torch.long)
        if len(rows): labels[torch.as_tensor(rows,device=cls.device)]=1
        lcls=F.cross_entropy(cls,labels)
        if len(rows):
            pr=logits[rows];tg=gt[cols]
            lbce=F.binary_cross_entropy_with_logits(pr,tg)
            ldice=dice_loss(pr,tg).mean()
            # boundary supervision on the predicted mask boundary probabilities
            p=pr.sigmoid().unsqueeze(1);pred_boundary=boundary_band(p)
            gt_boundary=boundary_band(tg.unsqueeze(1))
            lboundary=F.binary_cross_entropy(pred_boundary.clamp(1e-5,1-1e-5),gt_boundary)
        else:lbce=ldice=lboundary=logits.sum()*0
        total=total+lcls+2*lbce+2*ldice+lboundary
    return total/b

def consistency_loss(a,b):
    # b is prediction of horizontally flipped image, mapped back to a's coordinates.
    # Cross-view Hungarian matching avoids assuming query index identity.
    loss=a['logits'].new_zeros(())
    for i in range(a['logits'].shape[0]):
        pa=a['logits'][i].sigmoid();pb=b['logits'][i].flip(-1).sigmoid()
        with torch.no_grad():
            x=F.adaptive_avg_pool2d(pa.unsqueeze(1),(32,32)).flatten(1)
            y=F.adaptive_avg_pool2d(pb.unsqueeze(1),(32,32)).flatten(1)
            overlap=x@y.T
            cost=1-(2*overlap+1)/(x.sum(1)[:,None]+y.sum(1)[None,:]+1)
            r,c=linear_sum_assignment(cost.cpu().numpy())
        loss=loss+F.l1_loss(pa[r],pb[c])
    return loss/a['logits'].shape[0]

def mask_nms(masks,scores,threshold=0.7):
    order=np.argsort(-scores);keep=[]
    for k in order:
        if all((np.logical_and(masks[k],masks[t]).sum() /
                max(np.logical_or(masks[k],masks[t]).sum(),1))<=threshold for t in keep):keep.append(k)
    return keep

def predictions(output,info,score_threshold=0.30):
    scores=output['classes'][0].softmax(-1)[:,1].detach().cpu().numpy()
    indices=np.where(scores>=score_threshold)[0]
    if not len(indices):return [],np.zeros((info['height'],info['width']),np.uint8)
    logits=output['logits'][0,indices].detach()
    masks=[]
    for m in logits:
        up=F.interpolate(m[None,None],size=(info['height'],info['width']),mode='bilinear',align_corners=False)[0,0]
        masks.append((up.sigmoid()>0.5).cpu().numpy())
    scores=scores[indices];kept=mask_nms(masks,scores)
    results=[];union=np.zeros(masks[0].shape,np.uint8)
    for k in kept[:100]:
        m=masks[k].astype(np.uint8)
        if not m.any():continue
        union|=m
        rle=mask_utils.encode(np.asfortranarray(m));rle['counts']=rle['counts'].decode('ascii')
        ys,xs=np.where(m);bbox=[float(xs.min()),float(ys.min()),float(xs.max()-xs.min()+1),float(ys.max()-ys.min()+1)]
        results.append({'image_id':int(info['id']),'category_id':0,'segmentation':rle,
                        'bbox':bbox,'score':float(scores[k])})
    return results,union

def run_evaluation(model,ds,device,out,tag='val'):
    model.eval();results=[];rows=[]
    for sample in tqdm(ds,desc=f'Evaluate {tag}'):
        x=sample['image'][None].to(device)
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            output=model(x)
        pred,union=predictions(output,sample['info']);results.extend(pred)
        gt=np.zeros(union.shape,np.uint8)
        for m in get_masks(ds.coco,sample['info']):gt|=m
        rows.append({'image_id':sample['id'],'gt_area':int(gt.sum()),**metrics(union,gt)})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_json(results,out/f'{tag}_predictions.json')
    pd.DataFrame(rows).to_csv(out/f'{tag}_foreground.csv',index=False)
    summary=pd.DataFrame(rows)[['iou','dice','precision','recall']].mean().to_dict()
    if results:
        # pycocotools expects predictions' category ID to match the COCO category ID.
        coco=ds.coco
        category_ids=coco.getCatIds()
        if len(category_ids)!=1:raise ValueError('SSDD evaluation requires one ship category')
        for r in results:r['category_id']=category_ids[0]
        write_json(results,out/f'{tag}_predictions.json')
        dt=coco.loadRes(results);ev=COCOeval(coco,dt,'segm');ev.params.imgIds=ds.ids
        ev.params.catIds=category_ids;ev.params.maxDets=[1,10,100]
        ev.evaluate();ev.accumulate();ev.summarize()
        summary.update({'AP':float(ev.stats[0]),'AP50':float(ev.stats[1]),'AP75':float(ev.stats[2])})
        np.savez_compressed(out/f'{tag}_coco_precision.npz',precision=ev.eval['precision'],recall=ev.eval['recall'])
    else:summary.update({'AP':0.,'AP50':0.,'AP75':0.})
    write_json(summary,out/f'{tag}_metrics.json');print(tag,summary)
    return summary

def train(args):
    seed_all(args.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda':print('WARNING: CPU training of SAM ViT-B will be extremely slow.')
    train_ds=SSDD(args.train_json,args.images,True);val_ds=SSDD(args.val_json,args.images)
    loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,collate_fn=collate)
    model=SARBAPSAM(args.sam_checkpoint,refine=not args.no_boundary,queries=args.queries,chunk=args.chunk).to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4,weight_decay=.05)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);best=-1
    for epoch in range(args.epochs):
        if epoch<3:lr=1e-4*(epoch+1)/3
        else:lr=1e-6+(1e-4-1e-6)*0.5*(1+math.cos(math.pi*(epoch-3)/max(args.epochs-3,1)))
        for group in optimizer.param_groups:group['lr']=lr
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for step,batch in enumerate(tqdm(loader,desc=f'Epoch {epoch+1}/{args.epochs}')):
            x=torch.stack([v['image'] for v in batch]).to(device)
            targets=[v['masks'].to(device) for v in batch]
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                pred=model(x);loss=supervised_loss(pred,targets)
                if not args.no_consistency:
                    flipped=model(x.flip(-1))
                    loss=loss+0.2*consistency_loss(pred,flipped)
                loss=loss/args.accum
            scaler.scale(loss).backward();losses.append(loss.item()*args.accum)
            if (step+1)%args.accum==0 or step+1==len(loader):
                scaler.unscale_(optimizer);nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        print('Training mean loss:',float(np.mean(losses)))
        val=run_evaluation(model,val_ds,device,out/f'epoch_{epoch+1:02d}','val')
        torch.save({'model':model.state_dict(),'epoch':epoch+1,'val':val,'args':vars(args)},out/'last.pt')
        if val['AP']>best:
            best=val['AP'];torch.save({'model':model.state_dict(),'epoch':epoch+1,'val':val,'args':vars(args)},out/'best.pt')
    print('Best validation mask AP:',best)

def evaluate(args):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    saved=checkpoint.get('args',{});model=SARBAPSAM(args.sam_checkpoint,refine=not saved.get('no_boundary',False),
        queries=saved.get('queries',100),chunk=args.chunk).to(device)
    model.load_state_dict(checkpoint['model']);ds=SSDD(args.json,args.images)
    run_evaluation(model,ds,device,args.out,'test')

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('split');s.add_argument('--dev-json',required=True);s.add_argument('--out',required=True);s.add_argument('--seed',type=int,default=2026)
    s=sub.add_parser('baselines');s.add_argument('--json',required=True);s.add_argument('--images',required=True);s.add_argument('--out',required=True);s.add_argument('--scene-json',default=None)
    s=sub.add_parser('train');s.add_argument('--train-json',required=True);s.add_argument('--val-json',required=True)
    s.add_argument('--images',required=True);s.add_argument('--sam-checkpoint',required=True);s.add_argument('--out',required=True)
    s.add_argument('--epochs',type=int,default=40);s.add_argument('--batch-size',type=int,default=2);s.add_argument('--accum',type=int,default=4)
    s.add_argument('--workers',type=int,default=0);s.add_argument('--seed',type=int,default=2026)
    s.add_argument('--queries',type=int,default=100);s.add_argument('--chunk',type=int,default=10)
    s.add_argument('--no-boundary',action='store_true');s.add_argument('--no-consistency',action='store_true')
    s=sub.add_parser('evaluate');s.add_argument('--json',required=True);s.add_argument('--images',required=True)
    s.add_argument('--sam-checkpoint',required=True);s.add_argument('--checkpoint',required=True);s.add_argument('--out',required=True)
    s.add_argument('--chunk',type=int,default=10)
    args=p.parse_args();{'split':split_coco,'baselines':baseline,'train':train,'evaluate':evaluate}[args.command](args)
if __name__=='__main__':main()
