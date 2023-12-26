s="exp_id,model,data,epochs,time,patience,batch,imgsz,save,save_period,cache,device,workers,project,name,exist_ok,pretrained,optimizer,verbose,seed,deterministic,single_cls,rect,cos_lr,close_mosaic,resume,amp,fraction,profile,freeze,lr0,lrf,momentum,weight_decay,warmup_epochs,warmup_momentum,warmup_bias_lr,box,cls,dfl,pose,kobj,label_smoothing,nbs,overlap_mask,mask_ratio,dropout,val,plots"

for ss in s.split(','):
    print(f"{ss}=row.{ss},",end="")