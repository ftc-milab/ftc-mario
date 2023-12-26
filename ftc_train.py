from ultralytics import YOLO
import pandas as pd
import os
# Load a model


df=pd.read_csv('param-train.csv')

for index, row in df.iterrows():
    print('(',index+1,'/',len(df),') rows')
    model = YOLO(row.model)  # load a pretrained model (recommended for training)
    if os.path.exists(f"weights/{row.exp_id}"):
        print("Already exists. Next")
        continue
    results = model.train(data=row.data,\
                            epochs=row.epochs,\
                            patience=row.patience,\
                            batch=row.batch,\
                            imgsz=row.imgsz,\
                            save=row.save,\
                            save_period=row.save_period,\
                            cache=row.cache,\
                            # device=row.device,\
                            workers=row.workers,\
                            project="weights",\
                            name=row.exp_id,\
                            exist_ok=row.exist_ok,\
                            pretrained=row.pretrained,\
                            # optimizer=row.optimizer,\
                            verbose=row.verbose,\
                            seed=row.seed,\
                            deterministic=row.deterministic,\
                            single_cls=row.single_cls,\
                            rect=row.rect,\
                            cos_lr=row.cos_lr,\
                            close_mosaic=row.close_mosaic,\
                            resume=row.resume,\
                            amp=row.amp,\
                            fraction=row.fraction,\
                            profile=row.profile,\
                            freeze=row.freeze,\
                            lr0=row.lr0,\
                            lrf=row.lrf, \
                            momentum=row.momentum,\
                            weight_decay=row.weight_decay,\
                            warmup_epochs=row.warmup_epochs,\
                            warmup_momentum=row.warmup_momentum,\
                            warmup_bias_lr=row.warmup_bias_lr,\
                            box=row.box,cls=row.cls,\
                            dfl=row.dfl,\
                            pose=row.pose,kobj=row.kobj,\
                            label_smoothing=row.label_smoothing,\
                            nbs=row.nbs,\
                            overlap_mask=row.overlap_mask,\
                            mask_ratio=row.mask_ratio,\
                            dropout=row.dropout,\
                            val=row.val,plots=row.plots)