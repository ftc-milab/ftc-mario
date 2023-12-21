from ultralytics import YOLO

# Load a model
model = YOLO('best-organizers.pt')  # load a pretrained model (recommended for training)

# Train the model
#results = model.train(data='ftc20.yaml', epochs=100, imgsz=640)
#results = model.train(data='ftc.yaml', epochs=100, imgsz=640,workers=5,device=0)
#results = model.train(data='ftc.yaml', epochs=100, imgsz=640,workers=10)
#results = model.train(data='ftc.yaml', epochs=100, imgsz=640,workers=9)
#results = model.train(data='ftc.yaml', epochs=100, imgsz=640,workers=10,project='ftc-train',name='epoch100')
#results = model.train(data='ftc.yaml', epochs=100, imgsz=640,workers=10,project='ftc-train',name='epoch100-8l')
#results = model.train(data='ftc.yaml', epochs=100, imgsz=640,workers=10,project='ftc-train',name='epoch100-8x')
#results = model.train(data='ftc.yaml', epochs=5, imgsz=640,workers=10,project='ftc-train',name='epoch5-8x')
results = model.train(data='ftc.yaml', epochs=100, imgsz=640,workers=10,project='ftc-train',name='best-organizers-tuned')
