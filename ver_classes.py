from ultralytics import YOLO

modelo = YOLO('best.pt') 
print("As classes exatas do seu modelo são:")
print(modelo.names)