import urllib.request
import os

print("="*50)
print("A DESCARREGAR MODELOS SSD DO OPENCV...")
print("="*50)

prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

try:
    print("A transferir a arquitetura (deploy.prototxt)...")
    urllib.request.urlretrieve(prototxt_url, "deploy.prototxt.txt")
    print("[OK] Arquitetura transferida!")

    print("A transferir os pesos do modelo (caffemodel) - Isto pode demorar uns segundos...")
    urllib.request.urlretrieve(model_url, "res10_300x300_ssd_iter_140000.caffemodel")
    print("[OK] Pesos transferidos!")
    
    print("\n[SUCESSO] Os ficheiros do SSD estão prontos! Podes executar o Módulo 2.")
except Exception as e:
    print(f"\n[ERRO] Falha na transferência: {e}")