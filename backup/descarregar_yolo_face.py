import urllib.request

print("="*50)
print("A TRANSFERIR O YOLOv8-FACE (NANO) DO HUGGINGFACE...")
print("="*50)

# URL Estável do HuggingFace (não dá Erro 404)
url = "https://huggingface.co/spaces/cc1234/stashface/resolve/main/.deepface/weights/yolov8n-face.pt"

try:
    urllib.request.urlretrieve(url, "yolov8n-face.pt")
    print("[SUCESSO] O ficheiro yolov8n-face.pt foi descarregado com sucesso!")
    print("Agora sim, a arena de benchmarking está pronta!")
except Exception as e:
    print(f"[ERRO] Falha na transferência: {e}")