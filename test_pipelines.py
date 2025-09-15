import requests

# Probar listar pipelines
r = requests.get("http://127.0.0.1:8000/v1/pipelines")
print(r.status_code, r.json())
 
