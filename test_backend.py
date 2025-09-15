import requests  
r = requests.get("http://127.0.0.1:8000/health")  
print(r.status_code, r.json()) 
