import requests

url = "http://localhost:8000/classify"
files = {"file": open(r"C:\Users\Admin\Desktop\Video Classification\test.mov", "rb")}

response = requests.post(url, files=files)
print(response.json())