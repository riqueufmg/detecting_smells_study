import os
import requests

ACCESS_TOKEN = "X9s8Vwhs2PBp7YchhXxcj8Bt1SgwkiAhRuhhkkhS0O60lapIxI4YzQoPlF8b"
FILEPATH = "replication_package.zip"

headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# 1) Criar draft
r = requests.post(
    "https://zenodo.org/api/deposit/depositions",
    json={},
    headers={**headers, "Content-Type": "application/json"},
)
print("create:", r.status_code)
print(r.text)
r.raise_for_status()

data = r.json()
deposit_id = data["id"]
bucket_url = data["links"]["bucket"]
html_url = data["links"]["latest_draft_html"]

print("deposit_id:", deposit_id)
print("draft:", html_url)

# 2) Upload do arquivo para o bucket
filename = os.path.basename(FILEPATH)
with open(FILEPATH, "rb") as fp:
    r = requests.put(
        f"{bucket_url}/{filename}",
        data=fp,
        headers=headers,
    )

print("upload:", r.status_code)
print(r.text)
r.raise_for_status()

# 3) Metadados mínimos
metadata = {
    "metadata": {
        "title": "Replication package",
        "upload_type": "dataset",
        "description": "Replication package do estudo.",
        "creators": [{"name": "Anonymous"}],
    }
}

r = requests.put(
    f"https://zenodo.org/api/deposit/depositions/{deposit_id}",
    json=metadata,
    headers={**headers, "Content-Type": "application/json"},
)
print("metadata:", r.status_code)
print(r.text)
r.raise_for_status()

# 4) Publicar
r = requests.post(
    f"https://zenodo.org/api/deposit/depositions/{deposit_id}/actions/publish",
    headers=headers,
)
print("publish:", r.status_code)
print(r.text)
r.raise_for_status()

print("Publicado com sucesso.")