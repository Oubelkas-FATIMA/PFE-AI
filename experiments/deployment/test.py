import base64
import os
import json


image_path = "PATH_TO_IMAGE_TO_BE_PREDICTED"
with open(image_path, 'rb') as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')

# Create the JSON payload
json_payload = f'{{"image": "data:image/jpeg;base64,{base64_image}"}}'

with open('payload.json', 'w') as payload_file:
    payload_file.write(json_payload)

curl_command = 'curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @payload.json'
output = os.popen(curl_command).read()

try:
    results = json.loads(output)
except json.JSONDecodeError:
    print("Error decoding JSON response")
    results = {}


with open('results2.json', 'w') as results_file:
    json.dump(results, results_file, indent=4)

print("Results saved to ")
