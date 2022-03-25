from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/windy-site-254307/locations/us-central1/endpoints/1225876300133040128"
)

test_instance={
    'Time': 80422,
    'Amount': 17.99,
    'V1': -0.24,
    'V2': -0.027,
    'V3': 0.064,
    'V4': -0.16,
    'V5': -0.152,
    'V6': -0.3,
    'V7': -0.03,
    'V8': -0.01,
    'V9': -0.13,
    'V10': -0.18,
    'V11': -0.16,
    'V12': 0.06,
    'V13': -0.11,
    'V14': 2.1,
    'V15': -0.07,
    'V16': -0.033,
    'V17': -0.14,
    'V18': -0.08,
    'V19': -0.062,
    'V20': -0.08,
    'V21': -0.06,
    'V22': -0.088,
    'V23': -0.03,
    'V24': 0.01,
    'V25': -0.04,
    'V26': -0.99,
    'V27': -0.13,
    'V28': 0.003
}

response = endpoint.predict([test_instance])

print('API response: ', response)