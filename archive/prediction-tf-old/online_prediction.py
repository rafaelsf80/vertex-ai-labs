""" Online prediction for the ULB dataset """

from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/4153708639133106176"  # <---- CHANGE THIS
)

test_instance={
    'Time':80422,
    'Amount':17.99,
    'V1':-0.24,
    'V2':-0.027,
    'V3':0.064,
    'V4':-0.16,
    'V5':-0.152,
    'V6':-0.3,
    'V7':-0.03,
    'V8':-0.01,
    'V9':-0.13,
    'V10':-0.18,
    'V11':-0.16,
    'V12':0.06,
    'V13':-0.11,
    'V14':2.1,
    'V15':-0.07,
    'V16':-0.033,
    'V17':-0.14,
    'V18':-0.08,
    'V19':-0.062,
    'V20':-0.08,
    'V21':-0.06,
    'V22':-0.088,
    'V23':-0.03,
    'V24':0.01,
    'V25':-0.04,
    'V26':-0.99,
    'V27':-0.13,
    'V28':0.003,
}

response = endpoint.predict([test_instance, test_instance])

print('API response: ', response)

print('Value predicted for the first sample: ', response.predictions[0][0])
