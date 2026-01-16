from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/3131050675115393024"  # replace this with your endpoint
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

FEATURES = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
            'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# The XGBoost container expects a list of lists (rows of values)
test_values = [test_instance[feature] for feature in FEATURES]
print(test_values)
response = endpoint.predict(instances=[test_values])

print('API response: ', response)