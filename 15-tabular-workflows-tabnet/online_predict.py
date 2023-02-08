from google.cloud import aiplatform

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/2357256172928499712"  # # <---- CHANGE THIS !!!
)

test_instance={
    'Time': "80422", # requires string
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

response = endpoint.predict([test_instance])

print('API response: ', response)

# TabNet output:
# API response:  Prediction(predictions=[{'feature_importance': {'V21': 0.5674177408218384, 'Amount': 0.0, 'V11': 4.632284164428711, 'V14': 1.625861644744873, 'V12': 1.003108978271484, 'V13': 0.0, 'Time': 0.0, 'V1': 0.8077748417854309, 'V10': 0.03072767704725266}, 
# 'scores': [0.9999723434448242, 2.766012585198041e-05], 
# 'classes': ['0', '1']}], deployed_model_id='2356992290137833472', model_version_id='1', model_resource_name='projects/989788194604/locations/europe-west4/models/3009081850246201344', explanations=None)