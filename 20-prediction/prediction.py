from google.cloud import aiplatform
import tensorflow as tf

endpoint = aiplatform.Endpoint(
    endpoint_name="projects/989788194604/locations/europe-west4/endpoints/5121560346552696832"  # replace this with your endpoint
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


#test_instance = {"Time": [80444,30002], "Amount": [17.99 20.99], "V1": [-0.24,-0.24], "V2": [-0.027,-0.027], "V3": [0.064,0.064], "V4": [-0.16,-0.16], "V5": [-0.152,-0.152], "V6": [-0.3,-0.3], "V7": [-0.03,-0.03], "V8": [-0.01,-0.01], "V9": [-0.13,-0.13], "V10": [-0.18,-0.18], "V11": [-0.16,-0.18], "V12": [0.06,0.18], "V13": [-0.11, -0.18], "V14": [2.1, 2.1], "V15": [-0.07,-0.18], "V16": [-0.033,-0.018], "V17": [-0.14,-0.18], "V18": [-0.08,-0.18], "V19": [-0.062,-0.18], "V20": [-0.08,-0.18], "V21": [-0.06,-0.18], "V22": [-0.088,-0.018], "V23": [-0.03,-0.18], "V24": [0.01,0.18], "V25": [-0.04,-0.18], "V26": [-0.99,-0.18], "V27": [-0.13,-0.18], "V28": [0.003,0.018]}
#test_instance = {"Time": [[80422], [80622]], "Amount": [[17.99], [20.99]], "V1": [[17.99], [20.99]], "V2": [[-0.027],[-0.027]], "V3": [[-0.027],[-0.027]], "V4": [[-0.027],[-0.027]], "V5": [[-0.027],[-0.027]], "V6": [[-0.027],[-0.027]], "V7": [[-0.027],[-0.027]], "V8": [[-0.027],[-0.027]], "V9": [[-0.027],[-0.027]], "V10": [[-0.027],[-0.027]], "V11": [[-0.027],[-0.027]], "V12": [[-0.027],[-0.027]], "V13": [[-0.027],[-0.027]], "V14": [[-0.027],[-0.027]], "V15": [[-0.027],[-0.027]], "V16": [[-0.027],[-0.027]], "V17": [[-0.027],[-0.027]], "V18": [[-0.027],[-0.027]], "V19": [[-0.027],[-0.027]], "V20": [[-0.027],[-0.027]], "V21": [[-0.027],[-0.027]], "V22": [[-0.027],[-0.027]], "V23": [[-0.027],[-0.027]], "V24": [[-0.027],[-0.027]], "V25": [[-0.027],[-0.027]], "V26": [[-0.027],[-0.027]], "V27": [[-0.027],[-0.027]], "V28": [[-0.027],[-0.027]]}

#input_dict = {name: tf.convert_to_tensor([value]) for name, value in test_instance.items()}
#print(input_dict)
#response = endpoint.predict(input_dict)


response = endpoint.predict([test_instance])

print('API response: ', response)