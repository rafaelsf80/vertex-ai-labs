""" This script will trigger alerts in an existing Vertex Model monitoring job """

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import time

SAMPLE_CSV_FILE = './000000000000.csv'

#### MODIFY ENDPOINT_ID
ENDPOINT_ID = 'projects/655797269815/locations/us-central1/endpoints/8576410598978355200' 

endpoint = aiplatform.Endpoint(ENDPOINT_ID)

sample_df = pd.read_csv(SAMPLE_CSV_FILE)
sample_df.head()

target = sample_df['churned']
features = sample_df.drop(['churned'], axis=1).select_dtypes(include=['int64'])

normalizer = keras.layers.experimental.preprocessing.Normalization()
normalizer.adapt(features)

clean_sample_df = sample_df.dropna()
target = clean_sample_df['churned']
features = clean_sample_df.drop(['churned', 'timestamp'], axis=1)
numeric_features = features.select_dtypes(include=['int64'])
categorical_features = features.drop(['entity_type_customer', 'user_pseudo_id'], axis=1).select_dtypes(include=['object']).astype(str)

dataset = tf.data.Dataset.from_tensor_slices(({**dict(numeric_features), **dict(categorical_features)}, target))

def convert_example_to_protobuf(example):
    clean_dtypes = {key: val.tolist()*4 if hasattr(val, 'dtype') else val.decode("utf-8") for key, val in example.items()}
    return json_format.ParseDict(clean_dtypes, Value())

 
test_time = 300
tests_per_sec = 1
sleep_time = 1 / tests_per_sec
iterations = test_time * tests_per_sec

for example, truth in dataset.skip(100).take(500).as_numpy_iterator():
    import random
    
    if random.choice([True, False]):
        #produce an irregular input
        example['cnt_post_score'] = example['cnt_post_score'] + 100000
        example['cnt_level_start_quickplay'] = example['cnt_level_start_quickplay'] ** 4
        example['cnt_ad_reward'] = example['cnt_ad_reward'] - 2000
        example['cnt_spend_virtual_currency'] = example['cnt_spend_virtual_currency'] + 2000
        example['cnt_post_score'] = example['cnt_post_score'] - 1000
        example['cnt_completed_5_levels'] = example['cnt_completed_5_levels'] + 19000
        example['cnt_use_extra_steps'] = example['cnt_completed_5_levels'] - 100000
        example['country'] = example['language']

    #convert the example to protobuf message
    instances = [convert_example_to_protobuf(example)]
    print(f'instance: {example},\npredicted: {endpoint.predict(instances).predictions[0][0]}, truth: {truth}')
    time.sleep(sleep_time)
