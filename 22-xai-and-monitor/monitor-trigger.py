""" Step 3: Trigger alerts in the Model monitoring job """

import random
import time
import numpy as np

from google.cloud.aiplatform_v1beta1.services.prediction_service import \
    PredictionServiceClient
from google.cloud.aiplatform_v1beta1.types.prediction_service import \
    PredictRequest
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


PROJECT_ID = "argolis-rafaelsanchez-ml-dev"    # <---- CHANGE THIS
REGION = "europe-west4"                        # <---- CHANGE THIS
SUFFIX = "aiplatform.googleapis.com"
ENDPOINT_ID = 'projects/989788194604/locations/europe-west4/endpoints/97091274779131904'    # <---- CHANGE THIS
API_ENDPOINT = f"{REGION}-{SUFFIX}"
PREDICT_API_ENDPOINT = f"{REGION}-prediction-{SUFFIX}"


def random_uid():
    digits = [str(i) for i in range(10)] + ["A", "B", "C", "D", "E", "F"]
    return "".join(random.choices(digits, k=32))


# Sampling distributions for categorical features...
DAYOFWEEK = {1: 1040, 2: 1223, 3: 1352, 4: 1217, 5: 1078, 6: 1011, 7: 1110}
LANGUAGE = {
    "en-us": 4807,
    "en-gb": 678,
    "ja-jp": 419,
    "en-au": 310,
    "en-ca": 299,
    "de-de": 147,
    "en-in": 130,
    "en": 127,
    "fr-fr": 94,
    "pt-br": 81,
    "es-us": 65,
    "zh-tw": 64,
    "zh-hans-cn": 55,
    "es-mx": 53,
    "nl-nl": 37,
    "fr-ca": 34,
    "en-za": 29,
    "vi-vn": 29,
    "en-nz": 29,
    "es-es": 25,
}
OS = {"IOS": 3980, "ANDROID": 3798, "null": 253}
MONTH = {6: 3125, 7: 1838, 8: 1276, 9: 1718, 10: 74}
COUNTRY = {
    "United States": 4395,
    "India": 486,
    "Japan": 450,
    "Canada": 354,
    "Australia": 327,
    "United Kingdom": 303,
    "Germany": 144,
    "Mexico": 102,
    "France": 97,
    "Brazil": 93,
    "Taiwan": 72,
    "China": 65,
    "Saudi Arabia": 49,
    "Pakistan": 48,
    "Egypt": 46,
    "Netherlands": 45,
    "Vietnam": 42,
    "Philippines": 39,
    "South Africa": 38,
}

# Means and standard deviations for numerical features...
MEAN_SD = {
    "julianday": (204.6, 34.7),
    "cnt_user_engagement": (30.8, 53.2),
    "cnt_level_start_quickplay": (7.8, 28.9),
    "cnt_level_end_quickplay": (5.0, 16.4),
    "cnt_level_complete_quickplay": (2.1, 9.9),
    "cnt_level_reset_quickplay": (2.0, 19.6),
    "cnt_post_score": (4.9, 13.8),
    "cnt_spend_virtual_currency": (0.4, 1.8),
    "cnt_ad_reward": (0.1, 0.6),
    "cnt_challenge_a_friend": (0.0, 0.3),
    "cnt_completed_5_levels": (0.1, 0.4),
    "cnt_use_extra_steps": (0.4, 1.7),
}

DEFAULT_INPUT = {
    "cnt_ad_reward": 0,
    "cnt_challenge_a_friend": 0,
    "cnt_completed_5_levels": 1,
    "cnt_level_complete_quickplay": 3,
    "cnt_level_end_quickplay": 5,
    "cnt_level_reset_quickplay": 2,
    "cnt_level_start_quickplay": 6,
    "cnt_post_score": 34,
    "cnt_spend_virtual_currency": 0,
    "cnt_use_extra_steps": 0,
    "cnt_user_engagement": 120,
    "country": "Denmark",
    "dayofweek": 3,
    "julianday": 254,
    "language": "da-dk",
    "month": 9,
    "operating_system": "IOS",
    "user_pseudo_id": "104B0770BAE16E8B53DF330C95881893",
}

def send_predict_request(endpoint, input):
    client_options = {"api_endpoint": PREDICT_API_ENDPOINT}
    client = PredictionServiceClient(client_options=client_options)
    params = {}
    params = json_format.ParseDict(params, Value())
    request = PredictRequest(endpoint=endpoint, parameters=params)
    inputs = [json_format.ParseDict(input, Value())]
    request.instances.extend(inputs)
    response = client.predict(request)
    return response

def monitoring_test(count, sleep, perturb_num={}, perturb_cat={}):
    # Use random sampling and mean/sd with gaussian distribution to model
    # training data. Then modify sampling distros for two categorical features
    # and mean/sd for two numerical features.
    mean_sd = MEAN_SD.copy()
    country = COUNTRY.copy()
    for k, (mean_fn, sd_fn) in perturb_num.items():
        orig_mean, orig_sd = MEAN_SD[k]
        mean_sd[k] = (mean_fn(orig_mean), sd_fn(orig_sd))
    for k, v in perturb_cat.items():
        country[k] = v
    for i in range(0, count):
        input = DEFAULT_INPUT.copy()
        input["user_pseudo_id"] = str(random_uid())
        input["country"] = random.choices([*country], list(country.values()))[0]
        input["dayofweek"] = random.choices([*DAYOFWEEK], list(DAYOFWEEK.values()))[0]
        input["language"] = str(random.choices([*LANGUAGE], list(LANGUAGE.values()))[0])
        input["operating_system"] = str(random.choices([*OS], list(OS.values()))[0])
        input["month"] = random.choices([*MONTH], list(MONTH.values()))[0]
        for key, (mean, sd) in mean_sd.items():
            sample_val = round(float(np.random.normal(mean, sd, 1)))
            val = max(sample_val, 0)
            input[key] = val
        print(f"Sending prediction {i}")
        try:
            send_predict_request(ENDPOINT_ID, input)
        except Exception:
            print("prediction request failed")
        time.sleep(sleep)
    print("Test Completed.")


test_time = 300
tests_per_sec = 1
sleep_time = 1 / tests_per_sec
iterations = test_time * tests_per_sec
perturb_num = {"cnt_user_engagement": (lambda x: x * 3, lambda x: x / 3)}
perturb_cat = {"Japan": max(COUNTRY.values()) * 2}
monitoring_test(iterations, sleep_time, perturb_num, perturb_cat)