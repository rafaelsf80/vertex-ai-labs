from dask.distributed import Client, wait
from xgboost.dask import DaskDMatrix
from google.cloud import storage
import xgboost as xgb
import dask.dataframe as dd
import sys
import os
import subprocess
import time
import json

IRIS_DATA_FILENAME = 'gs://cloud-samples-data/ai-platform/iris/iris_data.csv'
IRIS_TARGET_FILENAME = 'gs://cloud-samples-data/ai-platform/iris/iris_target.csv'
MODEL_FILE = 'model.bst'
MODEL_DIR = os.getenv("AIP_MODEL_DIR")
XGB_PARAMS = {
    'verbosity': 2,
    'learning_rate': 0.1,
    'max_depth': 8,
    'objective': 'reg:squarederror',
    'subsample': 0.6,
    'gamma': 1,
    'verbose_eval': True,
    'tree_method': 'hist',
    'nthread': 1
}


def square(x):
    return x ** 2

def neg(x):
    return -x

def launch(cmd):
    """ launch dask workers
    """
    return subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)


def get_chief_ip(cluster_config_dict):
    if 'workerpool0' in cluster_config_dict['cluster']:
      ip_address = cluster_config_dict['cluster']['workerpool0'][0].split(":")[0]
    else:
      # if the job is not distributed, 'chief' will be populated instead of
      # workerpool0.
      ip_address = cluster_config_dict['cluster']['chief'][0].split(":")[0]

    print('The ip address of workerpool 0 is : {}'.format(ip_address))
    return ip_address

def get_chief_port(cluster_config_dict):

    if "open_ports" in cluster_config_dict:
      port = cluster_config_dict['open_ports'][0]
    else:
      # Use any port for the non-distributed job.
      port = 7777
    print("The open port is: {}".format(port))

    return port

if __name__ == '__main__':
    cluster_config_str = os.environ.get('CLUSTER_SPEC')
    cluster_config_dict  = json.loads(cluster_config_str)
    print(json.dumps(cluster_config_dict, indent=2))
    print('The workerpool type is:', flush=True)
    print(cluster_config_dict['task']['type'], flush=True)
    workerpool_type = cluster_config_dict['task']['type']
    chief_ip = get_chief_ip(cluster_config_dict)
    chief_port = get_chief_port(cluster_config_dict)
    chief_address = "{}:{}".format(chief_ip, chief_port)

    if workerpool_type == "workerpool0":
      print('Running the dask scheduler.', flush=True)
      proc_scheduler = launch('dask-scheduler --dashboard --dashboard-address 8888 --port {} &'.format(chief_port))
      print('Done the dask scheduler.', flush=True)

      client = Client(chief_address, timeout=1200)
      print('Waiting the scheduler to be connected.', flush=True)
      client.wait_for_workers(1)

      X = dd.read_csv(IRIS_DATA_FILENAME, header=None)
      y = dd.read_csv(IRIS_TARGET_FILENAME, header=None)
      X.persist()
      y.persist()
      wait(X)
      wait(y)
      dtrain = DaskDMatrix(client, X, y)

      output = xgb.dask.train(client, XGB_PARAMS, dtrain,  num_boost_round=100, evals=[(dtrain, 'train')])
      print("Output: {}".format(output), flush=True)
      print("Saving file to: {}".format(MODEL_FILE), flush=True)
      output['booster'].save_model(MODEL_FILE)
      bucket_name = MODEL_DIR.replace("gs://", "").split("/", 1)[0]
      folder = MODEL_DIR.replace("gs://", "").split("/", 1)[1]
      bucket = storage.Client().bucket(bucket_name)
      print("Uploading file to: {}/{}{}".format(bucket_name, folder, MODEL_FILE), flush=True)
      blob = bucket.blob('{}{}'.format(folder, MODEL_FILE))
      blob.upload_from_filename(MODEL_FILE)
      print("Saved file to: {}/{}".format(MODEL_DIR, MODEL_FILE), flush=True)

      # Waiting 10 mins to connect the Dask dashboard
      time.sleep(60 * 10)
      client.shutdown()

    else:
      print('Running the dask worker.', flush=True)
      client = Client(chief_address, timeout=1200)
      print('client: {}.'.format(client), flush=True)
      launch('dask-worker {}'.format(chief_address))
      print('Done with the dask worker.', flush=True)

      # Waiting 10 mins to connect the Dask dashboard
      time.sleep(60 * 10)