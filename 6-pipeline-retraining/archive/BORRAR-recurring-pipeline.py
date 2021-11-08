from datetime import datetime
from kfp import dsl
from google_cloud_pipeline_components import aiplatform as vertex_pipeline_components
from google.cloud import aiplatform as vertex
from kfp.v2.dsl import component, Model, Input, Output, Metrics, Artifact, Dataset, OutputPath
from typing import NamedTuple, Dict, List, Sequence
from kfp.v2 import compiler

PROJECT_ID = "windy-site-254307"  # @param {type:"string"}
REGION = "us-central1"
PIPELINE_BUCKET_NAME = "vertex-model-governance-lab" # @param {type:"string"}
PIPELINE_JSON_PKG_PATH = "./rapid_prototyping.json"
PIPELINE_ROOT = f"gs://{PIPELINE_BUCKET_NAME}/pipeline_root"
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
DATA_FOLDER = f"{PIPELINE_BUCKET_NAME}/data"

TEST_DATASET_CSV_LOCATION = f"gs://{DATA_FOLDER}/test_dataset"
RAW_INPUT_DATA = f"gs://{DATA_FOLDER}/abalone.csv"
BQ_DATASET = 'prototyping_bqml_automl' # @param {type:"string"} #Dataset IDs must be alphanumeric (plus underscores) and must be at most 1024 characters long.
BQML_EXPORT_LOCATION = f"gs://{PIPELINE_BUCKET_NAME}/artifacts/bqml/"
BQML_SERVING_CONTAINER_IMAGE_URI = 'us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
GCS_BATCH_PREDICTION_OUTPUT_PREFIX = f"gs://{PIPELINE_BUCKET_NAME}/predictions/"

@component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery"])
def import_data_to_bigquery(project: str, 
                            bq_dataset: str, 
                            gcs_data_uri: str,
                            raw_dataset: Output[Artifact],
                            bq_location: str = 'us',
                            table_name_prefix: str = 'abalone'
                           ):
    from google.cloud import bigquery
    from collections import namedtuple
    # Construct a BigQuery client object.
    client = bigquery.Client(project=project, location=bq_location)

    def load_dataset(gcs_uri, table_id):
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("Sex", "STRING"),
                bigquery.SchemaField("Length", "NUMERIC"),
                bigquery.SchemaField("Diameter", "NUMERIC"),
                bigquery.SchemaField("Height", "NUMERIC"),
                bigquery.SchemaField("Whole_weight", "NUMERIC"),
                bigquery.SchemaField("Shucked_weight", "NUMERIC"),
                bigquery.SchemaField("Viscera_weight", "NUMERIC"),
                bigquery.SchemaField("Shell_weight", "NUMERIC"),
                bigquery.SchemaField("Rings", "NUMERIC"),
            ],
            skip_leading_rows=1,
            # The source format defaults to CSV, so the line below is optional.
            source_format=bigquery.SourceFormat.CSV,
        )
        print(f'Loading {gcs_uri} into {table_id}')
        load_job = client.load_table_from_uri(
            gcs_uri, table_id, job_config=job_config
        )  # Make an API request.

        load_job.result()  # Waits for the job to complete.
        destination_table = client.get_table(table_id)  # Make an API request.
        print("Loaded {} rows.".format(destination_table.num_rows))
        
    print('Checking for existence of bq dataset. If it does not exist, we will create one')
    bq_dataset_id = f"{project}.{bq_dataset}"
    client.create_dataset(bq_dataset_id, exists_ok=True, timeout=300)
    
    raw_table_name = f'{table_name_prefix}_raw'
    table_id = f"{project}.{bq_dataset}.{raw_table_name}"
    print('Deleting any tables that might have the same name on the dataset')
    client.delete_table(table_id, not_found_ok=True)
    print('will load data to table')
    load_dataset(gcs_data_uri, table_id)

    raw_dataset_uri = f"bq://{table_id}"
    raw_dataset.uri = raw_dataset_uri
    result_tuple = namedtuple('bqml_import', ['raw_dataset_uri'])
    #return result_tuple(raw_dataset_uri=str(table_id))




@component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow", "fsspec", "gcsfs"]) #pandas, pyarrow and fsspec required to export bq data to csv
def split_datasets(
        # project: str,
        # bq_dataset: str,
        # bq_raw_table: str,
        raw_dataset: Input[Artifact],
        #dataset: Output[Dataset],
        #train_dataset: Output[Dataset],
        #test_dataset: Output[Dataset],
        #validate_dataset: Output[Dataset],
        bqml_dataset: Output[Dataset],
        test_dataset_folder: str,
        bq_location: str = 'us'
) -> NamedTuple('bqml_split', [('dataset_uri', str), ('test_features_jsonl_filenames', list), ('test_labels_csv_filename', str), ('train_dataset_uri', str), ('test_dataset_uri', str), ('validate_dataset_uri', str)]):

    from google.cloud import bigquery
    from collections import namedtuple
    import gcsfs

    raw_dataset_uri = raw_dataset.uri
    table_name = raw_dataset_uri.split('bq://')[-1]
    print(table_name)
    raw_dataset_uri = table_name.split('.')
    print(raw_dataset_uri)
    project = raw_dataset_uri[0]
    bq_dataset = raw_dataset_uri[1]
    bq_raw_table = raw_dataset_uri[2]

    client = bigquery.Client(project=project, location=bq_location)

    def split_dataset(table_name_dataset):
        training_dataset_table_name = f'{project}.{bq_dataset}.{table_name_dataset}'
        split_query = f'''
        CREATE OR REPLACE TABLE
            `{training_dataset_table_name}`
           AS
        SELECT
          Sex,
          Length,
          Diameter,
          Height,
          Whole_weight,
          Shucked_weight,
          Viscera_weight,
          Shell_weight,
          Rings,
            CASE(ABS(MOD(FARM_FINGERPRINT(TO_JSON_STRING(f)), 10)))
              WHEN 9 THEN 'TEST'
              WHEN 8 THEN 'VALIDATE'
              ELSE 'TRAIN' END AS split_col
        FROM
          `{project}.{bq_dataset}.abalone_raw` f
        '''
        dataset_uri = f'{project}.{bq_dataset}.{bq_raw_table}'
        print('Splitting the dataset')
        query_job = client.query(split_query)  # Make an API request.  
        query_job.result()
        print(dataset_uri)
        print(split_query.replace("\n", " "))
        return training_dataset_table_name

    def create_separate_tables(splits, training_dataset_table_name):
        output = {}
        for s in splits:
            destination_table_name = f'abalone_{s}'
            query = f'''
             CREATE OR REPLACE TABLE `{project}.{bq_dataset}.{destination_table_name}` AS SELECT
          Sex,
          Length,
          Diameter,
          Height,
          Whole_weight,
          Shucked_weight,
          Viscera_weight,
          Shell_weight,
          Rings 
          FROM `{training_dataset_table_name}`  f
          WHERE 
          f.split_col = '{s}'
          '''
            print(f'Creating table for {s} --> {destination_table_name}')
            print(query.replace("\n", " "))
            output[s] = destination_table_name
            query_job = client.query(query)  # Make an API request.
            query_job.result()
            
        print(output)
        return output

    def create_bqml_dataset(training_dataset_table_name):
        bqml_dataset_table_name = 'dataset_bqml'

        query = f"""
        CREATE OR REPLACE TABLE
          `{project}.{bq_dataset}.{bqml_dataset_table_name}`  AS
        SELECT
          Sex,
          Length,
          Diameter,
          Height,
          Whole_weight,
          Shucked_weight,
          Viscera_weight,
          Shell_weight,
          Rings,
          CASE(split_col)
            WHEN 'VALIDATE' THEN 'EVAL'
            WHEN 'TRAIN' THEN 'TRAIN'
            WHEN 'TEST' THEN 'TEST'
        END
          AS split_col
        FROM
          `{project}.{bq_dataset}.{training_dataset_table_name}`  f
        WHERE
          split_col IN ('VALIDATE',
            'TRAIN')
        """

        #print(query)
        query_job = client.query(query)  # Make an API request.
        query_job.result()
        return bqml_dataset_table_name


    def export_test_features_to_gcs(bq_test_table_name, gcs_export_path_prefix):
        query_string = f"""
        SELECT
          Sex,
          Length,
          Diameter,
          Height,
          Whole_weight,
          Shucked_weight,
          Viscera_weight,
          Shell_weight,
          Rings
        FROM `{project}.{bq_dataset}.{bq_test_table_name}`  f
        """
        print(f'Exporting test dataset {project}.{bq_dataset}.{bq_test_table_name}')
        print(query_string.replace("\n", " "))
        dataframe = (
            client.query(query_string)
                .result()
                .to_dataframe(
                # Optionally, explicitly request to use the BigQuery Storage API. As of
                # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
                # API is used by default.
                create_bqstorage_client=True,
            )
        )

        test_labels_csv_filename = f'{gcs_export_path_prefix}/test_labels.csv'
        labels = dataframe['Rings']
        print(f'Exporting test labels into {test_labels_csv_filename}')
        labels.to_csv(test_labels_csv_filename, index=False, header=True)

        test_features_jsonl_filename = f'{gcs_export_path_prefix}/test_features.jsonl'
        features = dataframe.drop(columns=['Rings'])
        jsonl = features.to_json(orient='records', lines=True)
        gcs = gcsfs.GCSFileSystem()
        with gcs.open(test_features_jsonl_filename, "w") as text_file:
            text_file.write(jsonl)

        print(f'Exporting test labels into {test_features_jsonl_filename}')
        return test_features_jsonl_filename, test_labels_csv_filename

    table_name_dataset = 'dataset'

    dataset_uri = split_dataset(table_name_dataset)
    splits = ['TRAIN', 'VALIDATE', 'TEST']
    table_names_dict = create_separate_tables(splits, dataset_uri)
    bqml_dataset_table_name = create_bqml_dataset(table_name_dataset)
    test_table_name = table_names_dict['TEST']
    test_features_jsonl_filename, test_labels_csv_filename = export_test_features_to_gcs(test_table_name,
                                                                                         test_dataset_folder)
    dataset_uri = 'bq://' + dataset_uri
    train_dataset_uri = f"bq://{project}.{bq_dataset}.{table_names_dict['TRAIN']}"
    test_dataset_uri = f"bq://{project}.{bq_dataset}.{table_names_dict['TEST']}"
    validate_dataset_uri = f"bq://{project}.{bq_dataset}.{table_names_dict['VALIDATE']}"
    bqml_dataset_uri = f"bq://{project}.{bq_dataset}.{bqml_dataset_table_name}"
    
    print(f'dataset: {dataset_uri}')
    print(f'training: {train_dataset_uri}')
    print(f'test: {test_dataset_uri}')
    print(f'validation: {validate_dataset_uri}')
    
    #dataset.uri = dataset_uri
    #train_dataset.uri = train_dataset_uri
    #test_dataset.uri = test_dataset_uri
    #validate_dataset.uri = validate_dataset_uri
    bqml_dataset.uri = bqml_dataset_uri
    
    
    result_tuple = namedtuple('bqml_split', ['dataset_uri', 'test_features_jsonl_filenames', 'test_labels_csv_filename', 'train_dataset_uri', 'test_dataset_uri', 'validate_dataset_uri'])
    return result_tuple(dataset_uri=str(dataset_uri), test_features_jsonl_filenames=[test_features_jsonl_filename], test_labels_csv_filename=test_labels_csv_filename, train_dataset_uri=train_dataset_uri, test_dataset_uri=test_dataset_uri, validate_dataset_uri=validate_dataset_uri)



@component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery"])
def train_bqml_model(dataset: Input[Dataset], bqml_model: Output[Artifact], bq_location: str = 'us',
                     model_name: str = 'linear_regression_model', num_trials: int = 4) -> NamedTuple(
    'bqml_training',
    [('query', str)]):
    from google.cloud import bigquery
    from collections import namedtuple

    dataset_uri = dataset.uri
    table_name = dataset_uri.split('bq://')[-1]
    print(table_name)
    uri_parts = table_name.split('.')
    print(uri_parts)
    project = uri_parts[0]
    bq_dataset = uri_parts[1]
    training_data = uri_parts[2]
    
    
    client = bigquery.Client(project=project, location=bq_location)

    model_table_name = f"{project}.{bq_dataset}.{model_name}"

    model_options = """OPTIONS
      ( MODEL_TYPE='LINEAR_REG',
        LS_INIT_LEARN_RATE=0.15,
        L1_REG=1,
        MAX_ITERATIONS=5,
        DATA_SPLIT_COL='split_col',
        DATA_SPLIT_METHOD='CUSTOM',
        input_label_cols=['Rings']
        
    """
    if num_trials > 0:
        model_options += f""", 
        
        NUM_TRIALS={num_trials},
        HPARAM_TUNING_OBJECTIVES=['mean_squared_error']
        """

    model_options += ")"""

    query = f"""
    CREATE OR REPLACE MODEL
      `{model_table_name}`
      {model_options}
     AS
    SELECT
      Sex,
      Length,
      Diameter,
      Height,
      Whole_weight,
      Shucked_weight,
      Viscera_weight,
      Shell_weight,
      Rings,
      split_col
    FROM
      `{table_name}`;
    """

    print(query.replace("\n", " "))
    query_job = client.query(query)  # Make an API request.
    print(query_job.job_id)
    query_job.result()
    bqml_model.uri = f'bq://{model_table_name}'

    result_tuple = namedtuple('bqml_training', ['query'])

    return result_tuple(query=str(query))



@component(base_image="python:3.9", packages_to_install=["google-cloud-bigquery"])
def export_bqml_to_tf(project: str, export_location: str, bqml_model: Input[Model], tf_model: Output[Artifact],
                      bq_location: str = 'us'):
    from google.cloud import bigquery
    bqml_table_name = bqml_model.uri.split('/')[-1]
    query = f"""
     EXPORT MODEL `{bqml_table_name}`
    OPTIONS(URI = '{export_location}')

    """
    client = bigquery.Client(project=project, location=bq_location)
    query_job = client.query(query)
    query_job.result()

    tf_model.uri = export_location



@component(
    base_image="python:3.9",
    packages_to_install=["google-cloud-aiplatform==1.3.0", "numpy==1.21.1",
                         "pandas==1.3.0", "scikit-learn==0.24.2", "pyarrow==5.0.0", "fsspec==2021.9.0", "gcsfs==2021.9.0"],
)
def evaluate_batch_predictions(batch_prediction_job: Input[Artifact],
                               gcs_ground_truth: str,
                               model_framework: str,
                               model_type: str,
                               metrics: Output[Metrics],
                               reference_metric_name: str = 'rmse',
                               ) -> NamedTuple('ModelEvaluationOutput',
                                               [
                                                   ('metric', float)
                                               ]):
    from google.cloud import aiplatform
    from collections import namedtuple
    import gcsfs
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error
    import json

    def treat_job_uri(uri):
        return uri[uri.find('projects/'):]

    def get_job_output_dir(batch_prediction_uri):
        bpj = aiplatform.BatchPredictionJob(batch_prediction_uri)
        output = bpj.output_info.gcs_output_directory
        return output

    def create_dicts_from_predictions(local_files):
        instances_output = []
        predictions_output = []
        for x in local_files:
            with gcs.open(x) as f:
                for line in f:
                    line_json = json.loads(line)
                    #print(json.dumps(line_json, indent=2))
                    instance = line_json['instance']
                    instances_output.append(instance)
                    pred = line_json['prediction']
                    if type(pred) is dict and 'value' in pred.keys():
                        # AutoML predictions
                        prediction = pred['value']
                    elif type(pred) is list:
                        # BQML Predictions return different format
                        prediction = pred[0]

                    predictions_output.append(prediction)
        return instances_output, predictions_output

    def evaluate(predictions, test_results):
        return dict(rmse=np.sqrt(mean_squared_error(test_results, predictions)))

    def get_prediction_file_names(gcs, gcs_dir):
        reg_expression = f'{gcs_dir}/prediction.results*'
        file_names = [f'gs://{x}' for x in gcs.glob(reg_expression)]
        return file_names

    bpj_uri = batch_prediction_job.uri
    print(f'Batch Prediction Job URI: {bpj_uri}')

    treated_uri = treat_job_uri(bpj_uri)
    gcs_dir = get_job_output_dir(treated_uri)
    print(f'Results saved to {gcs_dir}')
    gcs = gcsfs.GCSFileSystem()
    prediction_files = get_prediction_file_names(gcs, gcs_dir)
    print(f'Predictions available on following files: {prediction_files}')

    instances, predictions = create_dicts_from_predictions(prediction_files)
    print(f'{len(predictions)} predictions found')

    labels = pd.read_csv(gcs_ground_truth).to_numpy()
    print(f'{len(labels)} ground truth labels found')

    evaluation_metric = evaluate(predictions, labels)
    print(evaluation_metric)

    print('Logging metrics to output artifact')
    metrics.metadata["model_type"] = model_type
    metrics.metadata["framework"] = model_framework
    metrics.metadata["SAMPLE_KEY"] = "You can add other metrics here"

    for k, v in evaluation_metric.items():
        print(f'{k} -> {v}')
        metrics.log_metric(k, v)
    ModelEvaluationOutput = namedtuple('ModelEvaluationOutput', ['metric'])

    return ModelEvaluationOutput(metric=float(evaluation_metric[reference_metric_name]))




@component(base_image="python:3.9")
def select_best_model(model_one: Input[Model], metrics_one: float, model_two: Input[Model],
                      metrics_two: float, thresholds_dict_str: str, best_model: Output[Model], reference_metric_name: str = 'rmse') -> NamedTuple("Outputs",
                                                                                                              [(
                                                                                                               "deploy_decision",
                                                                                                               str), (
                                                                                                               "metric",
                                                                                                               float), (
                                                                                                               "metric_name",
                                                                                                               str)]):
    import json
    from collections import namedtuple

    message = None
    best_metric = float('inf')
    if metrics_one <= metrics_two:
        best_model.uri = model_one.uri
        best_metric = metrics_one
        message = 'Model one is the best'
    else:
        best_model.uri = model_two.uri
        best_metric = metrics_two
        message = 'Model two is the best'
    thresholds_dict = json.loads(thresholds_dict_str)
    deploy = False
    if best_metric < thresholds_dict[reference_metric_name]:
        deploy = True
    if deploy:
        deploy_decision = "true"
    else:
        deploy_decision = "false"

    print(f'Which model is best? {message}')
    print(f'What metric is being used? {reference_metric_name}')
    print(f'What is the best metric? {best_metric}')
    print(f'What is the threshold to deploy? {thresholds_dict_str}')
    print(f'Deploy decision: {deploy_decision}')

    Outputs = namedtuple('Outputs', ['deploy_decision', 'metric', 'metric_name'])

    return Outputs(
        deploy_decision=deploy_decision,
        metric=best_metric,
        metric_name=reference_metric_name
    )




@component(base_image="python:3.9", packages_to_install=["google-cloud-aiplatform"])
def validate_infra(endpoint: Input[Artifact],
                  ) -> NamedTuple(
    'validate_infrastructure_output',
    [('instance', str), ('prediction', float)]):
    from google.cloud import aiplatform
    from google.protobuf import json_format
    from google.protobuf.struct_pb2 import Value
    from collections import namedtuple
    import json

    def treat_uri(uri):
        return uri[uri.find('projects/'):]

    def request_prediction(endp, instance):
        instance = json_format.ParseDict(instance, Value())
        instances = [instance]
        parameters_dict = {}
        parameters = json_format.ParseDict(parameters_dict, Value())
        response = endp.predict(instances=instances, parameters=parameters)
        print("deployed_model_id:", response.deployed_model_id)
        print('predictions: ', response.predictions)
        # The predictions are a google.protobuf.Value representation of the model's predictions.
        predictions = response.predictions

        for pred in predictions:
            if type(pred) is dict and 'value' in pred.keys():
                # AutoML predictions
                prediction = pred['value']
            elif type(pred) is list:
                # BQML Predictions return different format
                prediction = pred[0]
            return prediction
    
    endpoint_uri = endpoint.uri
    treated_uri = treat_uri(endpoint_uri)

    instance = {
        "Sex": "M",
        "Length": 0.33,
        "Diameter": 0.255,
        "Height": 0.08,
        "Whole_weight": 0.205,
        "Shucked_weight": 0.0895,
        "Viscera_weight": 0.0395,
        "Shell_weight": 0.055
    }
    instance_json = json.dumps(instance)
    print('Will use the following instance: ' + instance_json)

    endpoint = aiplatform.Endpoint(treated_uri)
    prediction = request_prediction(endpoint, instance)
    result_tuple = namedtuple('validate_infrastructure_output', ['instance', 'prediction'])

    return result_tuple(instance=str(instance_json), prediction=float(prediction))





pipeline_params = {
    'project': PROJECT_ID,
    'region': REGION,
    'gcs_input_file_uri': RAW_INPUT_DATA,
    'bq_dataset': BQ_DATASET,
    'bqml_model_export_location': BQML_EXPORT_LOCATION,
    'bqml_serving_container_image_uri': BQML_SERVING_CONTAINER_IMAGE_URI,
    'test_dataset_folder': TEST_DATASET_CSV_LOCATION,
    'gcs_batch_prediction_output_prefix': GCS_BATCH_PREDICTION_OUTPUT_PREFIX,
    'thresholds_dict_str': '{"rmse": 20.0}',

}


@dsl.pipeline(
    name='rapid-prototyping-bqml-vs-automl',
    description='Rapid Prototyping'
)
def train_pipeline(project: str, 
                   gcs_input_file_uri: str, 
                   region: str, 
                   bq_dataset: str, 
                   bqml_model_export_location: str,
                   bqml_serving_container_image_uri: str, 
                   test_dataset_folder: str,
                   gcs_batch_prediction_output_prefix: str,
                   thresholds_dict_str:str):
    import_data_to_bigquery_op = import_data_to_bigquery(project,
                                                         bq_dataset,
                                                         gcs_input_file_uri)

    model_display_name = 'rapid_prototyping_model'
    job_display_name = f'{model_display_name}_job'

    raw_dataset = import_data_to_bigquery_op.outputs['raw_dataset']
    
    split_datasets_op = split_datasets(raw_dataset, test_dataset_folder=test_dataset_folder)
    
    bqml_dataset = split_datasets_op.outputs['bqml_dataset']
    train_bqml_model_op = train_bqml_model(bqml_dataset)
    bqml_trained_model = train_bqml_model_op.outputs['bqml_model']
    export_bqml_to_tf_op = export_bqml_to_tf(export_location=bqml_model_export_location, project=project,
                                             bqml_model=bqml_trained_model)
    
    
    batch_prediction_input = split_datasets_op.outputs['test_features_jsonl_filenames']
    ground_truth = split_datasets_op.outputs['test_labels_csv_filename'] 

    bqml_model_upload_op = vertex_pipeline_components.ModelUploadOp(
        project=project,
        display_name=model_display_name + '_bqml',
        artifact_uri=bqml_model_export_location,
        serving_container_image_uri=bqml_serving_container_image_uri,
    )
    bqml_model_upload_op.after(export_bqml_to_tf_op)
    bqml_model = bqml_model_upload_op.outputs['model']

    bqml_model_batch_prediction_task = vertex_pipeline_components.ModelBatchPredictOp(
        project=project,
        model=bqml_model,
        job_display_name=job_display_name + '_bqml',
        gcs_source=batch_prediction_input,
        gcs_destination_prefix=gcs_batch_prediction_output_prefix,
        predictions_format='jsonl',
        instances_format='jsonl',
        machine_type="n1-standard-2",
    )
    
    dataset_create_op = vertex_pipeline_components.TabularDatasetCreateOp(
        project=project, display_name="abalone-pipeline", bq_source=split_datasets_op.outputs['dataset_uri']
    )

    automl_training_op = vertex_pipeline_components.AutoMLTabularTrainingJobRunOp(
        project=project,
        display_name=f"{model_display_name}_automl",
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse",
        predefined_split_column_name='split_col',
        dataset=dataset_create_op.outputs["dataset"],
        target_column="Rings",
        column_transformations=[
            {"categorical": {"column_name": "Sex"}},
            {"numeric": {"column_name": "Length"}},
            {"numeric": {"column_name": "Diameter"}},
            {"numeric": {"column_name": "Height"}},
            {"numeric": {"column_name": "Whole_weight"}},
            {"numeric": {"column_name": "Shucked_weight"}},
            {"numeric": {"column_name": "Viscera_weight"}},
            {"numeric": {"column_name": "Shell_weight"}},
            {"numeric": {"column_name": "Rings"}},

        ],
    )
    automl_model = automl_training_op.outputs['model']
    automl_model_batch_prediction_task = vertex_pipeline_components.ModelBatchPredictOp(
        project=project,
        model=automl_model,
        job_display_name=job_display_name + '_automl',
        gcs_source=batch_prediction_input,
        gcs_destination_prefix=gcs_batch_prediction_output_prefix,
        predictions_format='jsonl',
        instances_format='jsonl',
        machine_type="n1-standard-2",
    )
    
    
    automl_model_evaluation_task = evaluate_batch_predictions(
        batch_prediction_job=automl_model_batch_prediction_task.outputs["batchpredictionjob"],
        gcs_ground_truth=ground_truth,
        model_framework="AutoML",
        model_type="Regression")

    bqml_model_evaluation_task = evaluate_batch_predictions(
        batch_prediction_job=bqml_model_batch_prediction_task.outputs["batchpredictionjob"],
        gcs_ground_truth=ground_truth,
        model_framework="BQML",
        model_type="Regression")

    automl_model_metric = automl_model_evaluation_task.outputs['metric']
    bqml_model_metric = bqml_model_evaluation_task.outputs['metric']
    
    
    best_model_task = select_best_model(
        model_one=automl_model,
        metrics_one=automl_model_metric,  # ,
        model_two=bqml_model,
        metrics_two=bqml_model_metric,  # automl_model_evaluation_task.outputs['metric'],
        thresholds_dict_str=thresholds_dict_str,
    )

    with dsl.Condition(
            best_model_task.outputs["deploy_decision"] == "true",
            name="deploy_decision",
    ):
        endpoint_create_op = vertex_pipeline_components.EndpointCreateOp(
            project=project,
            display_name=f"pipelines-created-endpoint{TIMESTAMP}",

        )

        endpoint_create_op.after(best_model_task)

        model_deploy_op = vertex_pipeline_components.ModelDeployOp(  # noqa: F841
            project=project,
            endpoint=endpoint_create_op.outputs["endpoint"],
            model=best_model_task.outputs['best_model'],
            deployed_model_display_name=model_display_name + '_best',
            machine_type="n1-standard-2",
            #traffic_percentage=100
        ).set_caching_options(False)

        
        validate_infra_task = validate_infra(
            endpoint=model_deploy_op.outputs["endpoint"]
        ).set_caching_options(False)
        
        validate_infra_task.after(model_deploy_op)





compiler.Compiler().compile(
    pipeline_func=train_pipeline,
    package_path=PIPELINE_JSON_PKG_PATH,
)

vertex.init(project=PROJECT_ID, location=REGION)

pipeline_job = vertex.PipelineJob(
    display_name="BQML vs AutoML",
    template_path=PIPELINE_JSON_PKG_PATH,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=pipeline_params,
    enable_caching=True
)

response = pipeline_job.run()

print(response)



