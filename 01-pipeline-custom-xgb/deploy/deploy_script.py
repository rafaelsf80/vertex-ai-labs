import argparse
import json
import logging

from kfp.v2.components import executor
from kfp.v2.dsl import Input, Output, Model, Artifact

_logger = logging.getLogger(__name__)

def main(project_id: str,
            location: str,
            serving_container_image: str,
            model: Input[Model],
            endpoint: Output[Artifact]
):  
    from google.cloud import aiplatform

    aiplatform.init(project=project_id, location=location)
    
    logging.info(f'Model uri {model.uri}')


    # Upload model
    uploaded_model = aiplatform.Model.upload(
        location = location,
        display_name = 'kaggle_fraud_detection',
        artifact_uri = model.uri[0:-10],  # only directory, remove 'model.bst'
        serving_container_image_uri=serving_container_image
    )

    logging.info(f'uploaded model {uploaded_model.resource_name}')

    # Deploy
    endpoint = uploaded_model.deploy(
        machine_type='n1-standard-4'
    )

    logging.info(f'endpoint name {endpoint.resource_name}')

    endpoint.uri = endpoint.resource_name

    
    
def executor_main():
    """Main executor."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--executor_input', type=str)
    parser.add_argument('--function_to_execute', type=str)

    args, _ = parser.parse_known_args()
    print(args)
    executor_input = json.loads(args.executor_input)
    function_to_execute = globals()[args.function_to_execute]

    executor.Executor(
      executor_input=executor_input,
      function_to_execute=function_to_execute).execute()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    executor_main()
