""" Shows LIT for a trained AutoML text classification model """

""" IMPORTANT: You MUST run as a cell in a notebook (Vertex Workbench managed notebook).
You can only render HTML (LitWidget) in a browser or notebook and not in a Python console/editor environment.
It works in a browser, in Jupiter notebook, Jupyter Lab, etc."""

PROJECT_ID  = "argolis-rafaelsanchez-ml-dev"     # <---- CHANGE THIS
REGION      = "europe-west4"                     # <---- CHANGE THIS
ENDPOINT_ID = "5445749150979194880"              # <---- CHANGE THIS
DATA_BUCKET = "gs://argolis-vertex-europewest4"  # <---- CHANGE THIS
TEST_DATA   = "happyness-test.csv"
#LABEL_EVENT    = "1"
#LABEL_NONEVENT = "0"
VOCAB=["0", "1", "2", "3", "4", "5"]


from lit_nlp.api import dataset
from lit_nlp.api import model
from lit_nlp.api import types as lit_types
import requests
import json
import pandas as pd

from IPython.display import display, HTML

# Dataset class, loading from csv.
class TestData(dataset.Dataset):
  def __init__(self, path: str):
    with open(path) as fd:
      df = pd.read_csv(fd, header=0)
    self._examples = [{
        "text": row["text"],
        "label": str(row["label"]),
    } for _, row in df.iterrows()]

  def spec(self) -> lit_types.Spec:
    return {
        "text": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=VOCAB),
    }

# Get auth token for use in prediction requests
#token = !gcloud auth print-access-token
print('Update first the OAuth token at the code !!')
token = 'ya29[...]'               # <---- CHANGE THIS WITH OUTPUT FROM !gcloud auth print-access-token

# Setup URL and headers for prediction request.
url = f'https://{REGION}-aiplatform.googleapis.com/ui/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict'
headers = {'content-type': 'application/json', 'Authorization': f'Bearer {token[0]}'}

# Model class, using curl request to get online predictions.
class TestModel(model.Model):
  def input_spec(self) -> lit_types.Spec:
    return {
        "text": lit_types.TextSegment(),
        "label": lit_types.CategoryLabel(vocab=VOCAB, required=False),
    }
  def output_spec(self) -> lit_types.Spec:
    return {
        "preds": lit_types.MulticlassPreds(vocab=VOCAB, parent="label", null_idx=0),
    }
  def predict_minibatch(self, examples):
    # Online prediction can only handle one example at a time so run predictions in a loop.
    def get_pred(ex):
      # Escape quotes in text entries, to be able to send in payload.
      text = json.dumps(ex["text"])
      payload = '{"instances": {"mimeType": "text/plain","content": ' + text + ' }}'
      r = requests.post(url, data=payload, headers=headers)
      return r.json()['predictions'][0]['confidences']
    return [{"preds": get_pred(ex)} for ex in examples]


# Create th LIT widget with the model and dataset to analyze.
from lit_nlp import notebook

print('Loading dataset and model ...')
datasets = {'data': TestData('./happyness-test.csv')}
models = {'auto_nl': TestModel()}

print('Configuring Lit widget. You need a browser, like Jupyterlab or Colab.')
widget = notebook.LitWidget(models, datasets, height=800,  proxy_url="/proxy/%PORT%/")

# This needs a browser, with JuyterLab or Colab should work
display(HTML(widget.render()))