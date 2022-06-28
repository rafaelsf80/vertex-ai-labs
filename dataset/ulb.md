# ULB dataset

**Description**: This dataset contains anonymized credit card transactions made over 2 days in September 2013 by European cardholders, with 492 frauds out of 284,807 transactions (highly unbalanced). The dataset has been collected by ULB (Université Libre de Bruxelles).
More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the [DefeatFraud project](https://mlg.ulb.ac.be/wordpress/portfolio_page/defeatfraud-assessment-and-validation-of-deep-feature-engineering-and-learning-solutions-for-fraud-detection/).   
The dataset is also available in BigQuery as public data, at table `bigquery-public-data:ml_datasets.ulb_fraud_detection`, and in [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

**Details**:
```bash
bq show --format=pretty bigquery-public-data:ml_datasets.ulb_fraud_detection | awk -F'|' '{print "total_rows",$5}' | sed '6!d' && bq show --format=pretty bigquery-public-data:ml_datasets.ulb_fraud_detection | awk -F'|' '{print "total_bytes",$6 = $6/1024/1024 " MB"}' | sed '6!d'
total_rows  284807     
total_bytes 67.3601 MB
```

```bash
bq show --format=prettyjson bigquery-public-data:ml_datasets.ulb_fraud_detection  | jq -r '.schema.fields[]' | jq -r '[.name, .type, .description] | @tsv'
Time    FLOAT
V1      FLOAT
V2      FLOAT
V3      FLOAT
V4      FLOAT
V5      FLOAT
V6      FLOAT
V7      FLOAT
V8      FLOAT
V9      FLOAT
V10     FLOAT
V11     FLOAT
V12     FLOAT
V13     FLOAT
V14     FLOAT
V15     FLOAT
V16     FLOAT
V17     FLOAT
V18     FLOAT
V19     FLOAT
V20     FLOAT
V21     FLOAT
V22     FLOAT
V23     FLOAT
V24     FLOAT
V25     FLOAT
V26     FLOAT
V27     FLOAT
V28     FLOAT
Amount  FLOAT
```

**Preview**: Go to `bigquery-public-data project`, then select `ml-datasets`, and then select the `ulb-fraud-detection` table within it.

**Exploratory Data Analysis**: