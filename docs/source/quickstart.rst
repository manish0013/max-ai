Quickstart
==========


maxairesources
==============

##### Data checks
- This module is dedicated to perform quick checks on data.
- `AnalysisBase` class is a base class in `analysis_base.py` file which should be inherited by all analysis class specific to use case / requirement.
- every analysis class will produce dictionary, which could be saved into disk using `save_analysis_report`
- types of columns are identified based on column names as per feast output, or you can pass dictionary in below format
- `SparkDataFrameAnalyser` class is used to analyse `pyspark` `dataframe` with numerical and categorical columns mainly.

  ```python
  from maxairesources.datachecks.dataframe_analysis_spark import SparkDataFrameAnalyser
  col_types = {
    "numerical_cols": [], # numerical columns
    "bool_cols": [],# boolean columns with True / False values
    "categorical_cols": [], # with categorical columns only, not text columns
    "free_text_cols": [], # text column (NLP) TODO 
    "unique_identifier_cols": [] # unique ID columns (primary key)
  }
  df = "<your pyspark dataframe>" # your data here
  analyser = SparkDataFrameAnalyser(df=df, column_types=col_types) # create instance of analyser
  report = analyser.generate_data_health_report() # generate report
  analyser.save_analysis_report(report) # saves in json file
  ``` 
- Note: while running `generate_data_health_report` method, report will be prepared and all data health calculations / checkup results will be printed in two logging levels
  1. logging.WARNING: shows that data is not aligned with given thresholds, appropriate transformation might required on dataset
  2. logging.INFO: shows information about specific checkup, no action required.

- we can also provide `threshold` parameter value for checkups in following format
  ```python
  thresholds = {
    "general": {"column_count": 5, "record_count": 10},
    "uni_variate": {
        "skewness_score": 1.0,
        "kurtosis_score": 3,
        "outlier_percentage": 1,
        "null_percentage": 1,
        "unique_categories_count": 10,
        "stddev": 3,
        "boolean_balance": 0.1,
    },
    "bi_variate": {"correlation_score": 0.5},
  }
  ```


tutorials
=========

maxaifeaturization
==================
maxaimarketplace
================

maxaimetadata
=============

maxaimodel
==========

maxairesources
==============
