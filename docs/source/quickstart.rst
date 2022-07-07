Quickstart
==========
To use maxai, first install it using pip:

::

  (.venv) $ pip install maxai-1.0.0-py3-none-any.whl


maxairesources
==============

datachecks
___________
- This module is dedicated to perform quick checks on data.
- `AnalysisBase` class is a base class in `analysis_base.py` file which should be inherited by all analysis class specific to use case / requirement.
- every analysis class will produce dictionary, which could be saved into disk using `save_analysis_report`
- types of columns are identified based on column names as per feast output, or you can pass dictionary in below format
- `SparkDataFrameAnalyser` class is used to analyse `pyspark` `dataframe` with numerical and categorical columns mainly.

::
  
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
Note: while running `generate_data_health_report` method, report will be prepared and all data health calculations / checkup results will be printed in two logging levels
  1. logging.WARNING: shows that data is not aligned with given thresholds, appropriate transformation might required on dataset
  2. logging.INFO: shows information about specific checkup, no action required.

We can also provide `threshold` parameter value for checkups in following format
::
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
  
SparkPipeline
______________
**'SparkPipline'** offers an abstraction over transformers and estimator pipelines in PySpark, Here is how you can use this utility in your workflow.

::
  
  # import SparkPipeline from maxairesources
  from maxairesources.pipeline.spark_pipeline import SparkPipeline

  # input training dataframe
  training = spark.createDataFrame([
          (0, "a b c d e spark", 1.0),
          (1, "b d", 0.0),
          (2, "spark f g h", 1.0),
          (3, "hadoop mapreduce", 0.0)
      ], ["id", "text", "label"])

  # input test or scoring dataframe
  test = spark.createDataFrame([
      (4, "spark i j k"),
      (5, "l m n"),
      (6, "spark hadoop spark"),
      (7, "apache hadoop")
  ], ["id", "text"])

  # create a sparkpipline of transformers/estimators and their arguments as key value pairs as shown below
  sp = SparkPipeline({'Tokenizer':{'inputCol':'text','outputCol':'words'},
   'HashingTF':{'inputCol':'words','outputCol':'features','numFeatures':1024}})

  # fit a pipeline on training data
  sp.fit_pipeline(training)

  # call transform_pipeline on fitted pipeline to transform test data
  sp.transform_pipeline(test)



  # create a sparkpipline of same set of transformers/estimators and their arguments as key value pairs for multiple columns 
  # with same pipeline
  # Example:

  # input training dataframe
  training = spark.createDataFrame([
          (0, "a b c d e spark", "machine learning", 1.0),
          (1, "b d","deep learning", 0.0),
          (2, "spark f g h", "natural language processing",1.0),
          (3, "hadoop mapreduce","computer vision", 0.0)
      ], ["id", "text","domains", "label"])

  # input test or scoring dataframe
  test = spark.createDataFrame([
      (4, "spark i j k", "machine"),
      (5, "l m n", "learning"),
      (6, "spark hadoop spark", "language"),
      (7, "apache hadoop", "vision")
  ], ["id", "text", "domains"])


  # if you have to apply the same transformations for two text columns 
  # consider below as an example. Below is the dictionary created for two text columns.
   {'Tokenizer': {'inputCol': 'text', 'outputCol': 'texttk'},
    'StopWordsRemover': {'inputCol': 'texttk', 'outputCol': 'textsw'},
    'HashingTF': {'inputCol': 'textsw','outputCol': 'texthtf','numFeatures': 1024},
    'IDF': {'inputCol': 'texthtf', 'outputCol': 'textidf'},
    'Tokenizer': {'inputCol': 'domains', 'outputCol': 'domainstk'},
    'StopWordsRemover': {'inputCol': 'domainstk', 'outputCol': 'domainssw'},
    'HashingTF': {'inputCol': 'domainssw','outputCol': 'domainshtf','numFeatures': 1024},
    'IDF': {'inputCol': 'domainshtf', 'outputCol': 'domainsidf'},
    'VectorAssembler': {'inputCol': ['textidf', 'domainsidf'],'outputCol': 'assembler_features'},
    'MinMaxScaler': {'inputCol': 'assembler_features','outputCol': 'scaled_features'}}


  text_cols = ['text','domains']
  cols = []
  transformation_dict = {}
  for i in text_cols:
      transformation_dict[i] = {'Tokenizer':{'inputCol':i,'outputCol':i+'tk'},
       'StopWordsRemover':{'inputCol':i+'tk','outputCol':i+'sw'},
       'HashingTF':{'inputCol':i+'sw','outputCol':i+'htf','numFeatures':1024},
       'IDF': {'inputCol':i+'htf','outputCol':i+'idf'}}
      cols.append(i+'idf')

  transformation_dict['vectorassembler'] = {'VectorAssembler': {'inputCols': ['textidf','domainsidf'], 'outputCol':"assembler_features"}}
  transformation_dict['MinMaxScaler'] = {'MinMaxScaler' : {'inputCol': 'assembler_features', 'outputCol':"scaled_features"}}
  transformation_dict

  sp = SparkPipeline(transformation_dict)
  sp.fit_pipeline_multiple(training)
  sp.transform_pipeline(retail_dcf_temp_label)


Logging
_______

- Generic logging method is in `maxairesources/logging/logger.py` file. use `get_logger` method to get logger object.

- Do not use logging in test cases.

- logger support 5 levels of logging as below.

::

  | Level      | When it's used                                                                                                                                                                                                                                                                                                                                                                                                                                           |
  |------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | `DEBUG`    | Detailed information, typically of interest only when diagnosing problems. <br />Example<br />- Can be used to print intermediate information to debug code blocks <br />- Number of observations, column list in `Spark` `Dataframe` <br />- Parameters received to train the model<br />- `train` and `test` data size<br /><br />Do not print any raw data / information in debug messages as some data may be confidential to display in `log` also. |
  | `INFO`     | Confirmation that things are working as expected. <br />Example<br />- Log success message once model is trained<br />- Inform that `model` is persisted in disk space                                                                                                                                                                                                                                                                                   |
  | `WARNING`  | An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.<br />Example<br />- Warn user if data size is less<br />- Highlight long processing time if model parameters grid combination for optimization are more than limit.                                                                                                               |
  | `ERROR`    | Due to a more serious problem, the software has not been able to perform some function.<br />Example<br />- If `data frame` is empty when observations are expected<br />- Fail fast model checks are not passing                                                                                                                                                                                                                                        |
  | `CRITICAL` | A serious error, indicating that the program itself may be unable to continue running.<br />Example<br />- Database credentials are incorrect<br />- Certain path is not accessible from current user                                                                                                                                                                                                                                                    |

- Currently, logger support two types of handlers

1. `FileHandler`: produce log file which could be viewed using text editor and 
2. `StreamHandler`: send log messages to `terminal` `console`. This also gets printed along with spark log

- Log format

  ```
  %(asctime)s - [ID:xxx] [%(levelname)s] - [(%(name)s) - (%(filename)s) - (%(funcName)s) - line %(lineno)d]- [%(message)s]
  ```

- Example of usage

  ```python
  from maxairesources.logging.logger import get_logger #import function
  logger = get_logger(__name__) #get logger
  logger.debug(f"log this debug message") #log debug message
  ```

Model approval
______________

**`ModelApprover`** class checks whether the model performance is good enough to export as `ONXX` file or not. 

`Approver` class needs `Evaluator` class reference along with other arguments in constructors.

All required `constructor argument` for respective `evaluator` needs to pass as a `keyword argument` . Please refer `evaluator` documentation for details.


Config Store
____________
The [HashiCorp's Vault](https://www.vaultproject.io/docs) is currently being used as a config store, to store the Py-Configs and Spark-Configs. The Vault provides the option to create a Secret Engine (represented by `mount_path` in code snippet below). All secrets are stored in a Secret Engine and can also have a directory structure. 

*Assumptions* - This module assumes that OS environment variables HASH_VAULT_URL and HASH_VAULT_TOKEN are defined. 

*Usage* - The `config_store.config.main` can be used to a function where one wants to read these secrets/configs. The best practise would be read these secrets/configs once in a task, because everytime we make a call to `config_store.config.main`, it creates a temporary token to read these secrets.

::

  *Example of Usage* - 
  ```
  PATH = ""          # Path to the Config
  MOUNT_PATH = ""    # Secret Engin

  @config.main(path=PATH, mount_point=MOUNT_PATH)
  def execute(**kwargs):
      input_data = kwargs["data"]
      print("Printing Config = {}".format(input_data))

  >> Printing Config = {'split_seed': 19, 'target_column': 'target', 'test_size': 0.2}
  ```

tutorials
=========

maxaifeaturization
==================

FeatureSelector
_______________

**'FeatureSelector'** offers an abstraction for selecting features using the methods available in pyspark feature selection, 
Class expects method to use for fearure selection and corresponding as inputs. 

Currently supported methods are
::

  selectors = {
          "VectorSlicer": {
              "model": VectorSlicer,
              "fitted_model": VectorSlicer,
              "type": "transform",
          },
          "RFormula": {"model": RFormula, "fitted_model": RFormula, "type": "transform"},
          "ChiSqSelector": {
              "model": ChiSqSelector,
              "fitted_model": ChiSqSelectorModel,
              "type": "fit",
          },
          "UnivariateFeatureSelector": {
              "model": UnivariateFeatureSelectorN,
              "fitted_model": UnivariateFeatureSelectorModel,
              "type": "fit",
          },
          "VarianceThresholdSelector": {
              "model": VarianceThresholdSelector,
              "fitted_model": VarianceThresholdSelectorModel,
              "type": "fit",
          },
      }

Here is how you can use this utility in your workflow.

::

  # import FeatureSelector from maxaifeaturization
  from maxaifeaturization.selection.selector import FeatureSelector

  # Initializing FeatureSelector class
  fs = FeatureSelector(method = 'UnivariateFeatureSelector', 
                       params = {'featuresCol':"features",
                        'outputCol':'selectedFeatures',
                        'labelCol':'label',
                        'selectionThreshold':1,
                        'featureType':'continuous',
                        'labelType':'categorical'})


  # select features using the passed method
  fs.select_features(feature_df)

  #access the underlying spark feature selection method object
  fs.selector

  # save the model
  fs.save('path')

  # load the model
  fs.load('path')



maxaimarketplace
================

maxaimetadata
=============

Max AI Metadata 

Metadata Modules offers classes and funtions to log ml-metadata for lineage tracking.


**WorkFlow**

Collection of all the elements related to a datascience workflow. 

Workflow represent a jupyter notebook for a usecase or an airflow pipeline. 

if workflow already exists in the backend , it will get reused.



::

  # import WorkFlow from maxaimetadata
  from maxaimetadata.metadata import WorkFlow

  # Initializing WorkFLow class
  wf = WorkFlow(
          name="Propensity1",
          description="test workflow",
          tags={"sample": "sample"},
          reuse_workflow_if_exists=True,
      )
  ```

**Run**

Captures a particular instance/run of the worlflow. A workflow can have multiple runs.

::

  # import WorkFlow from maxaimetadata
  from maxaimetadata.metadata import Run

  # Initializing Run class
  run = Run(workflow=wf, description="test run")
  run.update_status("running")
  ```

**Execution**

Represent a task in the workflow [training, preprocessing , validation etc]

::

  from maxaimetadata.metadata import Execution

  exec = Execution(
          name="test exec", workflow=wf, run=run, description="test execution"
      )

**Artifacts**

Artifacts reperesents input/output of any execution. Eg: Model, Data , Metrics etc


::

  from maxaimetadata.metadata import Execution, Model, DataSet, Metrics

  d = DataSet(
          uri="/data", name="test_data", description="test data", feature_view="test_iew"
      )

  d = exec.log_input(d)

  #model is any MaxAi Model
  m = Model(model=model, name="test_model", description="test model")
  m = exec.log_output(m)

  metrics = Metrics(
      name="Test Metrics", data_set_id=d.id, model_id=m.id, values={"rmse": 0.9}
  )
  metrics = exec.log_output(metrics)


**Registry**
Model registry represent a logical collection of models registered for Inference.

::

  from maxaimetadata.metadata import Registry

  r = Registry(wf)
  r.register_model(m.uri)
  r_m = r.get_registered_model("staging")
  p_m = r.promote_model(r_m["__maxai_version__"])

maxaimodel
==========

maxairesources
==============
