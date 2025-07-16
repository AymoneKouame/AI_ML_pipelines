## ~~~~~~~~~~~~~~~  Example 1: Collecting data from GBQ

# detector = InsuranceFraudDetector(service_account_filepath = 'windy-smoke-420803-05e83bb28a9b.json'
#                                   , project_id = 'bigquery-public-data', dataset_name = 'fhir_synthea'
#                                  , data_query = '''SELECT * FROM claim LIMIT 2'''
#                                  , target_column = '')

# results_dd, best_model_name, classification_report_tb, plots = detector.run_complete_pipeline()


## ~~~~~~~~~~~~~~~ Example 2: Collecting data from Google Cloud Bucket or Locally (currently supports, csv, excel and parquet files)

#detector = InsuranceFraudDetector(data_filepath='gs://bucket_id/data/Outpatientdata_insurance_claim.csv')
#results_dd, best_model_name, classification_report_tb, plots = detector.run_complete_pipeline()

## ~~~~~~~~~~~~~~~ Example 32: Using a loaded dataframe
detector = InsuranceFraudDetector(data_frame = insurance_claim_synth_data)
results_dd, best_model_name, classification_report_tb, plots =detector.run_complete_pipeline()