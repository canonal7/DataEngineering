from google.cloud import storage
import os
import csv
import io

# Set this to the location of your service account key file.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/pedro/Documents/JADS-Premaster/Machine--Learning--JADS/DataEngineering/data-engineering-2023-401511-54e3e5b8b81a.json"

# Instantiates a client
storage_client = storage.Client()

# The ID of your GCS bucket
bucket_name = "noshows_dataset"
# The ID of your GCS object
blob_name = "KaggleV2-May-2016.csv"

# Construct the full path for the blob
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(blob_name)

# Download the data as a string
csv_data = blob.download_as_text()

# You can use the CSV module to read this data now.
csv_reader = csv.reader(io.StringIO(csv_data))

# Do something with the data here. For example, print it.
for row in csv_reader:
    print(row)