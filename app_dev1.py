import streamlit as st
import os
import torch
from transformers import pipeline
import time


from dotenv import load_dotenv

import boto3
#=============================================================

load_dotenv()

#=============================================================
# CONFIGURATION
#_________________________________________________________
# Access environment variables
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

#_________________________________________________________
# # Creating s3 bucket client
# '''
# To consider when creating the s3.bucket client 
#     - If you are creating an bucket in any region different to 'us-east-1'
#         then you need to specify the region in which you will be working with 
#         when creating the s3 client
#         "https://dev.to/marviecodes/aws-s3-location-constraint-error-heres-what-youre-doing-wrong-4egp" 

# '''

s3 = boto3.client('s3', region_name=AWS_DEFAULT_REGION)
#_________________________________________________________
# Paths
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/' # This is the path inside ur bucket.



#=============================================================
# Utility functions 

# Download directory from S3 bucket 
def download_dir(local_path, s3_prefix, bucket_name):
    '''

    Parameters:
    -----------
        local_path: place where you are going to store the downloaded files. 
        s3_prefix: name of the target folder insider of the s3 bucket that you want to download
    
    '''
    
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']

                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))

                # Creating inner folder of the parent (target) folder s3_prefix. 
                os.makedirs(os.path.dirname(local_file), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file)

#=============================================================





#================================================================
# Frontend 

st.title("Machine Learning Model Deployment at the Server!!!")

button = st.button("Download Model")
if button:
    with st.spinner("Downloading... Please wait!"):
        
        download_dir(local_path=local_path,
                    s3_prefix=s3_prefix, 
                    bucket_name=BUCKET_NAME,
                    )
        # time.sleep(2) # seconds 


text = st.text_area("Type to predict sentiment") 

#____________________________________________
predict = st.button("Predict")

#----- 
# movel instantiation. 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
classifier = pipeline('text-classification', 
                      model='tinybert-sentiment-analysis', 
                      device=device)

#--- 
# Execution when predict button is hit. 
if predict:
    with st.spinner("Predicting..."):
        output = classifier(text)
        st.write(output)
        # st.info(output)


