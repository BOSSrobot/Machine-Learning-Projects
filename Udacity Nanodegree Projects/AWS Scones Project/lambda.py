############################ serializeImageData

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3 = boto3.resource('s3')
    s3.Bucket(bucket).download_file(key, "/tmp/image.png")

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
        }
    }

############################ classifyFunction

import json
import base64
import boto3

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2023-10-16-23-59-25-032'
runtime = boto3.Session().client('sagemaker-runtime')

def lambda_handler(event, context):
    
    clean_event = event['body']

    # Decode the image data
    image = base64.b64decode(clean_event['image_data'])

    # # Instantiate a Predictor
    # predictor = sagemaker.Predictor(ENDPOINT)

    # # For this model the IdentitySerializer needs to be "image/png"
    # predictor.serializer = IdentitySerializer("image/png")
    
    # # Make a prediction:
    # inferences = predictor.predict(event['image_data'])
    
    # # We return the data back to the Step Function    
    # event["inferences"] = inferences.decode('utf-8')
    
    response = runtime.invoke_endpoint(EndpointName = ENDPOINT, ContentType = 'image/png', Body = image)
    predictions = json.loads(response['Body'].read().decode())
    
    clean_event['inferences'] = predictions
    
    return {
        'statusCode': 200,
        'body': clean_event
    }

############################ filterInferences

import json

THRESHOLD = .70

def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event['body']['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = [inf >= THRESHOLD for inf in inferences]
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if not any(meets_threshold):
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event['body']
    }