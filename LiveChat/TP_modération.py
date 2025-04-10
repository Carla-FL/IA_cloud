import boto3
import os
from dotenv import load_dotenv
load_dotenv()
session = boto3.Session(
aws_access_key_id=os.getenv(aws_access_key_id),
aws_secret_access_key=os.getenv(aws_secret_access_key),
region_name='us-east-1'
)
comprehend = session.client('comprehend')

text = "I really hate working with AWS services."
response = comprehend.detect_sentiment(Text=text, LanguageCode='en')
print(response['Sentiment'])

def modération(text):
    input_text = text
    response = comprehend.detect_sentiment(Text=input_text, LanguageCode='en')
    print(response['Sentiment'])
    return response['Sentiment']