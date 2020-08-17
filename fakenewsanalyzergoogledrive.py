from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

link = "https://drive.google.com/drive/folders/1v7Ei7AAG6DZaqjf-yxS6KRbtSymyADXu?usp=sharing" #Colocar em seu Google Drive a pasta com o dataset de notícias verdadeiras e falsas 
