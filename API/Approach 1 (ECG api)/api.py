# Importing Necessary modules
from fastapi.responses import FileResponse
from fastapi import FastAPI, File, UploadFile
import uvicorn
import os
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

model_path = 'ECG_DT.pkl'
app = FastAPI(
    title='Afib Predictor',
    description="<ul><li>Select the <b>'/predict'</b> method and click <b>'try it out'</b></li><li>Upload an ECG file</li><li>click on <b>'execute'</b>,scroll down to see the result in response body</li>"
    )

model = None
em_model = None
encoder = LabelEncoder()

## Load models on startup or first run of API
@app.on_event("startup")
def load_models():
  global model
  global imname
  # Loading model for Afib prediction
  model = pickle.load(open(model_path, 'rb'))
  imname = 'img.png'

  print('MODELS LOADED !!!\n')

## Instructions TO USE API when link is opened
@app.get('/',include_in_schema=False)
def index():
    return {
        'Message':'Hello!',
        '1)':'redirect to [link_generated]/docs',
        '2)':'Select upload method',
        '3)':'Click on Try it Out',
        '4)':'Upload an ECG file and click execute',
        '5)':'After loading is complete you can view the results in Responses section'
        }


## Prediction is done: 
## File upload -> Execute -> Get name, distance, Registration status
@app.post("/predict")
async def prediction(file: UploadFile = File(...)):

    # Read file
    contents = await file.read()
    ecg = np.load(BytesIO(contents))
    X = ecg
    n,r = X.shape
    out_arr = np.column_stack((np.repeat(1,n),X.reshape(1*n,-1)))
    out = [0,0,0]
    num_di = {0:'SR', 1:'AF', 2:'VA'}
    for i in out_arr:
        temp=list(i[1:])
        prediction = int(model.predict([temp])[0])
        out[prediction]+=1
    # Predict distance of input from available classes
    prediction = num_di[out.index(max(out))]
    print('PREDICTION DONE !!!!!!')

    fig,ax = plt.subplots(12,1,figsize=(20,25))
    #plt.plot(ecg)
    s_name=['I','II','III','aVF','aVR','aVL','V1','V2','V3','V4','V5','V6']
    for i in range(12):
        ax[i].plot(ecg[:,i])
        ax[i].title.set_text('Rod '+s_name[i])
    fig.tight_layout(pad=3.0)
    fig.savefig('img.png')   # save the figure to file
    plt.close(fig)    # close the figure window
    # Setting THRESHOLD as 30
    #msg = 'Face Registered'
    #if prob>90: msg = 'Face Not registered'
    
    return {
        #'message':{msg},
        'SR':{out[0]},
        'AF':{out[1]},
        'VA':{out[2]},
        'Prediction':{prediction} 
        }

@app.get("/image")
def image_endpoint():
    # Returns a cv2 image array from the document vector, *, vector
    return FileResponse(imname, media_type='image/png')