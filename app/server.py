from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
#import uvicorn, aiohttp, asyncio
from io import BytesIO
import numpy as np
import re
import base64
from PIL import Image
from scipy.misc import imsave,imread,imresize
from keras.models import load_model
import json 
import tensorflow as tf


# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
file_id = '1SGfDjrpHJHxglvE7FilqiN4xIpi4ciix'
export_file_name = 'fruits.h5'

FRUITS = {0: "Apple", 1: "Book", 2: "Cactus", 3: "Cake",4:"Crown",5:" Diamond",6:"Donut",7:"Fan",8:"Fish",
          9:"Flower",10:"House",11:"Hurricane",12:"Ladder",13:"Pants",14:"Tree"}


classes = ["apple.npy", "book.npy", "cactus.npy", "cake.npy","crown.npy","diamond.npy","donut.npy","fan.npy",
         "fish.npy","flower.npy","house.npy","hurricane.npy","ladder.npy","pants.npy","tree.npy"]
classes = [x.split('.')[0] for x in classes]

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


def download_file_from_google_drive(id, destination):
    if destination.exists(): 
        return

    session = requests.Session()
    
    URL = "https://docs.google.com/uc?export=download"
    
    response = session.get(URL, params = {'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params= {'id':id, 'confirm': token}
        response = session.get(URL, params= params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def setup_learner():
    download_file_from_google_drive(file_id, path/export_file_name)
    try:
        learn = load_model(export_file_name)
        return learn
    except RuntimeError as e:
        print('error loading the model')

model = setup_learner()

def normalize(data):
    return np.interp(data,[0,255],[-1,1])


@app.route('/')
def index(request):
"""    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())"""
    if request.method == "GET":
        html = path/"view"/"index1.html"
        return HTMLResponse(html.open().read())
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]
        #net = request.form["net"]

        img = base64.b64decode(data)
        with open('temp.png', 'wb') as output:
            output.write(img)
        x = imread('temp.png', mode='L')
        # resize input image to 28x28
        x = imresize(x, (28, 28))
        #net = "ConvNet":
        #model = conv
        x = np.expand_dims(x, axis=0)
        x = np.reshape(x, (28, 28, 1))
        # invert the colors
        x = np.invert(x)
        # brighten the image by 60%
        for i in range(len(x)):
            for j in range(len(x)):
                if x[i][j] > 50:
                    x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = normalize(x)
        val = model.predict(np.array([x]))
        pred = FRUITS[np.argmax(val)]
        #classes = ["Apple", "Banana", "Grape", "Pineapple"]
        print (pred)
        print( list(val[0]))
        html = path/"view"/"index1.html"
        return HTMLResponse(html.open().read(), preds=list(val[0]), classes=json.dumps(classes), chart=True, putback=request.form["payload"])


"""@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})"""

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
