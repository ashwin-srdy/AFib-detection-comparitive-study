from flask import Flask, request, render_template
import numpy as np
import pickle
app=Flask(__name__)
xx = ['lollytest']
model = pickle.load(open("22f_rf.sav", 'rb'))
print('Model Loaded')
@app.route("/", methods=['GET', 'POST'])
def upload_file():
    return render_template('form.html')

@app.route("/predict", methods=['GET', 'POST'])
def scrape():
    df_name=None
    if request.method == 'POST':
        ind=request.form                  # returns an immutable dictionary
        
        l=dict(ind)
        print(l)
        print(request.files['file'])
        f = request.files['file']
        
        ecg = np.load(f)
    inp = [float(i) for i in ind.values()]
    # convert 3d array to 2d array and convert it to a dataframe
    X = ecg
    n,r = X.shape
    out_arr = np.column_stack((np.repeat(1,n),X.reshape(1*n,-1)))
    out = [0,0,0]
    num_di = {0:'SR', 1:'AF', 2:'VA'}
    for i in out_arr:
        temp=list(i[1:])
        temp.extend(inp)
        prediction = int(model.predict([temp])[0])
        out[prediction]+=1
    prediction = num_di[out.index(max(out))]
    return render_template('out.html',p='Prediction is : '+prediction)
    
app.run(host='0.0.0.0',port=8080)