import scipy.io
import numpy as np
import pandas as pd

data = scipy.io.loadmat('dataX.mat')
X_flat = data['inputtrain'].flatten()
y_flat = data['outputtrain'].flatten()

def check_order(X_flat, y_flat, order):
    X = X_flat.reshape(500, 500, order=order)
    y = y_flat.reshape(500, 500, order=order)
    
    df = pd.DataFrame({'y': y.flatten(), 'x': X.flatten()})
    df['x_up'] = df['x'].shift(500)
    df['x_left'] = df['x'].shift(1)
    
    print(f"\nOrder: {order}")
    print(df.dropna().corr()['y'])

check_order(X_flat, y_flat, 'C')
check_order(X_flat, y_flat, 'F')
