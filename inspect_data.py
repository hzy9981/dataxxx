import scipy.io
data = scipy.io.loadmat('/home/hzy9981/datax/dataX.mat')
print("Keys in .mat file:", data.keys())
for key in data.keys():
    if not key.startswith('__'):
        print(f"Key: {key}, Shape: {data[key].shape}")
