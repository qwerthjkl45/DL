import os
scores = {} # scores is an empty dict already
if os.path.getsize('./model_parameter.txt') > 0:      
    with open(target, "rb") as f:
        unpickler = pickle.Unpickler(f)
        # if file is not empty scores will be equal
        # to the value unpickled
        scores = unpickler.load()
        print("123");
print(123)

