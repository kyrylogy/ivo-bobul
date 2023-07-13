import math
import os

import numpy as np


def serialize(submission: list[np.ndarray], path_or_filehandle):
    max_len = max(len(p) for p in submission)
    max_len_bytes = math.ceil(math.log2(max_len) / 8)
    
    def write(f):
        f.write(max_len_bytes.to_bytes(length=1, byteorder="big", signed=False))
        for prediction in submission:
            if prediction.dtype != np.uint8:
                raise TypeError("all arrays must be of data type np.uint8")
            if prediction.ndim != 1:
                raise ValueError("all arrays must be 1D")
            length = len(prediction) - 1
            f.write(length.to_bytes(length=max_len_bytes, byteorder="big", signed=False))
            f.write(prediction.tobytes())
    
    if isinstance(path_or_filehandle, (str, bytes, os.PathLike)):
        with open(path_or_filehandle, "wb") as fh:
            return write(fh)
    return write(path_or_filehandle)


def deserialize(path_or_filehandle) -> list[np.ndarray]:
    def read(f):
        submission = []
        max_len_bytes = int.from_bytes(f.read(1), byteorder="big", signed=False)
        
        while True:
            length_bytes = f.read(max_len_bytes)
            if not length_bytes:
                return submission
            length = int.from_bytes(length_bytes, byteorder="big", signed=False) + 1
            prediction = np.frombuffer(f.read(length), dtype=np.uint8)
            submission.append(prediction)
    
    if isinstance(path_or_filehandle, (str, bytes, os.PathLike)) and os.path.isfile(path_or_filehandle):
        with open(path_or_filehandle, "rb") as fh:
            return read(fh)
    return read(path_or_filehandle)



# data = deserialize("targets_debug.data")
# predicted_data = deserialize("predictions.bin")
# print(len(data))
