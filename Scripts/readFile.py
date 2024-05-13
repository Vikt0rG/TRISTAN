import struct
import numpy as np

# =======================================================================================================
# Defining functions
# =======================================================================================================

def readFile(file):
    with open(file, "rb") as dfile:
        dfile.seek(21)  # Position where n_samples is saved in the file
        n_samples = struct.unpack('i', dfile.read(4))[0]
        datatype = np.dtype([
            ('board', np.uint16),
            ('channel', np.uint16),
            ('timestamp', np.uint64),
            ('energy', np.uint16),
            ('flags', np.uint32),
            ('wavecode', np.uint8),
            ('n_samples', np.uint32),
            ('wave', np.int16, n_samples)])
        dfile.seek(2)
        data = np.fromfile(dfile, dtype=datatype)
    return data