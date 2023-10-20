import os
import uuid
import json
import numpy as np

def uuid_factory() -> str:
    """
    Generates and returns a UUID (Universally Unique Identifier).

    Returns:
        str: A unique identifier in the form of a string.
    """
    return str(uuid.uuid1())

import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
