from models.sem_pcyc import SEM_PCYC
from models.sake import SAKE


def get_model(params_model):
    if "sem_pcyc" in params_model['model']:
        return SEM_PCYC(params_model)
    elif "sake" in params_model['model']:
        return SAKE(params_model)
    else:
        print("No model specified, using SEM-PCYC")
        return SEM_PCYC(params_model)
