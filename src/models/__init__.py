from src.models.sem_pcyc import SEM_PCYC


def get_model(params_model):
    if "sem_pcyc" in params_model['model']:
        return SEM_PCYC(params_model)
    else:
        print("No model specified, using SEM-PCYC")
        return SEM_PCYC(params_model)
