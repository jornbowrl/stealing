from .base import EvaluateBase


def get_evaluator(opt,**kwargs):
    name = opt.name 
    
    if name =="EvaluateBase":
        return EvaluateBase(**kwargs)
    
    raise Exception(f"unknow classname :{name}")