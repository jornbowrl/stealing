from .base import EvaluateBase


def get_evaluator(opt,**kwargs):
    name = opt.class_name 
    
    if name =="EvaluateBase":
        return EvaluateBase(**kwargs)
    
    raise Exception(f"unknow classname :{name}")