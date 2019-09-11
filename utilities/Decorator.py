######################################################################
#  script name  : Decorator.py
#  author       : Chen Xuanhong
#  created time : 2019/9/11 22:36
#  modification time ：2019/9/11 22:36
#  modified by  : Chen Xuanhong
######################################################################
import time

def time_it(fn):
    def new_fn(*args):
        start = time.time()
        result = fn(*args)
        end = time.time()
        duration = end - start
        print('%.4f seconds are consumed in executing function:%s'\
              %(duration, fn.__name__))
        return result
    return new_fn