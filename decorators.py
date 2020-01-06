import functools
import time
import pathlib
import path


def timeit(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        return_ = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"Finished {func.__name__!r} in {elapsed:.4f} secs")
        return return_
    return wrapped


def mkDir(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if len(args) == 3:
            pathStr, fileName = args[1], args[2]
            pathStr = path.Path(pathStr).joinpath(fileName) if len(fileName) > 0 else pathStr
            pathStr = path.Path(pathStr)
            pathlib.Path(pathStr.parent).mkdir(exist_ok=True, parents=True)
        else:
            print("Wrong input format on mkDir, did not make.")
        return func(*args, **kwargs)
    return wrapped
