import os


def dir_num_ok(file_path, expect):
    return len(os.listdir(file_path))==expect


def file_size_ok(file_path, expect):
    return os.stat(file_path).st_size==expect


def file_ok(file_path, expect):
    if os.path.isfile(file_path):
        return file_size_ok(file_path, expect)
    elif os.path.isdir(file_path):
        return dir_num_ok(file_path, expect)
    else:
        raise NotImplemented('Incorrect file type')