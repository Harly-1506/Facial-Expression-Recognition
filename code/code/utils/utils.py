import os, sys

"""
RUN FILES
"""
def run_ipynb(cmd, global_scope = globals()):
    import sys, shlex, IPython

    cmd_argv = shlex.split(cmd)
    cmd_name = cmd_argv[0]

    sys_argv = sys.argv
    sys.argv = cmd_argv

    try:
        IPython.get_ipython().magic(f"%run {cmd_name}")
    except SystemExit as ex:
        pass
 
    sys.argv = sys_argv
    global_scope.update(**locals())
    pass # run_ipynb

def run_python(cmd, global_scope = globals()):
    import sys, shlex, runpy

    cmd_argv = shlex.split(cmd)
    cmd_name = cmd_argv[0]

    sys_argv = sys.argv
    sys.argv = cmd_argv

    try:
        runpy.run_path(cmd, global_scope)
    except SystemExit as ex:
        pass

    sys.argv = sys_argv

    global_scope.update(**locals())
    pass # run_ipynb


"""
PARSER
"""
# parse params
def parse_params(node, scope = globals()):
    import os, yaml
    try:
        import json5
    except:
        import json as json5
    try:
        if type(node) is str:
            if node.startswith("eval(") and node.endswith(")"): 
                node = eval(node[5:-1], scope)
                node = parse_params(node, scope)
            elif node.startswith("json(") and node.endswith(")"): 
                node = eval("f'{}'".format(node[5:-1]), scope)
                if os.path.exists(node) == True:
                    with open(node, "rt") as file: 
                        node = parse_params(json5.load(file), scope)
            elif node.startswith("yaml(") and node.endswith(")"): 
                node = eval("f'{}'".format(node[5:-1]), scope)
                if os.path.exists(node) == True:
                    with open(node, "rt") as file: 
                        node = parse_params(yaml.load(file), scope)                        
            else:
                node = eval("f'{}'".format(node), scope)
                if str.isnumeric(node): node = eval(node)
        elif type(node) is dict:
            for key in node:
                node[key] = parse_params(node[key], {**scope, **node})
        elif type(node) in [list, tuple, set]:
            node = list(node)
            for pos, sub_node in enumerate(node):
                node[pos] = parse_params(sub_node, scope)
    except:
        pass
    return node
    pass # parse_params

"""
TEE_LOG
"""
# Context manager that copies stdout and any exceptions to a log file
class TeeLog(object):
    """
    https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    """
    def __init__(self, filename = None, mode = "wt"):
        self.files = []
        self.filenames = []
        if filename is not None: 
            self.append(filename, mode)
        self.stdout = sys.stdout
    # __init__

    def append(self, filename, mode = "wt"):
        try:
            file = open(filename, mode) if filename is not None and filename != "" else None
        except:
            file = None
        # try
        if file is not None: 
            self.files.append(file)
            self.filenames.append(filename)
    # init

    def __enter__(self):
        sys.stdout = self
        return self
    # __enter__

    def __exit__(self, exc_type = None, exc_value = None, tb = None):
        import traceback
        sys.stdout = self.stdout
        if exc_type is not None:
            if exc_type is SystemExit and str(exc_value)!='0':
                for file in self.files: 
                    try:
                        file.write(traceback.format_exc())
                    except:
                        pass
        for file in self.files: 
            try:
                file.close()
            except:
                pass
        pass # __exit__

    def __del__(self):
        try:
            self.__exit__()
        except:
            pass
        pass

    def write(self, data):
        for pos, file in enumerate(self.files):
            try:
                file.write(data)
                file.flush()
            except:
                self.files[pos] = open(self.filenames[pos], "at")
                pass
        self.stdout.write(data)
        pass
    # write

    def flush(self):
        for pos, file in enumerate(self.files): 
            try:
                file.flush()
            except:
                self.files[pos] = open(self.filenames[pos], "at")
                pass
        self.stdout.flush()
        pass
    # flush
# TeeLog    

class JupyterTeeLog:
    """
    JupyterTeeLog
    + sys.stdout on jupyter: <header> starting with header_tpl, <content>
    --> only override write method
    else override sys.stdout
    """
    def __init__(self, filename = None, mode = "wt", header_tpl = '<xml><var name='):
        self.files = []
        self.filenames = []
        self.modes = []
        if filename is not None: self.append(filename, mode)
        if not hasattr(sys.stdout, "org_write"):
            self.std_write = sys.stdout.write
        else:
            self.std_write = sys.stdout.org_write

        self.stdout = sys.stdout
        self.header_tpl = header_tpl # no write with header_tpl (for Jupyter)
        pass # __init__

    def append(self, filename, mode = "wt"):
        try:
            file = open(filename, mode) if filename is not None and filename != "" else None
        except:
            file = None
        pass # try

        if file is not None: 
            self.files.append(file)
            self.filenames.append(filename)
        pass # append

    # context manager
    def __enter__(self):
        if not hasattr(sys.stdout, "org_write"):
            setattr(sys.stdout, "org_write", self.std_write)
        sys.stdout.write = self.write
        return sys.stdout
        pass # enter

    def __exit__(self, exc_type = None, exc_value = None, tb = None):
        import traceback
        if hasattr(sys.stdout, "org_write"):
            setattr(sys.stdout, "write", self.std_write)
            delattr(sys.stdout, "org_write")
        if exc_type is not None:
            if exc_type is SystemExit and str(exc_value)!='0':
                for file in self.files:
                    try:
                        if not file.closed: file.write(traceback.format_exc())
                    except:
                        pass
        for file in self.files: 
            try:
                file.close()
            except:
                pass
        pass # __exit__

    def check_header(self, data):
        if self.header_tpl == "": return False
        if data.startswith(self.header_tpl) is False: return False
        return True
        pass # check_header

    def __del__(self):
        self.__exit__()
        pass

    def write(self, data):
        for pos, file in enumerate(self.files):
            try:
                if not self.check_header(data):
                    file.write(data)
                    file.flush()
            except:
                sys.stdout.org_write("Error")
                self.files[pos] = open(self.filenames[pos], "at")
                pass

        if not hasattr(sys.stdout, "org_write"):
            sys.stdout.org_write = self.std_write

        if not hasattr(self, "stdout"):
            self.stdout = sys.stdout
        
        self.stdout.write = self.write
        self.std_write(data)
        pass # write

    def flush(self):
        for pos, file in enumerate(self.files):
            try:
                if not file.closed: file.flush()
            except:
                self.files[pos] = open(self.filenames[pos], "at")
                pass
        sys.stdout.flush()
        pass # flush

    pass # JupyterTeeLog