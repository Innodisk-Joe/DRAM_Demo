import os
from distutils.core import setup
from Cython.Build import cythonize

py_list=[]
for root,dirs,files in os.walk(os.getcwd()):
    for name in files:
        if root==os.path.join(os.getcwd(),"routers"):
            continue
        if os.path.splitext(name)[1]==".py" :
            if not (os.path.splitext(name)[0]=="setup" or os.path.splitext(name)[0]=="main"):
                if os.path.isfile(os.path.join(root, f"{os.path.splitext(name)[0]}.c")):
                    os.remove(os.path.join(root, f"{os.path.splitext(name)[0]}.c"))
                py_list.append(os.path.join(root, name))
         
setup(ext_modules = cythonize(py_list))