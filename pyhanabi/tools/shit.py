import os
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(lib_path)
lib_path = os.path.dirname((os.path.abspath(__file__)))
print(lib_path)
lib_path = (os.path.abspath(__file__))
print(lib_path)
lib_path = (os.path.abspath(__file__))+str('/tool')
print(lib_path)
