import os
import sys
import inspect

path = inspect.getfile(inspect.currentframe())
path = os.path.dirname( os.path.abspath(path))
sys.path.insert(0, path + '/main')
sys.path.insert(0, path + '/main/model')
from main.CognitiveGridworld import CognitiveGridworld