import sys, os

print('-------------- PATHS -----------')
print(sys.path)
sys.path.append(os.path.join(sys.path[-1], 'core'))
print('--------------------------------')

__version__ = "0.0.4"
