from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
import sys
print(sys.path)

sys.path.append(sys.path[0], 'core')

__version__ = "0.0.10"
