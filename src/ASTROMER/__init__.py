from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)
import sys, os
print(sys.path)

sys.path.append(os.path.join(sys.path[-1], 'core'))

__version__ = "0.0.12"
