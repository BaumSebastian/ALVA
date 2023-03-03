# Import src folder
import sys
import os
from pathlib import Path

home = Path(__file__).parent.parent.parent
sys.path.append(os.path.join(home, "src"))

# Load the classifier
from .lenet5 import LeNet5

# Load the generator
from .conditional_gan import CGanGenerator, CGanDiscriminator