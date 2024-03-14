import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_OUTPUTS_PATH = os.path.join(os.getcwd(), "outputs/")
DEFAULT_MODELS_PATH = os.path.join(os.getcwd(), "models/")


def main():
    print(DEFAULT_OUTPUTS_PATH)
    print(DEFAULT_MODELS_PATH)


if __name__ == "__main__":
    main()