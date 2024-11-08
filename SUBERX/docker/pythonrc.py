import warnings

# Suppress the specific FutureWarning from torch
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')


