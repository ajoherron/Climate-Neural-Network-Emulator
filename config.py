# Type of model ('CNN', 'RESNET')
#MODEL = 'RESNET'
MODEL = "CNN"

# Model settings
SLIDER_LENGTH = 1
NUM_FOLDS = 2

# Training duration / rate
EPOCHS = 150
LEARNING_RATE = 0.005 # CNN
#LEARNING_RATE = 0.00005 # ResNet

# Ensemble members, SSP experiments to include
RUN_ID_MS = ["E213SSP585", "E213SSP245", "E213SSP126"]

# Physical forcings
INPUT_LIST = [
    "CO2n",
    "CH4",
    "SO2",
    "BCII",
    "Alkenes",
    "Paraffin",
    "NOx",
    "NH3",
    "CO",
    "OCII",
]

# Target variable
VARIABLE = ["prec"]
