# Type of model ('CNN', 'RESNET')
#MODEL = 'RESNET'
MODEL = 'CNN'

# Model settings
BATCH_SIZE = 16
EPOCHS = 150
SLIDER_LENGTH = 1

#LEARNING_RATE = 0.001 # This should probably be lower
LEARNING_RATE = 0.0005
#LEARNING_RATE = 0.00001

# Ensemble members, SSP experiments to include
ENS_MEMS = ["a", "b", "c", "d", "e"]
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
