from pathlib import Path

# -- Path Configuration --
SRC_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SRC_DIR.parent
MALE_MODEL_PATH = PROJECT_ROOT / 'models' / 'model_m.h5'
FEMALE_MODEL_PATH = PROJECT_ROOT / 'models' / 'model_f.h5'

# -- Model Configuration --
MALE_VOICE_SCALE = 3.38
FEMALE_VOICE_SCALE = 3.39
SAMPLE_RATE = 24000

# -- UI Configuration --
DEFAULT_TEXT = 'Hello. This is a voice predicted from the face image above.'

# -- Validation Configuration --
SESSION_STATE_KEYS = ['age', 'sex', 'wav']
ALLOWED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png']
MAX_TEXT_LENGTH = 200