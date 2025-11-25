from .dab_deformable_detrmodule import DABDeformableDetrModule
from .salience_sggmodule import SalienceSGGModule

MODELS={
     'DABDeformableDETR': DABDeformableDetrModule,
    'SalienceSGG': SalienceSGGModule,
}
def build_model(config, model_name):
    return MODELS[model_name](config)

