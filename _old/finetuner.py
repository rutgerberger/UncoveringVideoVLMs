from unsloth import FastVisionModel
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

def finetune_model(args, model, processor, input_ids, output_ids, tubelets, tubelets_teacher_forced, ground_truth):

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,  # Set to False to freeze vision layers
        finetune_language_layers=True, # Set to True to finetune language layers
        finetune_attention_modules=True, # Set to True to finetune attention layers
        finetune_mlp_modules=True, # Set to True to finetune MLP layers

        r=16,  # A higher value increases accuracy but may risk overfitting
        lora_alpha=16,  # Recommended alpha value (usually matches r)
        lora_dropout=0.1,
        bias="none",
        random_state=3407,
        use_rslora=False,  
        loftq_config=None,
    )
    
