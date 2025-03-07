# README: Fine-Tuning for Emotion-Based Conversational AI

## Project Overview
This project focuses on fine-tuning a conversational AI model using emotion-labeled dialogue data. The dataset is processed to enhance response generation using the Cognitive Appraisal Theory (CAT) framework. The goal is to improve the AI's ability to recognize and respond empathetically to human emotions in conversations.

## Dataset Preparation
1. **Data Source:** The dataset consists of human-agent dialogues labeled with emotions such as angry, joyful, surprised, sad, and afraid.
2. **Preprocessing Steps:**
   - Data cleaning and formatting.
   - Lowercasing all text entries.
   - Removing redundant agent/customer labels.
   - Structuring conversations into a consistent format.
   - Creating a `conversation` field aggregating dialogues related to each situation.
   - Splitting the dataset into `train`, `validation`, and `test` sets.
3. **Final Dataset Fields:**
   - `situation`: Context of the conversation.
   - `emotion`: Emotion associated with the situation.
   - `conversation`: Structured conversation formatted for model training.

## Model Fine-Tuning
1. **Base Model:**
   - Meta's Llama-2-7b-chat-hf
2. **Fine-Tuning Steps:**
   - Data transformation using a structured prompt.
   - Utilizing `peft` and `bitsandbytes` for efficient parameter tuning.
   - Training using `SFTTrainer` with specific training configurations:
     - Batch size: 4
     - Learning rate: 2e-4
     - Epochs: 1
     - Optimizer: Paged AdamW 32-bit
     - Gradient accumulation: 1
3. **Inference Pipeline:**
   - Emotion classification using `distilroberta-base` model.
   - Generating responses based on CAT principles.
   - Outputs a structured response addressing emotional state, reappraisal, and coping strategies.

## Installation & Dependencies
To set up the environment, install the following dependencies:
```bash
pip install accelerate peft bitsandbytes transformers trl datasets torch tensorboard
```

## Running the Fine-Tuned Model
1. **Dataset Preparation:**
   - Preprocess dataset using provided scripts.
   - Save train, validation, and test datasets.
2. **Fine-Tune Model:**
   - Run training script using `SFTTrainer`.
   - Save the fine-tuned model.
3. **Inference Example:**
```python
from transformers import pipeline

def generate_response(human_input):
    pipe = pipeline(task="text-generation", model="llama-2-7b-chat-emo")
    return pipe(f"<s>[INST] {human_input} [/INST]")

response = generate_response("I'm feeling really anxious about my exam.")
print(response[0]['generated_text'])
```

## Results & Logging
- TensorBoard is used to monitor training progress.
- The final model can be used for emotion-aware response generation in conversational AI applications.

## Future Enhancements
- Fine-tune using more diverse emotional datasets.
- Experiment with additional response structuring techniques.
- Deploy as an API for real-time applications.

## Contact
For any questions or contributions, feel free to reach out!

