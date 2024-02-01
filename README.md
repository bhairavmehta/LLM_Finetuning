# Finetuning LLM for Multilingual Support in Indic Languages

The primary challenge with smaller models is their lack of multilingual capabilities. These models typically excel in English but struggle with languages like Hindi, where translating a single token requires more computational resources, leading to decreased accuracy for multilingual content.

To address this, we are enhancing a model to support Indic languages, aiming to produce high-accuracy multilingual responses. The model in focus is the Zephyr 7B, initially trained on the Mistral 7B dataset and available on Hugging Face at [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta).

This project involves fine-tuning the open-source LLM, Zephyr 7B beta, with QLoRA, as detailed on [Analytics Vidhya Community](https://community.analyticsvidhya.com/c/generative-ai-tech-discussion/what-is-qlora), and deploying it using the Hugging Face LLM Inference DLC on Amazon SageMaker. Our approach includes leveraging Hugging Face Transformers, Accelerate, and PEFT for an efficient finetuning and deployment process, incorporating advanced techniques like Flash Attention and Mixed Precision Training.

## Project Steps:

### 1. Development Environment Setup:
To work with Zephyr-7B, you'll need to authenticate with Hugging Face using the following command to access the gated repository:
```
!huggingface-cli login --token YOUR_TOKEN
```
Additionally, ensure you have the necessary SageMaker permissions through an IAM Role for a seamless local development experience.

### 2. Dataset Preparation:
We're using the BB-Ultrachat-IndicLingual6-12k dataset, a subset of the larger HuggingFaceH4/ultrachat_200k dataset, featuring multi-turn conversations across six Indic languages. This dataset was enhanced using AI4Bharat's IndicTrans2 model for accurate translations.

The dataset is reformatted into an instruction-based layout, with conversations turned into `processed_text` columns suitable for tokenization and efficient training. This preparation involves packing multiple samples into a single sequence, separated by an EOS Token, for optimized training.

### 3. Finetuning Zephyr 7B with QLoRA on Amazon SageMaker:
QLoRA's methodology involves quantizing the pre-trained model, attaching trainable adapter layers, and finetuning these layers for efficient memory usage without compromising performance. We utilize the `run_qlora.py` script, integrating PEFT for training and merging LoRA weights post-training. Techniques like Flash Attention 2 are employed to expedite the training process.

### 4. Deploying the Finetuned Model:
Deployment involves retrieving the appropriate Hugging Face LLM DLC container URI from Amazon SageMaker and setting up the HuggingFaceModel with the trained model's S3 path. The deployment configuration is optimized for cost and performance, ensuring efficient resource utilization.

### 5. Streaming Inference Requests:
The final step involves setting up AWS Lambda to invoke the SageMaker endpoint, facilitating seamless model inference. The `lambda.py` script outlines the process, utilizing `json` and `boto3` for endpoint interaction.

This project demonstrates the potential of fine-tuning LLMs for enhanced multilingual support, particularly for Indic languages, leveraging advanced AI tools and techniques for improved accuracy and efficiency.
