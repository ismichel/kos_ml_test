# KOS ML Engineer Work Trial: Building HERA Medical Chatbot

## Overview
Welcome to the KOS ML Engineer work trial! Your task is to build a medical chatbot similar to our HERA model from scratch in one day. You'll fine-tune a 1-3B parameter language model on medical data and demonstrate measurable improvements in medical knowledge and conversation quality.

## Objective
Build, train, and deploy a medical chatbot that can:
1. Answer medical questions accurately
2. Provide helpful health information
3. Maintain context in conversations
4. Show clear improvement from baseline to fine-tuned model

## Timeline (8 Hours)
- **Hour 1-2**: Environment setup & data sourcing
- **Hour 3-4**: Data preprocessing & baseline evaluation
- **Hour 5-6**: Model fine-tuning
- **Hour 7**: Evaluation & metrics comparison
- **Hour 8**: Ollama deployment & documentation

## Technical Requirements

### 1. Infrastructure Setup (Hour 1-2)

#### RunPod GPU Access
- **Email**: aa@kos-ai.com
- **Password**: @(Vpjtqwvhesoyam)
- **Important**: Please terminate your pod when not actively using it to save costs

#### SSH Access to RunPod
1. Log into RunPod with provided credentials
2. Create/select a GPU pod (RTX 3090/4090 or A40/A100 recommended)
3. Once pod is running, copy the SSH command from the pod details
4. SSH into the pod: `ssh root@[pod-ip] -p [port]`
5. Install required dependencies

#### Required Setup
```bash
# Update system
apt update && apt upgrade -y

# Install Python dependencies
pip install transformers datasets accelerate peft bitsandbytes
pip install ollama openai pandas numpy matplotlib seaborn
pip install scikit-learn rouge-score bert-score
pip install jupyterlab notebook
```

### 2. Data Sourcing
You need to find and use appropriate medical datasets. Here are some suggestions to research:

#### Publicly Available Medical Datasets
- **MedQA**: Medical question answering dataset
- **MedMCQA**: Medical multiple choice questions
- **PubMedQA**: Biomedical question answering
- **HealthCareMagic**: Medical consultation conversations
- **MedDialog**: Medical dialogue datasets
- **MIMIC-III**: Clinical notes (requires PhysioNet access)
- **i2b2 Datasets**: Clinical NLP challenges
- **Medical Transcriptions** from Kaggle
- **WebMD Q&A** datasets
- **BioBERT pre-training corpora**

#### Dataset Requirements
- Minimum 1,000 examples for training
- Should include medical Q&A pairs or conversations
- Consider data quality and medical accuracy
- Ensure appropriate licensing for use

### 3. Model Selection
Choose one of these base models (1-3B parameters):
- **Microsoft Phi-2** (2.7B) - Recommended for medical tasks
- **StableLM-3B** (3B) - Good general performance
- **Qwen-1.8B** (1.8B) - Efficient option
- **Gemma-2B** (2B) - Google's efficient model
- **Mistral-7B** (if you can optimize it to fit)

### 4. Implementation Requirements

#### Data Preprocessing
Create a data preprocessing pipeline that:
- Loads and cleans your chosen dataset(s)
- Formats data for instruction tuning
- Creates train/validation/test splits
- Handles medical terminology appropriately

#### Training Pipeline
Implement efficient fine-tuning using:
- **LoRA** or **QLoRA** for memory efficiency
- Appropriate hyperparameters for your time constraint
- Proper evaluation metrics

#### Evaluation Framework
Create evaluation that measures:
1. **Medical Accuracy**: Correctness of medical information
2. **Response Quality**: Coherence and helpfulness
3. **Safety**: Appropriate medical disclaimers
4. **Before/After Comparison**: Clear improvement metrics

#### Ollama Deployment
1. Convert your fine-tuned model to GGUF format
2. Create an Ollama modelfile
3. Deploy and test locally
4. Demonstrate chat functionality

## Deliverables

### 1. Code Repository
Submit a GitHub repository with:
- All code written from scratch
- Clear documentation
- Requirements.txt
- Step-by-step instructions to reproduce

### 2. Performance Report
Document including:
- Dataset selection rationale
- Training approach and decisions
- Performance metrics (before/after)
- Sample conversations showing improvement
- Challenges and solutions

### 3. Live Demo
Be prepared to:
- Show your model running in Ollama
- Demonstrate improvement over base model
- Walk through your code and approach

## Evaluation Criteria

1. **Technical Execution (40%)**
   - Code quality and organization
   - Efficient resource utilization
   - Proper ML practices

2. **Model Performance (30%)**
   - Measurable improvement
   - Medical accuracy
   - Response quality

3. **Problem Solving (20%)**
   - Dataset selection
   - Creative solutions
   - Time management

4. **Documentation (10%)**
   - Clear code documentation
   - Reproducible results
   - Professional presentation

## Important Notes

### Resource Management
- **CRITICAL**: Terminate your RunPod instance when not in use
- Monitor GPU memory usage
- Use efficient training techniques (QLoRA recommended)

### Time Management
- Don't spend too much time on any single component
- Get a basic version working first, then iterate
- Document as you go

### Medical Considerations
- Ensure your model includes appropriate disclaimers
- Focus on general health information, not diagnoses
- Consider safety in your training data selection

## Tips for Success

1. **Start Simple**: Get end-to-end pipeline working first
2. **Use Existing Tools**: Leverage HuggingFace libraries
3. **Monitor Progress**: Set up logging early
4. **Test Incrementally**: Verify each step works
5. **Document Decisions**: Keep notes for your report

## Submission Instructions

1. Push all code to GitHub (public or private repo)
2. Share repository access
3. Submit performance report (PDF format)
4. Be ready for technical discussion

## Questions During Trial
- Document any blockers you encounter
- Make reasonable assumptions and document them
- Focus on demonstrating your skills within the time constraint

Good luck! We're looking forward to seeing your approach to building a medical AI assistant.

---
*Remember: This is a time-boxed exercise. We're evaluating your technical skills, problem-solving ability, and how you handle real-world constraints. Perfect results are not expected - we want to see your methodology and execution.*
