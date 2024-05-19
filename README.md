# Mixed_Precision_Quantization

**Abstract**  
Large language models like GPT, Gemini, Llama, etc., have transformed the comprehension and generation of natural language tasks. Moreover, pre-trained LLMs come with exceptional language processing capabilities. However, pre-trained models require substantial memory and are compute-intensive. Quantization techniques have emerged as a promising avenue for addressing these challenges while preserving accuracy and making energy efficient. We propose CPTQuant, a novel mixed precision post-training quantization technique combining three innovative strategies to achieve a higher compression rate with a minimal drop in accuracy. CPTQuant introduces correlation-based (CMPQ), pruning-based (PMPQ), and Taylor decomposition-based (TDMPQ) mixed precision techniques. CMPQ adapts the precision level based on canonical correlation analysis of different layers, whereas PMPQ optimizes by adjusting precision layer-wise based on their sensitivity to sparsity. TDMPQ modifies the precision of layers using Taylor decomposition to assess each layer’s sensitivity to input perturbation. These strategies employ k-harmonic means clustering to allocate greater precision to more sensitive layers while diminishing precision for robust layers. CPTQuant assesses the performance of each strategy across BERT, BioBERT, OPT-1.3B, OPT-2.7B, and Llama-2-7B. We demonstrate up to a 4x compression, resulting in a 2x-fold increase in efficiency with a significantly minimized accuracy drop compared to the Hugging Face FP16 implementation. Among the three strategies, PMPQ stands out for achieving a considerably higher level of model compression. Sensitivity analyses across various LLMs show that the initial and final 30% of layers exhibit higher sensitivities than the remaining layers. Consequently, CPTQuant assigns higher precision (16-bit) to more sensitive layers, ensuring minimal information loss, while aggressive quantization (4-bit) for the less sensitive middle layers. This strategic allocation of precision levels ensures an optimal balance between efficiency and perplexity. The PMPQ method demonstrates an 11% higher compression ratio than other methods for classification tasks, while TDMPQ achieves a 30% greater compression ratio for language modelling tasks.

## Commands:

```bash
# Classification tasks
cd /Quant_LLM/Mixed_Precision_Quantization/src/Classification_tasks
python Mixed_quantization_main_Opt1_final_1.3b.py

# Language Modelling
cd /Quant_LLM/Mixed_Precision_Quantization/src/Language_Modelling
python Mixed_quantization_main_wikitext_Llama2.py

pip install awscli
aws configure
pip install sagemaker



