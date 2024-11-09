# Mixed_Precision_Quantization

CPTQuant is a novel mixed precision quantization technique combining correlation-based, pruning-based, and Taylor decomposition-based methods to optimize LLM compression. By strategically adjusting precision levels, CPTQuant achieves up to 4x compression and 2x efficiency with minimal accuracy loss, particularly excelling in sensitive layer preservation. PMPQ provides superior compression for classification tasks, while TDMPQ excels in language modeling.

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

## Folder structure:
Mixed_precision_quantization/
│
├── src/
│   ├── Classification_tasks/
│   │   ├── Correlation based quantization
│   │   │      ├── Correlation based Mixed Precision Qunatization.py
│   │   └── Pruning based Mixed Precision Qunatization 
│   │   │      ├── Pruning based Mixed Precision Qunatization.py
│   │   └── Taylor decomposition based Mixed Precision Qunatization
│   │   │      ├── Taylor decomposition based Mixed Precision Qunatization.py
|   |   └── Main Mixed Precision Qunatization.py 
│   ├── Language modelling tasks
│   │   ├── Correlation based quantization
│   │   │      ├── Correlation based Mixed Precision Qunatization.py
│   │   └── Pruning based Mixed Precision Qunatization 
│   │   │      ├── Pruning based Mixed Precision Qunatization.py
│   │   └── Taylor decomposition based Mixed Precision Qunatization
│   │   │      ├── Taylor decomposition based Mixed Precision Qunatization.py
|   |   └── Main Mixed Precision Qunatization.py 
├── Readme.MD


    

