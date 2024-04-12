# LLM-VRAM-Calculator

Calculate VRAM (GPU Memory) required for Train and Inference any LLM Models

Just Run `python main.py` and set your requirements like this-

```
Enter Model Size in billion (e.g.: 7b, 13b, 70b) [Default=7b]: 70b
Enter LORA Trainable Parameters in billion (e.g.: 0.5b, 0.3b) [Default=0.5b]: 1b
Model Name (e.g.: 'teknium/OpenHermes-2.5-Mistral-7B') [Default='teknium/OpenHermes-2.5-Mistral-7B']: NousResearch/Llama-2-70b-chat-hf
Enter Sequence Length (e.g.: 1024, 2028, 4096 etc.) [Default=4096]: 4096
Enter Batch Size (1, 2, 4, 6, 8) [Default=1]: 1
```

Output-

```
============================== Name: NousResearch/Llama-2-70b-chat-hf | dtype: F32 | seq_length: 4096 | batch: 1 ==============================
Lora Model with 1.0 parameter in F32:  18.62645149230957 GB approx
kv_cache:  20.0 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITH GPU OVERHEAD and kv_cache]: 1584.621925354004 GB
TOTAL Training Size of Model 70bb Model Loaded in F32 (QUANT) Size with QLORA [WITH GPU OVERHEAD and kv_cache]: 355.27612686157227 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITHOUT GPU OVERHEAD and kv_cache]: 1323.85160446167 GB
TOTAL Training Size of Model 70bb Model Loaded in F32 (QUANT) Size with QLORA [WITHOUT GPU OVERHEAD and kv_cache]: 299.39677238464355 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in F32 (QUANT): 260.770320892334 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in F32 (QUANT): 260.770320892334 GB
============================== Name: NousResearch/Llama-2-70b-chat-hf | dtype: FP16 | seq_length: 4096 | batch: 1 ==============================
Lora Model with 1.0 parameter in FP16:  9.313225746154785 GB approx
kv_cache:  10.0 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITH GPU OVERHEAD and kv_cache]: 792.310962677002 GB
TOTAL Training Size of Model 70bb Model Loaded in FP16 (QUANT) Size with QLORA [WITH GPU OVERHEAD and kv_cache]: 177.63806343078613 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITHOUT GPU OVERHEAD and kv_cache]: 661.925802230835 GB
TOTAL Training Size of Model 70bb Model Loaded in FP16 (QUANT) Size with QLORA [WITHOUT GPU OVERHEAD and kv_cache]: 149.69838619232178 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP16 (QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP16 (QUANT): 130.385160446167 GB
============================== Name: NousResearch/Llama-2-70b-chat-hf | dtype: Int8 | seq_length: 4096 | batch: 1 ==============================
Lora Model with 1.0 parameter in Int8:  4.656612873077393 GB approx
kv_cache:  5.0 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITH GPU OVERHEAD and kv_cache]: 396.155481338501 GB
TOTAL Training Size of Model 70bb Model Loaded in Int8 (QUANT) Size with QLORA [WITH GPU OVERHEAD and kv_cache]: 88.81903171539307 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITHOUT GPU OVERHEAD and kv_cache]: 330.9629011154175 GB
TOTAL Training Size of Model 70bb Model Loaded in Int8 (QUANT) Size with QLORA [WITHOUT GPU OVERHEAD and kv_cache]: 74.84919309616089 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in Int8 (QUANT): 65.1925802230835 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in Int8 (QUANT): 65.1925802230835 GB
============================== Name: NousResearch/Llama-2-70b-chat-hf | dtype: Int4 | seq_length: 4096 | batch: 1 ==============================
Lora Model with 1.0 parameter in Int4:  2.3283064365386963 GB approx
kv_cache:  2.5 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITH GPU OVERHEAD and kv_cache]: 198.0777406692505 GB
TOTAL Training Size of Model 70bb Model Loaded in Int4 (QUANT) Size with QLORA [WITH GPU OVERHEAD and kv_cache]: 44.40951585769653 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITHOUT GPU OVERHEAD and kv_cache]: 165.48145055770874 GB
TOTAL Training Size of Model 70bb Model Loaded in Int4 (QUANT) Size with QLORA [WITHOUT GPU OVERHEAD and kv_cache]: 37.424596548080444 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in Int4 (QUANT): 32.59629011154175 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in Int4 (QUANT): 32.59629011154175 GB
============================== Name: NousResearch/Llama-2-70b-chat-hf | dtype: nf4 | seq_length: 4096 | batch: 1 ==============================
Lora Model with 1.0 parameter in nf4:  2.3283064365386963 GB approx
kv_cache:  2.5 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITH GPU OVERHEAD and kv_cache]: 198.0777406692505 GB
TOTAL Training Size of Model 70bb Model Loaded in nf4 (QUANT) Size with QLORA [WITH GPU OVERHEAD and kv_cache]: 44.40951585769653 GB
TOTAL Training Size of Model 70bb Model Loaded in FP32 (No QUANT) Size with Full Train [WITHOUT GPU OVERHEAD and kv_cache]: 165.48145055770874 GB
TOTAL Training Size of Model 70bb Model Loaded in nf4 (QUANT) Size with QLORA [WITHOUT GPU OVERHEAD and kv_cache]: 37.424596548080444 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in nf4 (QUANT): 32.59629011154175 GB
TOTAL Inference Size of Model 70bb Model Loaded in FP32 (No QUANT): 130.385160446167 GB
TOTAL Inference Size of Model 70bb Model Loaded in nf4 (QUANT): 32.59629011154175 GB
--------------------------------------------------: NOTE ADD 20GB-30GB Extra for Safe Scenario :--------------------------------------------------
```


## TODO

Streamlit App

# FAQs
- Calculation are accurate or mistmatch with your use case?
  - Please update the formulas used in project, raise 
a issue
