# Analyzing and Evaluating Faithfulness in Dialogue Summarization (EMNLP 2022)

For more details, please find our paper - [Analyzing and Evaluating Faithfulness in Dialogue Summarization](https://arxiv.org/abs/2210.11777)

## Human Evaluation Results

- [Results and descriptions](https://github.com/BinWang28/FacEval/tree/main/human_result)
- CSV format + 150 dialogues + 5 candidates per dialogue + 6 error types

## Generated FacEval Dataset

We provide our generated FacEval dataset and also the source code to automatically generate them, which can be further improved and extended to other datasets.

The following rest of the code is tested with the following environment:
1. python==3.7
2. nltk==3.7
3. numpy==1.21.5
4. torch==1.10.1 
5. torchvision==0.11.2 
6. torchaudio==0.10.1
7. torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
8. transformers==4.13
9. accelerate==0.5.1
10. datasets==2.4.0




FacEval dataset

Demo to generate factually corrupted samples:


## Evaluation Demo on BART-Large Model

We provide the evaluation demo on BART-Large model. Because our proposed model-level evaluation needs the direct access to the model's generation probabilities. The model needs to be loaded as well for testing.

## References

If you find our work useful, please consider citing our work.

```
@article{wang2022analyzing,
  title={Analyzing and Evaluating Faithfulness in Dialogue Summarization},
  author={Wang, Bin and Zhang, Chen and Zhang, Yan and Chen, Yiming and Li, Haizhou},
  journal={arXiv preprint arXiv:2210.11777},
  year={2022}
}
```

```
@article{to update with the EMNLP Processings,
  title={==},
  author={==},
  journal={==},
  year={==}
}
```

Contact to Bin Wang at [bwang28c@gmail.com](mailto:bwang28c@gmail.com) for any issues.