# Analyzing and Evaluating Faithfulness in Dialogue Summarization (EMNLP 2022)

For more details, please find our paper on arXiv [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2210.11777).

[(Poster)](poster.pdf)

## Human Evaluation Results

- [Results and descriptions](https://github.com/BinWang28/FacEval/tree/main/human_result)
- CSV format + 150 dialogues + 5 candidates per dialogue + 6 error types

## Generate FacEval Dataset

We provide our generated FacEval dataset and also the source code to automatically generate them, which can be further improved and extended to other datasets.

We test the code with python 3.7 and below requirements.

```
pip install -r requirements.txt
```

The generated FacEval dataset is 'data/faceval_samples.json'.

Demo to generate factually corrupted samples:
```
python -m spacy download en_core_web_sm
bash data_preparation.sh
```

## Evaluation Demo on BART-Large Model

We provide the evaluation demo on BART-Large model. Because our proposed model-level evaluation needs the direct access to the model's generation probabilities. The model needs to be loaded as well for testing.

First, download the trained BART-Large model (.bin) from [Google Drive](https://drive.google.com/drive/folders/1XRpewVDUZwaQr8CVYCDd85Ob-VL0LbhL?usp=sharing) and place it in folder 'trained_model'.

Run the following bash to compute the score on different error types for BART-Large mdoel.
```
bash eval_demo.sh
```
The model score will be saved in folder 'scores_log'

## References

If you find our work useful, please consider citing our work.

```bibtex
@inproceedings{wang2022analyzing,
  title={Analyzing and Evaluating Faithfulness in Dialogue Summarization},
  author={Wang, Bin and Zhang, Chen and Zhang, Yan and Chen, Yiming and Li, Haizhou},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2022}
}
```

```bibtex
@article{wang2022analyzing,
  title={Analyzing and Evaluating Faithfulness in Dialogue Summarization},
  author={Wang, Bin and Zhang, Chen and Zhang, Yan and Chen, Yiming and Li, Haizhou},
  journal={arXiv preprint arXiv:2210.11777},
  year={2022}
}
```



Contact to Bin Wang at [bwang28c@gmail.com](mailto:bwang28c@gmail.com) for any issues.