<div align="center">
  <h1>fedorAop: A Tool to solve distribution shift</h1>
  <h6 align="right">
    by presenting transfer learning models to you
  </h6>
</div>

<div>
  <br>
</div>

> **_Stage 0: Under development_**

- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Further Considerations](#further-considerations)
- [Keep in mind](#keep-in-mind)
- [Resources](#resources)

## Requirements

- [Python 3.12 +](https://www.python.org/downloads/)
- [git](https://git-scm.com/downloads)
- [requirements.txt](requirements.txt)

## Getting Started

- Fork this repo
- Make sure that the original **Datasets** folder is in the working directory
- Open your terminal

```bash
git clone https://github.com/{YOURGITHUBID}/fedorAop
cd fedorAop
ls  # or `dir` in windows, the `Datasets` folder should appear
pip install -r requirements.txt -U
pre-commit install
mkdir Models
```

- If you prefer a quick start

```bash
cd fedorAop
python3 main.py
```

- Make your changes, and then

```bash
black .
git add .
git commit -m "YOUR_MESSAGE_HERE"
git push
```

- Check the `doc` folder for further information

## Further Considerations

- Intending to use simpler machine learning models (e.g., regularized linear models) for well-performing datasets and more complex neural network architectures (e.g., incorporating CNN layers) for poorly performing datasets.
- Utilize the cost function $ ||Xw - y||^2_2 + \frac \lambda 2 ||w-w_0||^2_2 $ and employing the Adam optimizer with Learning Rate Scheduling for gradient descent training of the linear model. This aims to make the weight vector $w$ similar to the result of the previous training in each iteration.
- Utilize imbalanced-learn to resample
- Yeo-Johnson Transform (_Preprocessing_)
- Online Machine Learning, e.g. deep-river
- MLP-Mixer

## Keep in mind

> > **No classifier** will work well on **all distributions**

## Resources

……
