---
marp: true
theme: default
class: lead
paginate: true
footer: `RNA202202`
math: katex
style: |
  img { background-color: rgba(255, 255, 255, 0); }
  section.title {
    background-color: #333333;
    color: #EEEEEE;
  }
  section.title h1 {
    font-size: 50px;
    color: #EEEEEE;
    border-bottom: 1px solid #595959;
    margin-bottom: 1em;
  }
  section.title h2 {
    font-size: 30px;
    color: #EEEEEE;
    border-bottom: 1px solid #595959;
    margin-bottom: 1em;
  }
  section.title footer {
    font-size: 18px;
    color: #EEEEEE;
    opacity: 0.5;
  }
  section.text {
    background-color: #EEEEEE;
    color: #333333;
    font-size: 23px;
  }
  section.text h1 {
    font-size: 60px;
    color: #333333;
    border-bottom: 1px solid #595959;
    margin-bottom: 1em;
  }
  section.text h2 {
    font-size: 30px;
    color: #333333;
    border-bottom: 1px solid #595959;
    margin-bottom: 1em;
  }
  section.text footer {
    font-size: 18px;
    color: #333333;
    opacity: 0.5;
  }
  section.text h3 {
    font-size: 20px;
  }
  section.text h4 {
    font-size: 50px;
    font-weight: lighter;
    margin-top: -0.2em;
    margin-bottom: 0.5em;
  }
---

<!-- _class: title -->
<!-- _paginate: false -->

# Understanding class separability in the hidden layer projection of a binary classifier neural network in over-fitting situations

![bg auto left:40%](img/ufmg.jpg)

---

<!-- _class: title -->
<!-- _paginate: false -->

<style scoped>
ol {columns: 1;}
</style>

# Agenda

1. Context
1. Experiments
1. Results

---

<!-- _class: title -->
<!-- _paginate: false -->

# Context

----

<!-- _class: text -->

<style scoped>
.columns {
  display: grid;
  grid-template-columns: 1fr 100px 1fr;
}
</style>

### Context
#### Motivation

- Representation learning with neural networks is an important topic nowadays
- Deep architectures are a great tool for representation learning

![w:900](img/circles.png)

----

<!-- _class: text -->

<style scoped>
.columns {
  display: grid;
  grid-template-columns: 0.4fr 100px 1fr;
}
</style>

<div class="rect-bottom"></div>

<div class="columns">
<div>

### Context
#### Motivation

- Models that target only the objective metric are prone to over-fit

</div>
<div>

</div>
<div>

![center](img/representation.png)

</div>
</div>

---

<!-- _class: text -->

<style scoped>
p { font-size: 50px; }
</style>

### Context
#### Hypothesis

> **Large over-fit** effects lead to **large separability** between classes in the n-dimensional
**hidden-layer projection** space

---

<!-- _class: title -->
<!-- _paginate: false -->

# Experiments

---

<!-- _class: text -->

### Experiments
#### Model Architecture

- Over-fit induced Multi-layer Perceptron

<br>

![w:800](img/architecture.png)

---

<!-- _class: text -->

<style scoped>
.columns {
  display: grid;
  grid-template-columns: 1fr 100px 1fr;
}
</style>

### Experiments
#### Metrics

<div class="rect-bottom"></div>

<div class="columns">
<div>

## Separability

- Calinski-Harabasz Index
- Davies-Bouldin Index
- Silhouette Score
- Percentage of Negative Silhouette Observations
- Average Quality Index
- Percentage of Border Quality Observations

</div>
<div>

</div>
<div>

## Over-fit

- Regularization Parameter
- Train-test Accuracy Difference

</div>
</div>

---

<!-- _class: text -->

<style scoped>
.columns {
  display: grid;
  grid-template-columns: 1fr 100px 0.6fr;
}
</style>

### Experiments
#### Metrics

<div class="rect-bottom"></div>

<div class="columns">
<div>

## UCI Machine Learning Repository

- Statlog (Australian Credit Approval) Data Set
- Statlog (German Credit Data) Data Set
- Breast Cancer Coimbra Data Set
- Connectionist Bench (Sonar, Mines vs. Rocks) Data Set
- Statlog (Heart) Data Set

</div>
<div>

</div>
<div>

## Synthetic Data Sets

- Moons
- Linearly Separable Blobs
- XOR

</div>
</div>

---

<!-- _class: title -->
<!-- _paginate: false -->

# Results

---

<!-- _class: text -->

<style scoped>
.columns {
  display: grid;
  grid-template-columns: 1fr 100px 0.65fr;
}
</style>

### Results
#### Expected

<div class="rect-bottom"></div>

<div class="columns">
<div>

- By evaluating the Pearson Correlation between separability and over-fit metrics a certain behavior is expected based on the hypothesis

</div>
<div>

</div>
<div>

|              | Regularization | Accuracy Diff |
|--------------|:--------------:|:-------------:|
| CH Index     |        -       |       +       |
| DB Index     |        +       |       -       |
| Perc. Border |        +       |       -       |
| Avg Q Index  |        -       |       +       |
| Perc. NegSil |        +       |       -       |
| Sil Score    |        -       |       +       |

</div>
</div>

---

<!-- _class: text -->

<style scoped>
table {
  font-size: 20px;
}
</style>

### Results
#### Evaluation

| **Regularization Paremeter** | CH Index | DB Index | Perc. Border | Avg Q Index | Perc SilNeg | Sil Score |
|------------------------------|---------:|---------:|-------------:|------------:|------------:|----------:|
| breast coimbra               |   -0.595 |    0.766 |        0.712 |      -0.700 |       0.746 |    -0.695 |
| cred aus                     |   -0.049 |   -0.075 |        0.883 |      -0.870 |       0.855 |    -0.877 |
| cred ger                     |   -0.417 |   -0.149 |        0.812 |      -0.746 |       0.804 |    -0.752 |
| heart                        |   -0.687 |    0.933 |        0.913 |      -0.809 |       0.769 |    -0.874 |
| sonar                        |   -0.646 |    0.942 |        0.888 |      -0.916 |       0.917 |    -0.912 |
| xor                          |    0.447 |   -0.342 |        0.342 |      -0.345 |       0.556 |    -0.462 |
| blobs                        |   -0.083 |   -0.073 |        0.887 |      -0.884 |       0.826 |    -0.893 |
| moons                        |    0.345 |   -0.325 |        0.826 |      -0.789 |       0.808 |    -0.816 |

---

<!-- _class: text -->

<style scoped>
table {
  font-size: 20px;
}
</style>

### Results
#### Evaluation

| **Accuracy Difference** | CH Index | DB Index | Perc. Border | Avg Q Index | Perc SilNeg | Sil Score |
|-------------------------|---------:|---------:|-------------:|------------:|------------:|----------:|
| breast coimbra          |    0.861 |   -0.819 |       -0.846 |       0.884 |      -0.886 |     0.900 |
| cred aus                |    0.601 |   -0.133 |       -0.736 |       0.865 |      -0.702 |     0.832 |
| cred ger                |    0.799 |   -0.297 |       -0.879 |       0.978 |      -0.854 |     0.970 |
| heart                   |    0.860 |   -0.733 |       -0.689 |       0.898 |      -0.915 |     0.880 |
| sonar                   |    0.595 |   -0.765 |       -0.811 |       0.825 |      -0.816 |     0.820 |
| xor                     |   -0.629 |    0.604 |       -0.554 |       0.574 |      -0.564 |     0.250 |
| blobs                   |    0.015 |    0.080 |       -0.322 |       0.326 |      -0.302 |     0.319 |
| moons                   |   -0.379 |    0.229 |       -0.329 |       0.243 |      -0.303 |     0.242 |

---


<!-- _class: text -->

<style scoped>
.columns {
  display: grid;
  grid-template-columns: 0.6fr 50px 1fr;
}
</style>

<div class="rect-bottom"></div>

<div class="columns">
<div>

### Results
#### Visual Example

- Australian Credit Data Set
- Hidden Layer Sizes:
`256, 128, 64, 32, 16, 8, 2`
- Over-fit Model:
  - `reg_alpha=1e0`
  - Accuracy Difference: 10%
- Right-fit Model:
  - `reg_alpha=1e1`
  - Accuracy Difference: 5%

</div>
<div>

</div>
<div>

![](img/proj-overfit.png)
*Over-fit model hidden-layer projection*

<br>

![](img/proj-rightfit.png)
*Right-fit model hidden-layer projection*

</div>
</div>

---

<!-- _class: text -->

<style scoped>
.columns {
  display: grid;
  grid-template-columns: 1fr 100px 0.65fr;
}
</style>

### Results
#### Conclusion

- Valid confirmation of the **expected behavior**
- Certain metrics are **heavily influenced by convexity** of the clusters
- **Synthetic datasets** portray a harder challenge for the model to perform as expected
- The **relationship between the observations**, their **topology** and their **arrangement in the hidden-layer projection space** could motivate a **new methodology** for regularized neural network models

---

<!-- _class: title -->

<style scoped>
section { background-color: #000 }
footer { opacity: 0 }
img {
  -webkit-filter: invert(1);
  filter: invert(1);
}
p {
  color: #000;
  margin-top: -1em;
}
</style>

# Gustavo Vieira Maia

[![invert w:32](https://cdn-icons-png.flaticon.com/512/25/25236.png)](mailto:vmgustavo@ufmg.br) - [![invert w:32](https://cdn-icons-png.flaticon.com/512/733/733609.png)](https://github.com/vmgustavo) - [![invert w:32](https://cdn-icons-png.flaticon.com/512/25/25325.png)](https://www.linkedin.com/in/vmgustavo/)
