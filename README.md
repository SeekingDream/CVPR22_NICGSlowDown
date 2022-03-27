# NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models

# Rebuttal Results
## Hyperparameter Sensibility
### Sensibility In terms of Iteration Number $T$
|Subject|L2| | |Linf| | |
|:----|:----|:----|:----|:----|:----|:----|
| |800|1000|1200|800|1000|1200|
|A|583.86 |583.86 |583.86 |449.05 |451.23 |465.93 |
|B|581.32 |581.32 |581.32 |578.67 |581.32 |581.32 |
|C|533.58 |533.58 |533.58 |480.99 |478.99 |482.12 |
|D|510.95 |513.77 |502.21 |226.55 |224.95 |220.83 |

### Sensibility In terms of Perturbation $e$
|Subject|30|40|50|0.02|0.03|0.04|
|:----|:----|:----|:----|:----|:----|:----|
|A|583.86 |583.86 |583.86 |185.69 |451.92 |573.70| 
|B|581.32 |581.32 |581.32 |578.67 |581.32 |581.32|
|C|533.58 |533.58 |533.58 |367.80 |481.75 |522.39|
|D|482.67 |514.15 |521.47 |170.13 |225.68 |302.42 |

## Adversarial Training Results

### $L_{inf}# Results
|I-Loop|A|B|C|D|
|:----|:----|:----|:----|:----|
|Before|354.11 |271.19 |379.81 |115.02|
|After (Train)|7.50 |2.34 |0.15 |0.73 |
|After (Test)|233.82 |150.31 |394.96 |106.55 |

### $L_{2}# Results

|I-Loop|A|B|C|D|
|:----|:----|:----|:----|:----|
|Before|483.8|481.32 |433.58 |408.90|
|After (Train)|10.50 |6.42 |0.31 |0.42|
|After (Test)|333.31 |432.31 |356.31 |300.31 |





## Description

NICGSlowDown is designed to generate  **efficiency adversarial examples** to evaluate the efficiency robustness of NICG models.
The generated adversarial examples are realistic and human-unnoticable images while consume more computational resources than benign images.



## Approach Ovewview
![](https://github.com/anonymousGithub2022/1/blob/main/fig/0001.jpg)

Our approach overview is shown in the above figure, for the detail design, please refer to our papers.


## File Structure
* **src** -main source codes.
  * **./src/model** -the model architecture of the NICG models.
  * **./src/attack** -the implementation of proposd attack algorithm.
* **train.py** -the script to train the NICG models.
* **generate_adv.py** -this script generate the adversarial examples.
* **test_latency.py** -this script measure the latency of the generated adversarial examples.
* **gpu4.sh** -bash script to generate adversarial examples and measure the efficiency.


## How to run
We provide the bash script that generate adversarial examples and measure the efficiency in **gpu4.sh**. **gpu5.sh**, **gpu6.sh**,**gpu7.sh** are implementing the similar functionality but for different gpus. 

 So just run `bash gpu4.sh`


## Efficiency Degradation Results
![](https://github.com/anonymousGithub2022/1/blob/main/fig/res.jpg)
The above figure shows the *efficiency* distribution of the benign images and the adversarial images. The ares under the cumulative distribution function (CDF) represent the victim NICG models' efficiency. A large area implies the model is less efficiency.


## Generated Adversarial Examples

![](https://github.com/anonymousGithub2022/1/blob/main/fig/0001%202.jpg)
The first row shows the benign images and the second row shows the generated efficiency adversarial images.







