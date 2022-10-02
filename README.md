# NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models

## Description

NICGSlowDown is designed to generate  **efficiency adversarial examples** to evaluate the efficiency robustness of NICG models.
The generated adversarial examples are realistic and human-unnoticable images while consume more computational resources than benign images.



## Approach Ovewview
![](https://github.com/SeekingDream/NICGSlowDown/blob/main/fig/0001.jpg)

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




## Citation

If you find this repository is helpful to you, please consider cite 
````
@inproceedings{chen2022nicgslowdown,
  title={NICGSlowDown: Evaluating the Efficiency Robustness of Neural Image Caption Generation Models},
  author={Chen, Simin and Song, Zihe and Haque, Mirazul and Liu, Cong and Yang, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15365--15374},
  year={2022}
}
````



