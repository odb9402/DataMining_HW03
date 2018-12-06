**Data Mining HW 3**
===================
MNIST classification with CNN
----

 주어진 소스 파일들을 이용해서, CNN 모델을 사용하여 MNIST 데이터를 분류하여, 0.9 이상의 **TEST** 정확도를 보인 후, 보고서를 작성하십시오. 단, mnist.py 와 mnist.ipynb 마지막에 주석 처리된 부분 이하는 수정하면 안됩니다. 
 
- 모델과 학습 횟수에 대하여 전혀 수정없이 **TEST** 정확도를 구할 경우 약 [0.86,0.89] 정도의 정확도를 보입니다.  mnist.py 에 구현된 모델은 총 2개층의 Convolution layer, 2개층의 Max pooling layer, 1개층의 Fully connected layer를 가집니다.

- 모델의 구조 및 [Hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) 를 변경하거나, 다른 [Gradient-based Optimization](http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html) 알고리즘을 사용하거나, 다른 추가적인 기법을 이용하여 개선해 볼 수 있습니다. 다만, 학습 횟수 (epoch) 를 조정하는 것은 본 과제에서는 허용하지 않습니다.

- 해당 모델을 개선하여 0.95 **이상** 의 **TEST** 정확도를 구할 경우 추가 점수(+0.5)를 얻을 수 있습니다. 이 경우, 보고서에 해당 내용을 기재해 주십시오.

- 해당 모델을 개선하여 0.98 **이상** 의 **TEST** 정확도를 구할 경우 추가 점수(+0.5)를 얻을 수 있습니다. 이 경우, 보고서에 해당 내용을 기재해 주십시오.

> 참고 - 


> [Tensorflow API](https://www.tensorflow.org/api_docs/python/tf)


> [Github repository of Tensorflow cookbook - CNN](https://github.com/nfmcclure/tensorflow_cookbook#ch-8-convolutional-neural-networks)
