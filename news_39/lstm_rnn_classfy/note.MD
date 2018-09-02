# 初步看来还行，按照字符级的CNN做。
- 参考 https://github.com/jakeywu/tf_classification
- 参考 http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
- 中间遇到的坑，nn、bp的精髓之一在于权重共享，所以在需要共享参数的时候用 `tf.get_variable` 比 `tf.Variable` 好，算是我在多轮计算中准确率提升不上去的一个教训。
- 数据混淆很重要，不然很容易陷入局部最优。数据准备很重要。
