import tensorflow as tf


class MyLinerRegression(object):
    def __init__(self):
        self.learning_rate = 0.1

    def get_weight(self, shape):
        # 初始化权重
        with tf.variable_scope('get_weight'):
            weight = tf.Variable(
                initial_value=tf.random_normal(
                    shape=shape,
                    mean=0.0,
                    stddev=1.0
                ),
                name='weight'
            )
        return weight

    def get_bias(self, shape):
        # 初始化偏执
        with tf.variable_scope('get_bias'):
            bias = tf.Variable(
                initial_value=tf.random_normal(
                    shape=shape,
                    mean=0.0,
                    stddev=1.0
                ),
                name='bias'
            )

        return bias

    def build_data(self, shape):
        with tf.variable_scope('build_data'):
            x = tf.random_normal(
                shape=shape,
                mean=0.0,
                stddev=1.0,
                name='x'
            )

            y = tf.matmul(x, [[0.7]]) + 0.8

        return x, y

    def liner_model(self, x):
        with tf.variable_scope('liner_model'):
            self.weight = self.get_bias(shape=(x.shape[-1].value, 1))
            self.bias = self.get_bias(shape=())

            y_predict = tf.matmul(x, self.weight) + self.bias

        return y_predict

    def losses(self, y_true, y_pred):
        with tf.variable_scope('losses'):
            loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return loss

    def sgd(self, loss):
        with tf.variable_scope('sgd'):
            # tf.train.GradientDescentOptimizer sgd随机梯度下降优化算法
            sgd = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            # 优化均方差损失 -- 损失减小的方向
            train_op = sgd.minimize(loss)
        return train_op

    def train(self):
        # 构建收据
        x, y = self.build_data(shape=(100, 1))

        # 构建线性模型
        y_predict = self.liner_model(x)

        # 计算均方误差损失
        loss = self.losses(y, y_predict)

        # 指定sgd优化算法 优化loss
        train_op = self.sgd(loss)

        # 开启会话 -- 执行run--train_op
        with tf.Session() as ss:
            ss.run(tf.global_variables_initializer())
            # 序列化 events
            tf.summary.FileWriter('./tmp/', graph=ss.graph)
            # tensorboard --logdir ./tmp --host 127.0.0.1
            for i in range(600):
                ss.run(train_op)
                ret = '第{}次的损失{}，权重为{}，偏置为{}'.format(i+1, loss.eval(), self.weight.eval(), self.bias.eval())
                print(ret)


if __name__ == '__main__':
    lr = MyLinerRegression()
    lr.train()
