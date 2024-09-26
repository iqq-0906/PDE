isamp = 3000
t = np.arange(0,1,0.01)
pps = 20
tsamp = pps*isamp

data_train = spi.loadmat('AD_5000_DP_TrData.mat')

u_in = data_train['u_in'][0:tsamp,:]
x_t_in = data_train['x_t_in'][0:tsamp,:]
s_in = data_train['s_in'][0:tsamp,:]

max_u = np.max(u_in)
min_u = np.min(u_in)

u_in = (u_in-min_u)/(max_u-min_u)

max_t = np.max(x_t_in)
min_t = np.min(x_t_in)

x_t_in = (x_t_in-min_t)/(max_t-min_t)

max_s = np.max(s_in)
min_s = np.min(s_in)

s_in = (s_in-min_s)/(max_s-min_s)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Dense

bs = tsamp

def fn(x):
    y = tf.einsum("ij, ij->i", x[0], x[1])
    y = tf.expand_dims(y, axis = 1)
    return y

tfd = tfp.distributions
tfb = tfp.bijectors

def normal_sp(params):
    return tfd.Normal(loc = params[:, 0:1], scale = 0.001+tf.math.softplus(params[:, 1:2]))

def negloglikelihood(y_true, y_pred):
    return tf.keras.backend.sum(-y_pred.log_prob(y_true))+(sum(model.losses)/bs)

hln = 30

inputsB = Input(shape = (100,), name = 'inputsB')
hiddenB = tfp.layers.DenseFlipout(hln, activation = "relu")(inputsB)
hiddenB = tfp.layers.DenseFlipout(hln, activation = "relu")(hiddenB)
hiddenB = tfp.layers.DenseFlipout(hln, activation = "relu")(hiddenB)

inputsT = Input(shape = (1,), name = 'inputsT')
hiddenT = tfp.layers.DenseFlipout(hln, activation = "relu")(inputsT)
hiddenT = tfp.layers.DenseFlipout(hln, activation = "relu")(hiddenT)
hiddenT = tfp.layers.DenseFlipout(hln, activation = "relu")(hiddenT)

combined = Lambda(fn, output_shape = [None, 1])([hiddenB, hiddenT])
output = tfp.layers.DenseFlipout(2)(combined)

dist = tfp.layers.DistributionLambda(normal_sp)(output)
model = Model(inputs = [inputsB, inputsT], outputs = dist)

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
spe = 25
string = './model/model_s1/'+str(spe)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss_value = 0
        for i in range(0,spe):
            logits = model({"inputsB":u_in, "inputsT":x_t_in}, training=True)
            loss_value = loss_value + negloglikelihood(s_in, logits)
        loss_value = loss_value*(1/spe)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

epochs = 10000
loss = np.zeros(epochs)

for epoch in range(epochs):
    loss_value = train_step()
    loss[epoch] = loss_value.numpy()
    if loss[epoch] <= np.min(loss[0:epoch+1]):
        model.save_weights(string)
        last_saved_wt = epoch
    if epoch%10 == 0:
        print("Epoch %d, loss %.2f" % (epoch, loss[epoch]))

print(last_saved_wt)
t = np.arange(0,1,0.01)
testdata = spi.loadmat('AD_TestData.mat')

u_in_test = testdata['u_in_test']
x_t_in_test = testdata['x_t_in_test']
s_in_test = testdata['s_in_test']
u_in_test = (u_in_test-min_u)/(max_u-min_u)
x_t_in_test = (x_t_in_test-min_t)/(max_t-min_t)
s_in_test = (s_in_test-min_s)/(max_s-min_s)
nsamples = 100
nps = 1
pred = np.zeros([nsamples*nps,1000000])
for i in range(0,nsamples):
    if i%5 == 0:
        print(i)
    pred[nps*i:nps*(i+1),:] = np.squeeze((model({"inputsB":u_in_test, "inputsT":x_t_in_test})).sample(nps))

pred = (pred*(max_s-min_s))+min_s
s_in_test = (s_in_test*(max_s-min_s))+min_s

print()
print(np.mean((s_in_test-np.mean(pred, axis = 0)[..., np.newaxis])**2))
print(np.mean((s_in_test)**2))
print(np.mean((s_in_test-np.mean(pred, axis = 0)[..., np.newaxis])**2)/np.mean((s_in_test)**2))