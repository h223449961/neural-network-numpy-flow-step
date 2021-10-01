'''
隨機地初始化權重
'''
wg = 2 * np.random.random((4, 1)) - 1
def sigmoid(x):
  return (1 + np.tanh(.5 * x))
def sigmoidd(x):
  return (x * (1 - x))
def think(x):
  output = sigmoid(np.dot((x).astype('float'),(wg)))
  return output

firstprb = think(num)
sigmd = sigmoidd(think(num))
print(sigmd)
'''
誤差
'''
abseror = (ranut-firstprb)
print(abseror*sigmd)
print(np.dot(num.T,abseror*sigmd))
'''
使用 iteration 來訓練
'''
for i in range(iteration):
  firstprb = think(num)
  abseror = (ranut-firstprb)
  sigmd = sigmoidd(think(num))
  adjust = np.dot(num.T,abseror*sigmd)
  wg = wg + adjust
prb = sigmoid(np.dot(num,wg).astype('float'))
print(softmax(prb))
