import math 
import random
import sys

matrix = [[1,2, 3], [4, 5, 6], [7, 8 ,9]]
p = [[4, 2, 8], [10, 12, 4], [4, 5, 9]]

y = [1, 6, 2 ,3 , 9, 7, 8, 4, 5, 10, 78 ,96, 45 , 32, 98 , 71, 4, 32, 85, 43, 14, 65 ,12, 4, 78 ,633 ,15, 824, 966, 74, 56, 14, 78, 96, 48]


def matrix_transpon(matrix):
  for k in range(len(matrix)):
    assert len(matrix[0]) == len(matrix[k])
  
  nova_matice = []
  sloupce = len(matrix[0])
  radky = len(matrix)
  for n in range(sloupce):
    nova_matice.append([])
    for s in range(radky):
      nova_matice[n].append([])
    
  for i in range(radky):
    for x in range(sloupce):
      radek = matrix[i]
      num = radek[x]
      nove_cislo = nova_matice[x]
      supa_nove_cislo = nove_cislo[i]
      supa_nove_cislo.append(num)

  supa_nova_matice = []

  for a in range(sloupce):
    supa_nova_matice.append([])

  for q in range(sloupce):
    radek_n = nova_matice[q]
    for l in range(radky):
      skoro = radek_n[l][0]
      supa_nova_matice[q].append(skoro)      

  for t in range(sloupce):
    print(supa_nova_matice[t])

  return None



def nasobeni_matice(matrix, snd_matrix):
  new_matrics = []
  for t in range(len(matrix)):
    new_matrics.append([])
  
  assert len(matrix[0]) == len(snd_matrix)
  for i in range(len(matrix)):
    s = 0
    for x in range(len(snd_matrix[0])):
      for l in range(len(matrix[0])):
        k = matrix[i][l] * snd_matrix[l][x]
        s += k

      new_matrics[i].append(s)
      s = 0

  return new_matrics



def scitani_matic(matrix, snd_matrix):
  assert len(matrix) == len(snd_matrix)
  assert len(matrix[0]) == len(snd_matrix[0])
  
  new_matrix = []
  for m in range(len(matrix)):
    new_matrix.append([])

  for i in range(len(matrix)):
    for x in range(len(matrix[0])):
      num = matrix[i][x] + snd_matrix[i][x]
      new_matrix[i].append(num)

      num = 0

  return new_matrix


def rozptyl(y):
  all_nums = []
  k = sum(y) / len(y)
  for i in(y):
    rozdil = i - k 
    num = rozdil * rozdil
    all_nums.append(num)

  rozptyls = sum(all_nums) / len(all_nums)
  
  return rozptyls


def smerodatna_odchylka(y):
  
  smerodatna_odchylka = math.sqrt(rozptyl(y))

  return smerodatna_odchylka



def sorting_numbers_algorithm(num_list, mode):
  new_list = []
  if mode.lower() == "fromhigher":
    for q in range(len(num_list)):
      for i in num_list:
        n = i 
        for x in num_list:
          if n > x:
            pass

          else:
            n = x

        new_list.append(n)
    
        num_list.remove(n)
    
    return new_list
    
  else:
    for z in range(len(num_list)):
      for i in range(len(num_list) - 1):
        w = num_list[i]
        b = num_list[i + 1]
        if b < w:
          num_list[i] = b
          num_list[i + 1] = w
          
    return num_list

k = sorting_numbers_algorithm(y, "fromhigher")


sup_text = ["Hello how are you", "what are you doing man"]

class one_hot_encoding:
  def __init__(self, training_text, text):
    self.training_text = training_text
    self.text = text

  def one_hot_encoding_of_text_words(self):
    self.words = []
    for i in self.training_text:
      for x in i.lower().split(" "):
        if x in self.words:
          pass

        else:
          self.words.append(x)

    return self.words
  
  def one_hot_encoding_vectors(self, words):
    self.words = words
    self.matrix = []
    for i in range(len(self.text)):
      self.matrix.append([])

    for s, x in enumerate(self.text):
      for y in range(len(x.split(" "))):
        self.matrix[s].append([])

    for i in range(len(self.text)):
      for x, q in enumerate(self.text[i].lower().split(" ")):
        k = []
        for h in range(len(self.words)):
          k.append(0)

        c = self.words.index(q)

        k[c] = 1
        
        for a in k:
          self.matrix[i][x].append(a)

    return self.matrix


def BoW(training_text, text):
  vocabulary = []
  matrix = []
  for i in training_text:
    for x in i.lower().split(" "):
      if x not in vocabulary:
        vocabulary.append(x)

      else:
        pass
  
  for y in range(len(text)):
    matrix.append([])

  for i, x in enumerate(text):
    k = []
    for q in range(len(vocabulary)):
      k.append(0)
    
    for y in x.lower().split(" "):
      s = vocabulary.index(y)
      k[s] += 1

    for a in k:
      matrix[i].append(a)

  return matrix




def TF_IDF(text):
  words = []
  for i in text:
    i = i.lower()
    for x in i.split(" "):
      if x not in words:
        words.append(x)
 
  k = []
  for i in text:
    i = i.lower()
    for x in i.split(" "):
      k.append(x)
  
  tf = dict()
  for i in words:
    c = 0
    for x in k:
      if x == i:
        c += 1
      
      else:
        pass
    
    num = c / len(k)

    tf[i] = num
  
  
  idf = dict()
  for i in words:
    q = 0
    for x in text:
      x = x.lower()
      if i in x:
        q += 1 

      else:
        pass
  
    b = len(text) / q

    idf[i] = math.log2(b)

  tfidf = dict()

  for i in words:
    fin_num = tf[i] * idf[i]
    tfidf[i] = fin_num

  matrix = []
  for i in range(len(text)):
    matrix.append([])

  for x in matrix:
    for y in range(len(words)):
      x.append(0)
  
  for r, i in enumerate(text):
    i = i.lower()
    for x in i.split(" "):
      u = list(tfidf).index(x)
      matrix[r][u] = tfidf[x]


  return matrix

y = [1, 1 , 1, 0, 0, 0, 0]
test = ["fuck bitch", "hello how are you my guy", "Hello fuck you bitch", "hello bro"]

text = ["Fuck yeah bitch hello my friend", "how are you bro", "hello man", "what going on man", "Fuck you are bitch", "You little bitch how", "hello I hate you"]

def naive_bayes(text, labels, test_text):
  y = []
  for i in labels:
    if i not in y:
      y.append(i)

  k = []

  for c in range(len(y)):
    k.append([])

  for s, i in enumerate(y):
    for x in range(len(labels)):
      if labels[x] == i:
        k[s].append(text[x].lower())
  
  prob = []
  
  for i in range(len(k)):
    prob.append(dict())
      
  for i in range(len(k)):
    words = []
    for x in k[i]:
      for c in x.split(" "):
        words.append(c)

    for t in words:
      num = 0
      for q in words:
        if t == q:
          num += 1

      supa_num = num / len(words)

      prob[i][t] = supa_num

  int_guess = []

  for i in range(len(k)):
    int_guess.append(len(k[i]) / len(text))
 
  predictions = []
  
  add_num = 0.1
  for i in range(len(k)):
    for x in prob[i]:
      prob[i][x] += add_num

  for i in test_text:
    i = i.lower()
    fin_probs = []
    snd_probs = []
    for x in range(len(k)):
      p = int_guess[x]
      for v in i.split(" "):
        try:
          p = p * prob[x][v]

        except:
          p = p * add_num

      fin_probs.append(p)
      snd_probs.append(p)
    
    fin = fin_probs
    pred = sorting_numbers_algorithm(fin, "fromhigher")

    q = pred[0]
    t = snd_probs.index(q)
    predictions.append(y[t])

  return predictions
    

weight = [0.8, 0.9, 0.75, 1, 0.65]
height = [1.84, 1.9, 1.8, 1.92, 1.7]
lab = [1, 1, 0, 1, 0 ,1 ,0]
test_x = [90, 50, 100]
test_y = [180, 200, 200]

def K_nearestneighbors(x, y, labels,test_x, test_y, K=3):
  train = dict()
  for i in range(len(x)):
    train[x[i]] = y[i]

  test_data = dict()
  for x in range(len(test_x)):
    test_data[test_x[x]] = test_y[x]

  pravda = []

  for s, i in test_data.items():
    pure_labels = dict()
    for i in labels:
      if i not in pure_labels:
        pure_labels[i] = 0
    rozdily = []
    for x, u in train.items():
      s = abs(s)
      i = abs(i)
      x = abs(x)
      u = abs(u) 
      x_rozdil = abs(x - s)
      y_rozdil = abs(u - i)
      rozdil = x_rozdil**2 + y_rozdil**2
      rozdil = math.sqrt(rozdil)
      rozdily.append(rozdil)
  
    true_rozdily = []
    for x in rozdily:
      true_rozdily.append(x) 
  
    k_rozdily = sorting_numbers_algorithm(rozdily, "fromlower")
    k_rozdily = k_rozdily[0:K]

    test_labels = []

    for i in k_rozdily:
      c = true_rozdily.index(i)
      test_labels.append(labels[c])

    for i, s in pure_labels.items():
      for x in test_labels:
        if i == x:
          pure_labels[i] += 1
        else:
          pass

    another_labels = []

    for i in pure_labels.values():
      another_labels.append(i)
      
    another_labels = sorting_numbers_algorithm(another_labels, "fromhigher")
    d = another_labels[0]
    num = list(pure_labels.values()).index(d)
    supa_num = list(pure_labels.keys())[num]
    pravda.append(supa_num)

  return pravda



def K_means_clustering(x, y, K=2, iterations = 10):
  points = dict()
  for i in range(len(x)):
    points[x[i]] = y[i]

  labels = []
  c = 0
  for i in range(K):
    labels.append(c)
    c += 1
  
  clust_points = dict()
  for i in range(K):
    random_xclust = random.randint(0, 100)
    random_yclust = random.randint(0, 200)
    clust_points[random_xclust] = random_yclust

  for i in range(iterations):
    print(clust_points)
    clusters = []
    for x in range(len(labels)):
      clusters.append(dict())
    for s, i in points.items():
      vzdalenost_supa = []
      for m, t in clust_points.items():
        s = abs(s)
        i = abs(i)
        m = abs(m)
        t = abs(t)
        rozdil_x = abs(s - m)
        rozdil_y = abs(i - t)
        skoro_vzdalenost = rozdil_x**2 + rozdil_y **2
        vzdalenost = math.sqrt(skoro_vzdalenost)
        vzdalenost_supa.append(vzdalenost)
      
      dalsi_vzdalenost = []
      for q in vzdalenost_supa:
        dalsi_vzdalenost.append(q)

      vzdalenost_supa = sorting_numbers_algorithm(vzdalenost_supa, "fromlower")
      vzdalenost_supa = vzdalenost_supa[0]
      d = dalsi_vzdalenost.index(vzdalenost_supa)
      clusters[d][s] = i
      
    clust_points = {}

    for i in clusters:
      x_avg = []
      y_avg = []
      for w, e in i.items():
        x_avg.append(w)
        y_avg.append(e)
    
      try: 
        x_avg = sum(x_avg) / len(x_avg)
        y_avg = sum(y_avg) / len(y_avg)

        clust_points[x_avg] = y_avg

      except:
        pass


  return clusters




chest_pain = [0, 1, 1, 1, 0, 1, 0, 1]
blocked_athretis = [0, 1, 0, 1, 1, 1, 0, 1]
labels= [0, 1, 0, 0, 1, 1, 0, 1]

class decision_tree:
  def yes_no(*args, labels):
    
    q = len(args)
    
    nodes = []

    for m in range(len(args)):
      gini_imp = []
      for i in range(len(args)):
        gini_imp.append([])
      
      yes_no = []
      for i in range(len(args)):
        yes_no.append([])
  
      
      for i in range(len(gini_imp)):
        yes_t = 0
        no_t = 0
        yes_f = 0
        no_f = 0
        for x in range(len(args[i])):
          if args[i][x] == 1 and labels[x] == 1:
            yes_t += 1
        
          elif args[i][x] == 1 and labels[x] == 0:
            no_t += 1
        
          elif args[i][x] == 0 and labels[x] == 1:
            yes_f += 1

          elif args[i][x] == 0 and labels[x] == 0:
            no_f += 1

        has = 1 - (yes_t/(yes_t+no_t))**2  - (no_t/(yes_t+no_t))**2
        d_has = 1 - (yes_f/(yes_f+no_f)) - (no_f/(yes_f+no_f))
        total = ((yes_t+no_t)/len(labels)) * has + ((yes_f + no_f) / len(labels)) * d_has
        gini_imp[i].append(total)
       
        yes_no[i].append(yes_t)
        yes_no[i].append(no_t)
        yes_no[i].append(yes_f)
        yes_no[i].append(no_f)
      
      supa_gini = []
      for x in gini_imp:
        supa_gini.append(x[0])
    
      gini = sorting_numbers_algorithm(gini_imp, "fromlower")
      gini = gini[0]
      print(gini)
      l = supa_gini.index(gini[0])
      nodes.append(yes_no[l])
      


    return nodes

test_weigt = [0.8]

def linearregresion(x, y, test, lr=0.01):
  d_e = []
  a = 0.5
  b = 0
  
  for m in range(1000):
    d_inter = []
    d_slope = []
    for i in range(len(x)):
      eq = (-2*x[i])*(y[i] - (a * x[i] + b)) 
      der = -2*(y[i] - (a * x[i] + b))

      d_inter.append(der)
      d_slope.append(eq)
      

    step_size_inter = sum(d_inter) * lr
    step_size_slope = sum(d_slope) * lr
    a = a - (step_size_slope)
    b = b - (step_size_inter)
    d_e.append(step_size_inter)

  preds = []
  for q in range(len(test)):
    pred = a * test[q] + b
    preds.append(pred)

  return preds

data_x = [0.1, 1, 1.5, 0.05, 0.75, 0.07, 0.21, 1.15, 0.14]
data_y = [0, 1, 1, 0, 1, 0, 0, 1, 0]
test_log = [90]

def logisticregresion(x, y, test, lr=0.0001):
  assert len(x) == len(y)
  a = 1
  b = 1
  for q in range(1000):
    d_a = []
    d_b = []
    for i in range(len(x)):
      if y[i] == 1:
        du = math.exp((a*x[i])+b) + 1 
        der_a = x[i] / du
        der_b = 1 / du
        d_a.append(der_a)
        d_b.append(der_b)

      else:
        v = (math.exp(a*x[i] + b) + 1)
        dera_a = x[i] * math.exp(a*x[i] + b) / v
        dera_b = math.exp(a*x[i] + b) / v

        d_a.append(dera_a)
        d_b.append(dera_b)
    
    l = sum(d_a) * lr
    k = sum(d_b) * lr
    a = a - l
    b = b - k

  predsoslav = []
  for w in range(len(test)):
    s = (test[w] * a) +b
    print(s)
    u = 1 + (math.exp(-(s)))
    asab = 1 / u
    predsoslav.append(asab)
  
  for z in predsoslav:
    if z > 0.5:
      print(1)

    else:
      print(0)

    
svm_d = [0.24, 0.3, 0.14, 0.9, 1, 0.15, 1.4, 0.85]
svd_y = [0.35, 0.4, 0.3, 0.93, 1.2, 0.36, 1.7, 0.9]
lob = [0, 0, 0, 1, 1, 0, 1, 1]
tosto = [0.74, 0.3, 1.5]
tosto_e = [1, 0.25, 1]


class SVM:
  def one_D(x, y, test):
    ones = []
    zeros = []
    for s in range(len(x)):
      if y[s] == 1:
        ones.append(x[s])
      
      else:
        zeros.append(x[s])

    rozdily = []
    for q in ones:
      for w in zeros:
        rozdily.append(abs(q-w))
    
    supa_rozdily = []
    a = 0
    b = len(ones)
    while b < len(rozdily)+1:
      supa_rozdily.append(rozdily[a:b])
      a += len(ones)
      b += len(ones)
    
    lowest = []
    
    for i in supa_rozdily:
      p = sorting_numbers_algorithm(i, "fromlower")
      lowest.append(p[0])
    
    lowest_ = []

    for i in lowest:
      lowest_.append(i)

    lowest = sorting_numbers_algorithm(lowest, "fromlower")
    
    t = lowest_.index(lowest[0])
    
    supa_rozdily_ = []

    a = 0
    b = len(ones)
    while b < len(rozdily)+1:
      supa_rozdily_.append(rozdily[a:b])
      a += len(ones)
      b += len(ones)

    n = supa_rozdily_[t].index(lowest[0])

    r = ones[t] + zeros[n]
    
    border = r / 2
     

    if ones[t] > zeros[n]:
      onesarebigger = True

    else:
      onesarebigger = False
    
    preds = []

    for i in test:
      if onesarebigger:
        if i > border:
          preds.append(1)

        else:
          preds.append(0)

      else:
        if i > border:
          preds.append(0)

        else:
          preds.append(1)

    return preds

  def two_D(x, y, labels, test_x, test_y):
    a = 3
    b = 2
    c = 5
    lr = 0.01
    x_ones = []
    x_zeros = []
    for i in range(len(x)):
      if labels[i] == 1:
        x_ones.append(x[i])

      else:
        x_zeros.append(x[i])
    

    if sum(x_ones) > sum(x_zeros):
      ones = "higher"
      zeros = "lower"

    else:
      zeros = "higher"
      ones = "lower"

    for m in range(10000):
      point = random.randint(0, len(x)-1)
      if ones == "higher":
        if labels[point] == 0 and (a*x[point]) + (b*y[point]) + c > 0:
          a = a - (lr * x[point])
          b = b - (lr * y[point])
          c = c - lr

        elif labels[point] == 1 and 0 > (a*x[point]) + (b*y[point]) + c:
          a = a + (lr * x[point])
          b = b + (lr * y[point])
          c = c + lr

        else:
          pass

      elif ones == "lower":
        if labels[point] == 1 and (a*x[point]) + (b*y[point]) + c > 0:
          a = a - (lr * x[point])
          b = b - (lr * y[point])
          c = c - lr

        elif labels[point] == 0 and 0 > (a*x[point]) + (b*y[point]) + c:
          a = a + (lr * x[point])
          b = b + (lr * y[point])
          c = c + lr

        else:
          pass
   
    a = a * 0.99
    b = b * 0.99
    c = c*0.99

    preds = []
   
    for i in range(len(test_x)):
      if ones == "higher":
        if ((a*test_x[i]) + (b*test_y[i]) + c) > 0:
          preds.append(1)

        else:
          preds.append(0)

      else:
        if ((a*test_x[i]) + (b*test_y[i]) + c) > 0:
          preds.append(0)

        else:
          preds.append(1)

    return preds


print(SVM.two_D(svm_d, svd_y, lob, tosto, tosto_e))


def simleneuralnetwork(x, y, test):
  w1 = 2
  w2 = 3
  w3 = 1.2
  w4 = 0.69
  b1 = 0
  b2 = 0
  b3 = 0
  lr = 0.001
  for i in range(20000):
    print(f"{i+1}: iteration")
    w1_d = []
    w2_d = []
    w3_d = []
    w4_d = []
    b1_d = []
    b2_d = []
    b3_d = []
    for q in range(len(x)):
      try_pred = x[q] * w1  +b1
      try_pred2 = math.log(1 + 2.7171**try_pred) * w3
      try_pred3 = x[q] * w2  +b2
      try_pred4 = math.log(1 + 2.7171**try_pred3) * w4
      fin = try_pred2 + try_pred4 + b3
      main_equation = -2 * (y[q] - fin)
      function_der1 = 2.7171 ** (x[q] * w1  +b1)  / (1 + (2.7171 ** (x[q] * w1  +b1)))
      function_der2 = 2.7171 ** (x[q] * w2  +b2)  / (1 + (2.7171 ** (x[q] * w2  +b2)))
      w1_de = main_equation * w3 * function_der1 * x[q]
      w1_d.append(w1_de)
      w2_de = main_equation * w4 * function_der2 * x[q]
      w2_d.append(w2_de)
      w3_de = main_equation * math.log(1 + 2.7171**try_pred)
      w3_d.append(w3_de)
      w4_de = main_equation * math.log(1 + 2.7171**try_pred2)
      w4_d.append(w4_de)
      b1_de = main_equation * w3 * function_der1
      b1_d.append(b1_de)
      b2_de = main_equation *  w4 * function_der2
      b2_d.append(b2_de)
      b3_de = main_equation
      b3_d.append(b3_de)
    
    w1 = w1 - (sum(w1_d) * lr)
    w2 = w2 - (sum(w2_d) * lr)
    w3 = w3 - (sum(w3_d) * lr)
    w4 = w4 - (sum(w4_d) * lr)
    b1 = b1 - (sum(b1_d) * lr)
    b2 = b2 - (sum(b2_d) * lr)
    b3 = b3 - (sum(b3_d) * lr)
  
  preds = []

  for i in test:
    try_pred = i * w1  +b1
    try_pred2 = math.log(1 + 2.7171**try_pred) * w3
    try_pred3 = i * w2  +b2
    try_pred4 = math.log(1 + 2.7171**try_pred3) * w4
    fin = try_pred2 + try_pred4 + b3
    preds.append(fin)

  return preds


def fibonacci_sequence(max_num):
  a = 0
  b = 1
  sequence = [0, 1]
  while b < max_num:
    c=a+b
    a = b
    b = c
    sequence.append(b)

  return sequence, b/a


    
  

    


      
      

  


      
      
        
    
