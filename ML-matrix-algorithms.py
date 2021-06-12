import math 
import random
from randomprojects import lol
import sys
import pandas
import numpy

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

print(nasobeni_matice([[2, -3]], [[-1, 0], [0, -1]]))

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


  return matrix, tfidf

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
tosto = [0.74, 0.3, 1.5, 0.15]
tosto_e = [1, 0.25, 1, 0.21]


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


nn_x = [1.4, 1.5, 0.2, 0.3, 1, 1.2, 1.35, 0.28, 0.1, 0.05, 0.6, 0.5, 0.45, 0.52, 0.14, 1.3, 3, 4, 3.5, 2.9]
nn_y = [2, 2, 0, 0, 2, 2, 2, 0, 0, 0, 1, 1, 1, 1, 0, 2, 3, 3, 3, 3]
nn_test = [1.6, 0.12, 0.54, 3.3]

def simleneuralnetwork(x, y, test):
  w1 = 2
  w2 = 3
  w3 = 1.2
  w4 = 0.69
  b1 = 0
  b2 = 0
  b3 = 0
  lr = 0.001
  for i in range(50000):
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
    preds.append(round(fin))

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

datasa = [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]



def expected_value(data, bet=5, value= 1):
  values = []
  for x in data:
    if x not in values:
      values.append(x)

  real = []
  probs = []

  for x in range(len(values)):
    real.append([0])

  for r in data:
    real[r][0] += 1

  for x in real:
    c = x[0] / len(data)
    probs.append(c)

  final = []

  for x in range(len(values)):
    if x == value:
      f = probs[x] * bet
      final.append(f)

    else:
      f = probs[x] * (-bet)
      final.append(f)

  return sum(final)




def binary_search(data, num=410):
  r = data[math.floor(len(data)/2)]
  pred_num = 0 
  while r != num:
    first = len(data) / 2
    first =  math.floor(first)    
    if r > num:
      data = data[:first]

    else:
      data = data[first:]
    
    r = data[math.floor(len(data)/2)]
    
  print("Done: ", r)


p_value_test = [11100000, 1111111]

def simple_P_value(**args):
  for i in args:
    for x in i:
      main_list = []
      n = len(x)
      all_len = 2**n
      for w in range(n):
        main_list.append("0")
      main_num = ("").join(main_list)
      
      supa_main = []
      
      p = 0

      for q in range(all_len):
        supa_main.append(main_num)
        main_num[p] = 1
        
        p += 1

        
  
  return supa_main



def best_buy_sell(data, K=2):
  real_data = []
  for x in data:
    real_data.append(x)

  t = sorting_numbers_algorithm(data, "fromlower")
  
  ej = True
  
  while ej:

    lowest_highest = []
  
    lowest_highest.append(t[:K])
    lowest_highest.append(t[-K:])
  

    indexes = []
    for s in range(len(lowest_highest)):
      indexes.append([])

    for n, x in enumerate(lowest_highest):
      for y in x:
        indexes[n].append(real_data.index(y))

    true_indexes = []
    for x in indexes:
      true_indexes.append(sorting_numbers_algorithm(x, "fromlower"))

    for x in range(len(true_indexes[0])):
      if true_indexes[0][x] < true_indexes[1][x]:
        ej = False
     
      else:
        ej = True
        
        continue

  profit = []
  for x in range(len(true_indexes)):
    profit.append([])

  for s, m in enumerate(true_indexes):
    for i in m:
      profit[s].append(real_data[i])

  real_profit = 0

  for i, a in enumerate(profit):
    if i == 0:
      real_profit += -sum(a)

    else:
      real_profit += sum(a)

  print(true_indexes)
  for s in range(len(true_indexes[0])):
    print(f"Invest {real_data[true_indexes[0][s]]}: {true_indexes[0][s]+1} day, sell for {real_data[true_indexes[1][s]]}: {true_indexes[1][s]+1} day")
    

  return real_profit 

  

def Binary_Exponentiation(num, num2):
  if num2 == 0:
    return 1

  elif num2 == 1:
    return num

  else:
    if num2 % 2 == 0:
      c = num**(num2/2)
      c = c*c
      return c

    else:
      c = num**(num2/2)
      c = c*c*num
      return c


#class quantum_physics:
  


def Covariance(data_x, data_y):
  mean_x = sum(data_x) / len(data_x)
  mean_y = sum(data_y) / len(data_y)
  cov = []
  for i in range(len(data_x)):
    cov.append((data_x[i] - mean_x) * (data_y[i] - mean_y))

  covar = sum(cov) / (len(data_x))

  return(covar)

  
def correlation(data_x, data_y):
  c = Covariance(data_x, data_y)
  k = smerodatna_odchylka(data_x)
  q = smerodatna_odchylka(data_y)
  n = k*q
  fin = c/n
  return fin

def R_squared(data_x, data_y):
  return correlation(data_x, data_y)**2






def another_R_squared(lol, kol):
  k = linearregresion(lol, kol, lol)
  f = rozptyl(kol)
  fucku = []
  for t in range(len(kol)):
    g = kol[t] - k[t]
    fucku.append(g**2)

  ahojda = sum(fucku) / len(lol)

  finalejesmegusta = (f - ahojda) / f
  return finalejesmegusta



def F(x, y):
  k = linearregresion(x, y, x)
  mean = sum(y) / len(y)
  meansquared = []
  for i in y:
    meansquared.append((i-mean)**2)

  fucku = []
  for t in range(len(y)):
    g = y[t] - k[t]
    fucku.append(g**2)

  fin_eq = (sum(meansquared) - sum(fucku)) / (sum(fucku) /(len(x)-2))

  return fin_eq


  
  




def binomial_distribution(x, n, p=0.5):
  l= math.factorial(n) 
  k = math.factorial(x) * math.factorial(n-x) 
  c= l/k
  u = p**x
  c = c * u
  j = 1-p
  c = c * j**(n-x)
  print("Propability of that being random is :", c)
  return c


def gini_impurity(**args):
  pass
x1 = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1]
x2 = [130, 150, 70, 160, 100, 140, 64, 81, 120, 105, 95, 111, 66, 126, 75, 135, 84]
label = [1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1]
test_ada = [[1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1], []]


  



def AdaBoost(*args, y=[], test=[], k=10):
  amount_of_say = {}
  weights = []
  for i in range(len(args[0])):
    weights.append(1/len(args[0]))

  continuous = []

  for i in range(len(args)):
    continuous.append(0)
  
  for i in range(len(args)):
    for q in args[i]:
      if q > 1:
        continuous[i] = 1

      else:
        pass
  print(continuous)
  
  for x in range(k):
    best_num = []
    for t in range(len(continuous)):
      best_num.append([])
    
    gini_imp = []
    for i in range(len(continuous)):
      if continuous[i] == 0:
        base = [[0, 0], [0, 0]]
        for w in range(len(args[i])):
          if args[i][w] == 1 and y[w] == 1:
            base[0][0] += 1

          elif args[i][w] == 1 and y[w] == 0:
            base[0][1] += 1

          elif args[i][w] == 0 and y[w] == 1:
            base[1][0] += 1

          elif args[i][w] == 0 and y[w] == 0:
            base[1][1] += 1

          else:
            pass
        
        for u in base:
          for z in range(len(u)):
            u[z] += 0.001
       
        gini_imp1 = 1 - (base[0][0] / sum(base[0]))**2 - (base[0][1] / sum(base[0]))**2

        gini_imp2 = 1 - (base[1][0] / sum(base[1]))**2 - (base[1][1] / sum(base[1]))**2

        full_gini = (sum(base[0]) / (sum(base[0]) + sum(base[1]))) * gini_imp1 + (sum(base[1]) / (sum(base[0]) + sum(base[1]))) * gini_imp2

        gini_imp.append(full_gini)

      else:
        gini = []
        nums = []
        b = sorted(args[i])
        for e in range(len(b) - 1):
          num = (b[e] + b[e + 1]) / 2
          nums.append(num)

        for a in nums:
          base2 = [[0, 0], [0, 0]]
          for q in range(len(args[i])):
            if args[i][q] < a and y[q] ==1:
              base2[0][0] += 1

            elif args[i][q] < a and y[q] == 0:
              base2[0][1] += 1

            elif args[i][q] > a and y[q] == 1:
              base2[1][0] += 1
            
            else:
              base2[1][1] += 1

          for u in base:
            for z in range(len(u)):
              u[z] += 0.001
          
          gini_imp3 = 1 - (base[0][0] / sum(base[0]))**2 - (base[0][1] / sum(base[0]))**2

          gini_imp4 = 1 - (base[1][0] / sum(base[1]))**2 - (base[1][1] / sum(base[1]))**2

          ful_gini = (sum(base[0]) / (sum(base[0]) + sum(base[1]))) * gini_imp3 + (sum(base[1]) / (sum(base[0]) + sum(base[1]))) * gini_imp4

          gini.append(ful_gini)
          

      



        lowest = sorted(gini)

        low = lowest[0]
        real = gini.index(low)
        best_num[i].append(nums[real])
        gini_imp.append(low)
    
    
    
    best_gini = sorted(gini_imp)
    best_gini = best_gini[0]

    best = gini_imp.index(best_gini)
    true_true = []
    true_false = []
    false_true = []
    false_false = []
    err_weights = []
    supa_base = [[0, 0], [0, 0]]
    
    if continuous[best] == 0:
      for w in range(len(args[best])):
        if args[best][w] == 1 and y[w] == 1:
          supa_base[0][0] += 1
          true_true.append(args[best].index(args[best][w]))

        elif args[best][w] == 1 and y[w] == 0:
          supa_base[0][1] += 1
          true_false.append(args[best].index(args[best][w]))

        elif args[best][w] == 0 and y[w] == 1:
          supa_base[1][0] += 1
          false_true.append(args[best].index(args[best][w]))

        else:
          supa_base[best][1] += 1
          false_false.append(args[best].index(args[best][w]))

     
      if supa_base[0][0] > supa_base[0][1]:
        for g in true_false:
          err_weights.append(weights[g])

        for g in false_true:
          err_weights.append(weights[g])
        
        total_err = sum(err_weights)

        o = (1 - total_err) / total_err

        aos = math.log1p(o-1) / 2

        amount_of_say[best,1] = aos
        for d in range(len(weights)):
          if d in true_false or false_true:
            weights[d] = weights[d] * (2.7171**aos)

          else:
            weights[d] = weights[d] * (2.7171 ** -(aos))
      
      else:
        for g in true_true:
          err_weights.append(weights[g])

        for g in false_false:
          err_weights.append(weights[g])
        
        total_err = sum(err_weights)

        o = (1 - total_err) / total_err

        aos = math.log1p(o-1) / 2

        amount_of_say[best,0] = aos
        for d in range(len(weights)):
          if d in false_false or true_true:
            weights[d] = weights[d] * (2.7171**aos)

          else:
            weights[d] = weights[d] * (2.7171 ** -(aos))
    
    else:
      supa_num = best_num[best][0]
      for q in range(len(args[best])):
        if args[best][q] < supa_num and y[q] ==1:
          supa_base[0][0] += 1

        elif args[best][q] < supa_num and y[q] == 0:
          supa_base[0][1] += 1

        elif args[best][q] > supa_num and y[q] == 1:
          supa_base[1][0] += 1
            
        else:
          supa_base[1][1] += 1
      
      if supa_base[0][0] > supa_base[0][1]:
        for g in true_false:
          err_weights.append(weights[g])

        for g in false_true:
          err_weights.append(weights[g])
        
        total_err = sum(err_weights)

        o = (1 - total_err) / total_err

        aos = math.log1p(o-1) / 2

        amount_of_say[best, 1] = aos
        for d in range(len(weights)):
          if d in true_false or false_true:
            weights[d] = weights[d] * (2.7171**aos)

          else:
            weights[d] = weights[d] * (2.7171 ** -(aos))
     
     
      else:
        for g in true_true:
          err_weights.append(weights[g])

        for g in false_false:
          err_weights.append(weights[g])
        
        total_err = sum(err_weights)

        o = (1 - total_err) / total_err

        aos = math.log1p(o-1) / 2

        amount_of_say[best, 0] = aos
      
        for d in range(len(weights)):
          if d in false_false or true_true:
            weights[d] = weights[d] * (2.7171**aos)

          else:
            weights[d] = weights[d] * (2.7171 ** -(aos))

    if sum(weights) != 1:
      for t in range(len(weights)):
        weights[t] = weights[t] / sum(weights)
    
    rand_nums = []

    for r in range(len(weights)):
      rand_nums.append(random.randint(0, 100)/100)

    for supa_ind, e in enumerate(rand_nums):
      najs = 0
      for ind, f in enumerate(weights):
        najs += f
        if e <= najs:
          for k in range(len(continuous)):
            args[k][ind] = args[k][supa_ind]

  #for i in test:


  return amount_of_say


  
def multi_regresion(x, y, test, lr=0.01):
  d_e = []
  b = 0
  a = []


  for i in range(len(x)):
    a.append(0.5)

  #b + a1x + a2x +a3x....
  for m in range(1000):
    print(m+1)
    d_inter = []
    for s in range(len(x)):
      d_slope = []
      
      for i in range(len(x[0])):
        eq = (-2*x[s][i])*(y[i] - (a[s] * x[s][i] + b)) 
        der = -2*(y[i] - (a[s] * x[s][i] + b))
       
        d_slope.append(eq)
        d_inter.append(der)
      
      step_size_slope = sum(d_slope) * lr   
      a[s] = a[s] - (step_size_slope)
      
      

    step_size_inter = sum(d_inter) * lr
    b = b - (step_size_inter) 
    d_e.append(step_size_inter)

  preds = []
  for i in range(len(test[0])):
    predo = []
    for q in [0, 1]:
      pred = test[q][i]*a[q]
      predo.append(pred)

    preds.append(sum(predo) + b)

  return preds  

glo = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
bro = [0.1, 0.9, 0.1, 0.95, 0.2, 1.4, 0.09, 1.3, 0.26, 0.87]


def T_test(x, y):
  mean = sum(y) / len(x)
  ss_mean = []

  for i in y:
    ss_mean.append((i-mean)**2)

  lolajs = []
  for k in x:
    if k not in lolajs:
      lolajs.append(k)

  unique = []

  for i in range(len(lolajs)):
    unique.append([])

  for i in range(len(y)):
    unique[x[i]].append(y[i])

  ss_fit = []
  
  for i in range(len(unique)):
    a = sum(unique[i]) / len(unique[i])
    for u in range(100):
      der = []
      for t in range(len(unique[i])):
        b = 2*(a-unique[i][t])
        der.append(b)

      a = a - 0.001*(sum(der))

    for n in unique[i]:
      ss_fit.append((a-n)**2)

  p = sum(ss_mean) - sum(ss_fit) / (len(unique) - 1)
  fin = p / (sum(ss_fit) / (len(x) - len(unique)))

  return fin
  


c = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 0]

def Chi_squared(x):
  unique = []
  for t in x:
    if t not in unique:
      unique.append(t)
  
  xpect = len(x) / len(unique)
  ss_kam = []

  for h in unique:
    mao = 0
    for j in x:
      if j == h:
        mao += 1
    
    cock = (mao - xpect)**2
    skoro_fin = cock / xpect
    ss_kam.append(skoro_fin)

  return sum(ss_kam)



    
def ANOVA(x, y):
  ov_mean = sum(y) / len(y)
  ss_mean = []
  
  for i in y:
    ss_mean.append((ov_mean-i)**2)

  unique = []
  for u in x:
    if u not in unique:
      unique.append(u)

  supa_unique = []
  for j in range(len(unique)):
    supa_unique.append([])

  for t in unique:
    for n in range(len(x)):
      if x[n] == t:
        supa_unique[t].append(y[n])

  megusta = 0

  for r in range(len(unique)):
    k = 0
    a = sum(supa_unique[r]) / len(supa_unique[r])
    for u in range(100):
      der = []
      for t in range(len(supa_unique[r])):
        b = 2*(a-supa_unique[r][t])
        der.append(b)
      a = a - 0.001*(sum(der))

    for h in supa_unique[r]:
      k += (a-h)**2

    megusta += k

  fin = sum(ss_mean) - megusta / (len(unique) - 1)
  fin = fin / (len(x) - len(unique))
  return fin

class STD_err:

  def __init__(self, samples, n):
    self.samples = samples
    self.n = n
  

    def megusta(self):
      g = len(self.samples) / self.n
      g = math.ceil(g)
      h = len(self.samples) % g
      h = len(self.samples) - h 
      samples = self.samples[:h]
      return samples, g

    self.megusta = megusta


  def mean_error(self):
    bbc, g = self.megusta(self)
    means = []
    print(len(bbc))
    print(g)
    for i in range(g):
      print(i)

      sampl = bbc[i*self.n: (i+1) * self.n]
      mean = sum(sampl) / len(sampl)
      means.append(mean)

    std = smerodatna_odchylka(means)
    return std

  def std_error(self):
    bbc, g = self.megusta(self)
    stds = []
    for i in range(g):
      sampl = bbc[i*self.n: (i+1) * self.n]
      std = smerodatna_odchylka(sampl)
      stds.append(std)

    return stds




datacock = [2, 3, 9, 10, 6, 3, 4, 7, 5, 2, 1, 9 ,6, 3, 4]



def Bootstrapping(data):
  mean = sum(data) / len(data)
  j = 0
  if mean + abs(mean) == 0:
    j += abs(mean)

  else:
    j+= -mean

  new_data = []

  for i in data:
    new_data.append(i+j)

  hist_means = []

  for i in range(1000):
    some_data = []
    for t in range(len(new_data)):
      h = random.randint(0, len(new_data)-1)
      some_data.append(new_data[h])

    jojo = sum(some_data) / len(some_data)

    hist_means.append(jojo)

  return hist_means

def histogram(x):
  num_hist = 10
  k = sorted(x)
  rozdilek = k[-1] - k[0]
  pepa= rozdilek / num_hist
  pepa = pepa
  b = {}
  glo = k[-1]
  while len(b) != num_hist:
    b[glo] = glo-pepa
    glo = glo - pepa
    
  hist = []

  for i in range(num_hist):
    hist.append(0)


  for n in x:
    f = 0
    for g, m in b.items():
      if n < g and n > m:
        hist[f] += 1

      else:
        pass

      f += 1


  return hist

  

def Mean_absolute_deviation(x):
  k = sum(x) / len(x)
  d = []
  for t in x:
    d.append(abs(t-k))

  fin = sum(d) / len(d)

  return fin



def students_test(a, b):
  a_mean = sum(a) / len(a)
  b_maen = sum(b) / len(b)

  c = abs(a_mean - b_maen)

  o = rozptyl(a)
  u = rozptyl(b)

  j = o / len(a)
  h = u / len(b)

  l = j+h

  g = math.sqrt(l)

  return c/g  


def confidence_interval(data, percent=90):
  one = len(data) / 100
  f = one * percent
  n = sorting_numbers_algorithm(data, "fromlower")
  f = round(f)
  k = len(data) - f
  k = math.ceil(k/2)
  conf = n[k: len(data) - k]
  return conf 


def poisson_distribution(average=1, p= 100):
  k = average**p
  j = math.e**(-average)
  numeraotr = k*j
  denu = math.factorial(p)
  fin = numeraotr / denu

  return fin

m = [7, 9, 8, 10, 11, 12]
n = [4, 2, 3, 5, 1, 1.3]

def Permutation_test(a, b):
  g = []
  for x in a:
    g.append(x)

  for x in b:
    g.append(x)

  random.shuffle(g)

  new_a_mens = []
  new_b_means = []

  for h in range(500):
    new_a = []
    new_b = []   
    l = 0
    random.shuffle(g)
    for c in range(len(a)):
      new_a.append(g[c])
      l += 1

    for t in range(l - 1, len(g)):
      new_b.append(g[t])

    new_a_mens.append(sum(new_a) / len(new_a))
    new_b_means.append(sum(new_b) / len(new_b))
  
  return new_a_mens, new_b_means


def RELU(x):
  if x > 0:
    return x

  else:
    return 0

def efective_sample_size(a, b):
  n = 2
  m = 1 + correlation(a, b)
  return n / m



print(efective_sample_size([0.8, 0.95, 0.4, 0.2], [0.8, 0.95, 0.4, 0.2]))
  

l = [[0.9, 0.1], [0.05, 0.95]]
m = [[0.65, 0.3, 0.05], [0.1, 0.1, 0.8]]

v = [0, 1, 2, 2]

def RMSE(y_real, y_predicted):
  p = []
  for i in range(len(y_real)):
    b = (y_real[i] - y_predicted[i])**2
    p.append(b)

  return math.sqrt(sum(p) / len(p))


def backwardelimination(*args, target= []):
  pass





def HMM(observed, hidden, predict):
  n = len(observed) ** len(predict)
  all = len(observed[0]) + len(hidden[0])
  
  prob = []
  for t in range(len(observed)):
    prob.append([])

  for x in predict:
    for b in range(len(hidden)):
      prob[b].append(observed[b][x])
  
  all_combs = []
  snd_coms = []
  while len(all_combs) != n:
    c = []
    for t in range(len(predict)):
      l = random.randint(0, len(observed) - 1)
      c.append(l)

    app_list = []

    for q, s in enumerate(c):
      app_list.append(prob[s][q])

    if app_list not in all_combs:
      all_combs.append(app_list)
      snd_coms.append(c)

    else:
      pass

  dah = []

  for d in snd_coms:
    b = []
    for e in  range(len(d)):
      if e == len(d) -1:
        pass
      else:
        q = d[e]
        nxt = d[e+1]
        b.append(hidden[q][nxt])

    dah.append(b)

  probs = []

  for i in range(n):
    f = dah[i] + all_combs[i]
    a = 1
    for g in f:
      a = a*g
    
    probs.append(a)

  fin = max(probs)
  m = probs.index(fin)
  supafin = snd_coms[m]

  return supafin



      

      



   

  

    

    

    
def PCA():
  pass
          

      

      
print(HMM(m, l, v))

  


  



  
  
  





    
    
      

    

  


  

  

  
  
  
  
  

  
    


      

          
            

    


    

        



      






      

      







          


        

        

        

        
  
  

  

  
    

  

  
  
  

  











    


    

  
    
  

    







  
  





        











  
  
  

  
  

  


  



  



        





 
    
    

  

  



  




    



    

    







  












  







    


  
  

    

    
  

    


      
      

  


      
      
        
    
