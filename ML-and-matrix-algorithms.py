import math 

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
    

weight = [58, 50, 90, 45, 120, 40, 230]
height = [190, 185, 170, 180, 170, 178, 180]
lab = [1, 1, 0, 1, 0 ,1 ,0]
test_x = [90, 50]
test_y = [180, 200]

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

  


    


    
  

    


      
      

  


      
      
        
    
