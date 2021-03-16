import math 

matrix = [[1,2, 3], [4, 5, 6], [7, 8 ,9]]
p = [[4, 2, 8], [10, 12, 4], [4, 5, 9]]

y = [1, 6, 2 ,3 , 9, 7, 8, 4, 5, 10, 78 ,96, 45 , 32, 98 , 71, 4, 32, 85, 43, 14, 65 ,12, 4, 78 ,633 ,15, 824, 966, 74, 56, 14, 78, 96, 48] * 1000


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

k = sorting_numbers_algorithm(y, "fromhighe")

print(len(k))

  


    


    
  

    


      
      

  


      
      
        
    
