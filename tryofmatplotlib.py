import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import pandas


everything = pandas.read_csv("C:\\Users\\Tadeas\\Downloads\\slary\\Salary_Data.csv")

print(everything)


x = everything["YearsExperience"]
y = everything["Salary"]



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

preds = linearregresion(x/10, y/100000, x/10)


real_preds = []
for i in preds:
  real_preds.append(i*100000)

  
plt.figure(figsize=(16, 9), dpi=100)

plt.ylabel("Salary")
plt.xlabel("Years")

plt.title("Linear regresion", fontdict={"fontname": "Comic Sans MS", "fontsize": 20})


plt.plot(x, y, "b", label="real")
plt.plot(x, real_preds, label="predicted")


plt.legend()


plt.show()  