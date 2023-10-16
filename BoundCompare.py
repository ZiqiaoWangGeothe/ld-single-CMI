import numpy as np
from sklearn.model_selection import train_test_split
import torch
from sklearn.feature_selection import f_regression, mutual_info_regression, mutual_info_classif
from sklearn import metrics
import scipy.optimize as opt
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def SuperSample(num=2, num_class=2, d=2, sep=1.0):
  samples = make_classification(n_samples=2*num, n_features=d, n_redundant=0, n_classes=num_class, n_informative=d, n_clusters_per_class=1, class_sep=sep, flip_y=0)
  red = samples[0][samples[1] == 0]
  blue = samples[0][samples[1] == 1]
  for i in range(int(num_class)):
    cluster = samples[0][samples[1] == i]
    targets = np.zeros(len(cluster))+i
    if i==0:
      inputs = cluster
      labels = targets
    else:
      labels = np.append(labels,targets)
      inputs = np.concatenate((inputs,cluster),axis=0)

  X_0, X_1, y_0,  y_1 = train_test_split(
      inputs, labels, test_size=0.5, random_state=42)
  return X_0, X_1, y_0, y_1


def TrainClassification(X_0, X_1, y_0, y_1, U, d, num_class):
  class MultiClassification(torch.nn.Module):
      def __init__(self, input_dim, output_dim):
          super(MultiClassification, self).__init__()
          self.linear = torch.nn.Linear(input_dim, output_dim)
          
      def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


  criterion = torch.nn.CrossEntropyLoss()
  epochs = 500
  input_dim = d 
  output_dim = num_class
  learning_rate = 0.01
  model = MultiClassification(input_dim,output_dim)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

  X_train =(1-U).reshape(-1, 1)*X_0+U.reshape(-1, 1)*X_1
  y_train = (1-U)*y_0+U*y_1

  X_train, y_train = torch.Tensor(X_train),torch.Tensor(y_train).type(torch.LongTensor)
  X_0, y_0, X_1, y_1 = torch.Tensor(X_0),torch.Tensor(y_0).type(torch.LongTensor), torch.Tensor(X_1),torch.Tensor(y_1).type(torch.LongTensor)

  losses = []
  losses_test = []
  Iterations = []
  iter = 0
  for epoch in range(int(epochs)):
      x = X_train
      labels = y_train
      optimizer.zero_grad() 
      outputs = model(X_train)
      loss = criterion(outputs, labels) 
      loss.backward()
      optimizer.step()
      
      with torch.no_grad():
        total = 0
        correct = 0
        total += y_train.size(0)
        correct += np.sum(outputs.data.max(1)[1].detach().view_as(y_train).numpy() == y_train.detach().numpy())
        accuracy = 100 * correct/total
        if accuracy>=99:
          break
      
  with torch.no_grad():
      outputs_0 = torch.squeeze(model(X_0))
      predicted_0 = outputs_0.data.max(1)[1].detach().view_as(y_0).numpy()
      Err0= (predicted_0 != y_0.detach().numpy()).astype(int)

      outputs_1 = torch.squeeze(model(X_1))
      predicted_1 = outputs_1.data.max(1)[1].detach().view_as(y_1).numpy()
      Err1= (predicted_1 != y_1.detach().numpy()).astype(int)
      Delta = Err1-Err0

  return Err0, Err1, Delta, predicted_0.astype(int), predicted_1.astype(int)


def RunClassExp(n=2, n_class=2, dim=2, diff=1.0, num_Z=50, num_U=100):
  disint_mi, disint_fcmi=0, 0
  perins_Z=0
  for i in range(int(num_Z)):
    X_0, X_1, y_0, y_1 = SuperSample(num=n, num_class=n_class, d=dim, sep=diff)
    first_train, second_train = 0, 0
    first_trerr, second_trerr = 0, 0
    for j in range(int(num_U)):
      U=np.random.binomial(size=n, n=1, p=0.5)
      Err0, Err1, Delta, Yhat0, Yhat1 = TrainClassification(X_0, X_1, y_0, y_1, U, dim, n_class)
      error = ((-1)**U*Delta).mean()
      first_train += 1-U 
      second_train += U
      first_trerr += (1-U)*Err0 
      second_trerr += U*Err1 
      train_error = ((1-U)*Err0+U*Err1).mean()
      tr_square = train_error**2
            
      if j==0 and i==0:
        L0, L1, Del, Err, Tr_err, Trerr_square=Err0, Err1, Delta, error, train_error, tr_square
        F0, F1, Rad = Yhat0, Yhat1, U #if i==0 else np.vstack((Rad,U))
      else:        
        L0, L1, Del=np.vstack((L0,Err0)), np.vstack((L1,Err1)), np.vstack((Del,Delta))
        Err, Tr_err, Trerr_square=np.append(Err,error), np.append(Tr_err,train_error), np.append(Trerr_square,tr_square)
        F0, F1, Rad=np.vstack((F0,Yhat0)), np.vstack((F1,Yhat1)), np.vstack((Rad,U))
    
    perins_Z += (first_trerr/first_train)**2+(second_trerr/second_train)**2
    midelta, fcmi = np.zeros(n), np.zeros(n)
    for k in range(n):
      midelta[k]=mutual_info_classif(Del[i*num_U:(i+1)*num_U-1,k].reshape(-1, 1), Rad[i*num_U:(i+1)*num_U-1,k], discrete_features=[True])
      fcmi[k]=mutual_info_classif((n_class*F0[i*num_U:(i+1)*num_U-1,k]+F1[i*num_U:(i+1)*num_U-1,k]).reshape(-1, 1), Rad[i*num_U:(i+1)*num_U-1,k], discrete_features=[True])

    disint_mi += (np.sqrt(2*midelta)).mean()
    disint_fcmi += (np.sqrt(2*fcmi)).mean()
      

  mi, mil0, mil1, mipair=np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
  for i in range(n):
    mi[i]=mutual_info_classif(Del[:,i].reshape(-1, 1), Rad[:,i], discrete_features=[True])
    mil0[i]=mutual_info_classif(L0[:,i].reshape(-1, 1), Rad[:,i], discrete_features=[True])
    mil1[i]=mutual_info_classif(L1[:,i].reshape(-1, 1), Rad[:,i], discrete_features=[True])
    mipair[i]=mutual_info_classif((n_class*L0[:,i]+L1[:,i]).reshape(-1, 1), Rad[:,i], discrete_features=[True])

  bound=(np.sqrt(2*mi)).mean()
  bound0=2*(np.sqrt(2*mil0)).mean()
  bound1=2*(np.sqrt(2*mil1)).mean()
  bound2=(np.sqrt(2*mipair)).mean()
  disintbound=disint_mi/num_Z
  fcmibound=disint_fcmi/num_Z
  # bound3=(np.sqrt(2*fcmi)).mean()
  inpbound1= 2*mil0.mean()/np.log(2)
  inpbound2= mipair.mean()/np.log(2)
  L_n = Tr_err.mean()
  I_mean = mil0.mean()
  forsharp=perins_Z/(num_Z*2)
  subbound=4*I_mean+4*np.sqrt(I_mean*L_n)
  exactbound=mi.mean()/np.log(2)

  fun1 = lambda x: x[0]*L_n + I_mean/x[1]
  cons1 = ({'type': 'ineq', 'fun': lambda x:  -np.exp(2*x[1]) - np.exp(-2*x[1]*(x[0]+1)) + 2})
  bnds1 = ((0, None), (0, None))
 
  # init = 2*np.sqrt(I_mean/L_n)
  # (0.5, 0.1)
  res1 = opt.minimize(fun1, (0.5, 0.1), method='SLSQP', bounds=bnds1,
               constraints=cons1, options = {'disp':False})
  optbound = res1.fun

  fun2 = lambda x: x[0]*(L_n-(1-x[2]**2)*Trerr_square.mean()) + I_mean/x[1]
  cons2 = ({'type': 'ineq', 'fun': lambda x:  (-np.exp(2*x[1]) - np.exp(-2*x[1]*(x[0]*(x[2]**2)+1)) + 2)*x[1]})
  bnds2 = ((0, None), (0, None), (0, 1))

  res2 = opt.minimize(fun2, (1, 0.1, 0.5), method='SLSQP', bounds=bnds2,
               constraints=cons2, options = {'disp':False})
  varbound = res2.fun

  fun3 = lambda x: x[0]*(L_n-(1-x[2]**2)*forsharp.mean()) + I_mean/x[1]

  res3 = opt.minimize(fun3, (1, 0.1, 0.5), method='SLSQP', bounds=bnds2,
               constraints=cons2, options = {'disp':False})
  sharpbound = res3.fun

  def kl(q,p):
    if q>0:
        return q*np.log(q/p) + (1-q)*np.log( (1-q)/(1-p) )
    else:
        return np.log( 1/(1-p) )
  # def conkl(R):
  #   return (mipair.mean()-kl(L_n,L_n/2 + R/2))*R*(1-R)
  conkl = ({'type': 'ineq', 'fun': lambda x:  mipair.mean()-kl(L_n,L_n/2 + x[0]/2)},
        {'type': 'ineq', 'fun': lambda x: x[0]*(1-x[0])},
       {'type': 'ineq', 'fun': lambda x: 1-L_n/2-x[0]/2})
  objective = lambda R: -R
  # conskl = ({'type': 'ineq', 'fun' : lambda R: (mipair.mean()-kl(L_n,L_n/2 + R/2))*(1-L_n/2-R/2)})
  results = opt.minimize(objective, 0.5, method='SLSQP',
  constraints = conkl,
  options = {'disp':False})
  klbound = results.x[0]
  return bound, bound0, bound1, bound2, disintbound, fcmibound, subbound, optbound, varbound, sharpbound, klbound, inpbound1, inpbound2, exactbound, Err.mean(), Tr_err.mean()


import warnings
def PlotBound(n_class=2, dim=5, diff=1, num_Z=50, num_U=100, sample=[5, 10, 20, 30, 50]):
  print("------Class:", n_class, " Difficulty:", 1/diff,"------")
  delta_bound, pos_bound, neg_bound, ecmi_bound, disint_bound, fcmi_bound  =[], [], [], [], [], [] 
  sub_bound, opt_bound, var_bound, sharp_bound, kl_bound, single_ip_bound, pair_ip_bound, exact_bound = [], [], [], [], [], [], [], []
  err, tr_loss = [], []
  for num in sample:
    dmi, pmi, nmi, ecmi, discmi, fcmi, subcmi, optcmi, varcmi, sharpcmi, klcmi, inter1, inter2, exact, em_err, em_trloss = RunClassExp(n=num, n_class=n_class, dim=dim, diff=diff, num_Z=num_Z, num_U=num_U)

    print('Num %d: LD: %.3f | SL: %.3f | SL2: %.3f | eCMI: %.3f | disineCMI: %.3f | fCMI: %.3f | SubFast: %.3f | OptFast: %.3f | VarFast: %.3f|'
                          '| SharpFast: %.3f | BiKL: %.3f | SIB: %.3f |  PIB: %.3f | Exact: %.3f | Gap: %.3f | Train: %.3f' %
                          (num, dmi, pmi, nmi, ecmi, discmi, fcmi, subcmi, optcmi, varcmi, sharpcmi, klcmi,
                          inter1, inter2, exact, em_err, em_trloss))
    delta_bound.append(dmi)
    pos_bound.append(pmi)
    neg_bound.append(nmi)
    ecmi_bound.append(ecmi)
    disint_bound.append(discmi)
    single_ip_bound.append(inter1)
    pair_ip_bound.append(inter2)
    fcmi_bound.append(fcmi)
    sub_bound.append(subcmi)
    opt_bound.append(optcmi)
    var_bound.append(varcmi)
    sharp_bound.append(sharpcmi)
    kl_bound.append(klcmi)
    exact_bound.append(exact)
    err.append(np.abs(em_err))
    tr_loss.append(em_trloss)

  plt.figure()
  plt.plot(sample,delta_bound, color='orange', label="LD")
  plt.plot(sample,pos_bound, color='red', label="SL")
  plt.plot(sample,neg_bound, label="SL2")
  plt.plot(sample,ecmi_bound, color='blue', label="eCMI")
  plt.plot(sample,disint_bound, label="disInt")
  plt.plot(sample,fcmi_bound, label="fCMI")
  plt.plot(sample,single_ip_bound, color='cyan', label="SIP-bound")
  plt.plot(sample,pair_ip_bound, color='magenta', label="PIP-bound")
  plt.plot(sample,tr_loss, color='black', label="TrErr")
  plt.plot(sample,sub_bound, color='yellow', label="Sub")
  plt.plot(sample,opt_bound, color='pink', label="Opt")
  plt.plot(sample,var_bound, color='green', label="Var")
  plt.plot(sample,sharp_bound, marker = 'x',markersize = 8, label="Sharp")
  plt.plot(sample,kl_bound, marker = '2',markersize = 8, label="KL")
  plt.plot(sample,err, color='gray', label="Err")
  plt.plot(sample,exact_bound, label="Perfect-bound")
  plt.legend()

  plt.figure()
  plt.plot(sample,delta_bound, marker = '+',markersize = 8, color='orange', label="LD")
  plt.plot(sample,pos_bound, color='red', label="SL")
  plt.plot(sample,ecmi_bound, color='blue', label="eCMI")
  plt.plot(sample,disint_bound, label="disInt")
  plt.plot(sample,fcmi_bound, label="fCMI")
  plt.plot(sample,sub_bound, color='yellow', label="Sub")
  plt.plot(sample,err, color='gray', label="Err")
  plt.legend()

  plt.figure()
  plt.plot(sample,opt_bound, color='pink', label="Opt")
  plt.plot(sample,var_bound, color='green', label="Var")
  plt.plot(sample,sharp_bound, marker = 'x',markersize = 8, label="Sharp")
  plt.plot(sample,kl_bound, marker = '2',markersize = 8, label="KL")
  plt.plot(sample,err, color='gray', label="Err")
  plt.legend()

  plt.figure()
  plt.plot(sample,single_ip_bound, color='cyan', label="SIP-bound")
  plt.plot(sample,pair_ip_bound, color='magenta', label="PIP-bound")
  plt.plot(sample,exact_bound, label="Perfect-bound")
  plt.legend()

  print("----------------Results----------------")
  print('LD', delta_bound)
  print('pos', pos_bound)
  print('neg', neg_bound)
  print('ecmi', ecmi_bound)
  print('disint', disint_bound)
  print('fcmi', fcmi_bound)
  print('opt', opt_bound)
  print('var', var_bound)
  print('sharp', sharp_bound)
  print('kl', kl_bound)
  print('SIB', single_ip_bound)
  print('PIB', pair_ip_bound)
  print('Exact', exact_bound)
  print('Err', err)
  print('TrErr', tr_loss)