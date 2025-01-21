import torch
def get_values(pred, true, n=19):
  ret = torch.zeros_like(true)*0.0
  # print('ret',ret.shape)
  # print('pred',pred.shape)
  # print('pred max',pred.max(),pred.min())
  # print('true',true.shape)
  # print('true max',true.max(),true.min())
  for i in range(n):
    ret[true==i] = pred[:,i,:,:][true==i]
  return ret[ret.ne(0.0)] #ground truth has extra class 255 so ignoring corresponding zeros