import matplotlib
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
a=imread('minion2.jpg')
x=np.mean(a,-1)
print('Shape of image = '+ str(x.shape))
print('Size of image =' + str(x.size))
img=plt.imshow(x)
img.set_cmap('gray')
plt.axis('off')
plt.show()

# performing SVD

u,s,vt=np.linalg.svd(x,full_matrices=False)
#economy svd extracting only first m columns of u

s=np.diag(s)

j=0
for r in (5,20,50,60,75):
    # approximation,
    xapprox=u[:,:r] @ s[0:r,:r] @ vt[:r,:] 
    print('Rank ='+ str(r))
    print('Size'+ str((u[:,:r].size + s[0:r,:r].size + vt[:r,:].size)) )
    plt.figure(j+1)
    j+=1
    img=plt.imshow(xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r ='+ str(r))
    plt.show()
    
plt.figure(1)
plt.semilogy(np.diag(s))
plt.title('singular value')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(s))/np.sum(np.diag(s)))
plt.title('singular value:cumulative sum')
plt.show()
