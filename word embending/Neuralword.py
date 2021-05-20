import numpy as np
    
class Project_and_concat() : 
    """
    The input is a vector x = (w_{t-1}, w_{t-2}, ..., w_{t-n+1})
    where w_i is the index corresponding to the word. 
    For example, for n=4 the input vector x can be
    (4, 2, 10)
    where 4, 2 and 10 are the indexes of the corresponding words.
    """
    def __init__(self, features, dict_size) :
        self.nb_features = features
        self.dict_size = dict_size
        self.C = np.random.randn(dict_size, self.nb_features)
        self.nb_params = features * dict_size # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, 
        # prend un vecteur de la taille self.nb_params
        self.C = params.reshape((self.dict_size, self.nb_features))
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return np.concatenate(self.C)
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        temp = self.C[X, :]
        result = np.reshape(temp, (np.shape(X)[0], self.nb_features * np.shape(X)[1]))
        return result.T
    def backward(self,grad_sortie=None) :
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        Gtemp = np.zeros(self.C.shape)
        line_Ix, counts = np.unique(self.save_X, return_counts=True)
        for i, l_Ix in enumerate(line_Ix):
            Gtemp[l_Ix, :] = counts[i]
        grad_local = np.ravel(Gtemp)
        grad_entree = None
        return grad_local, grad_entree
# 2 étapes dans cette couche, les selections des lignes de C puis la concaténation
# est ce que la selection des lignes de C rentre dans le calcul du dradient d'entree

import numpy as np
    
class Project() : 
    """
    The input is a vector x = (w_{t-1}, w_{t-2}, ..., w_{t-n+1})
    where w_i is the index corresponding to the word. 
    For example, for n=4 the input vector x can be
    (4, 2, 10)
    where 4, 2 and 10 are the indexes of the corresponding words.
    """
    def __init__(self, features, dict_size) :
        self.nb_features = features
        self.dict_size = dict_size
        self.C = np.random.randn(dict_size, self.nb_features)
        self.nb_params = features * dict_size # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, 
        # prend un vecteur de la taille self.nb_params
        self.C = params.reshape((self.dict_size, self.nb_features))
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return np.concatenate(self.C)
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        temp = self.C[X, :]
        result = np.reshape(temp, (np.shape(X)[0], self.nb_features * np.shape(X)[1]))
        return temp
    def backward(self,grad_sortie=None) :
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        Gtemp = np.zeros(self.C.shape)
        line_Ix, counts = np.unique(self.save_X, return_counts=True)
        for i, l_Ix in enumerate(line_Ix):
            Gtemp[l_Ix, :] = counts[i]
        grad_local = np.ravel(Gtemp)
        grad_entree = None
        return grad_local, grad_entree
# 2 étapes dans cette couche, les selections des lignes de C puis la concaténation
# est ce que la selection des lignes de C rentre dans le calcul du dradient d'entree
    
class tanh() : 
    def __init__(self) :
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=X
        return np.tanh(X)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=(1/(np.cosh(self.save_X)))*grad_sortie
        return grad_local,grad_entree
     
        
class Dense() : 
    def __init__(self,nb_entree,nb_sortie) :
        self.n_entree=nb_entree
        self.n_sortie=nb_sortie
        self.nb_params=nb_entree*nb_sortie+nb_sortie # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        self.A=np.random.randn(nb_sortie,nb_entree)
        self.b=np.random.randn(nb_sortie)
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        self.A=np.reshape(params[:self.nb_params-self.n_sortie],(self.n_sortie,self.n_entree))
        self.b=params[self.nb_params-self.n_sortie:]
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        param=np.zeros(self.nb_params)
        param[:self.n_sortie*self.n_entree]=np.ravel(self.A)
        param[self.nb_params-self.n_sortie:]=self.b
        return param
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Y=self.A.dot(X)
        b=np.outer(self.b,np.ones(len(Y[0])))
        Z=Y+b
        return Z
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_entree=np.transpose(self.A).dot(grad_sortie)
        ga=grad_sortie.dot(np.transpose(self.save_X))
        gb=grad_sortie.dot(np.ones(len(grad_sortie[0])))
        grad_local=np.zeros(self.nb_params)
        grad_local[:self.n_sortie*self.n_entree]=np.ravel(ga)
        grad_local[self.nb_params-self.n_sortie:]=gb
        return grad_local,grad_entree
    
def ilogit(z) :
    """Fonction calculant l'opération ilogit sur la matrice z
    Fonction utilisée dans la couche Ilogit_and_KL"""
    
    return np.exp(z)/(np.sum(np.exp(z),axis=0))
    
class Ilogit_and_KL() : 
    def __init__(self,donnees) :
        self.D=donnees
        self.nb_params=0 # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return None
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        return -(self.D*np.log(ilogit(X))).sum()
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=ilogit(self.save_X)-self.D
        return grad_local,grad_entree        
          
class Network() : 
    def __init__(self, list_layers) :
        self.list_layers = list_layers
        self.nb_params=sum([l.nb_params for l in list_layers]) # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        current_position = 0
        for l in self.list_layers:
            n = l.nb_params
            l.set_params(params[current_position:current_position+n])
            current_position += n
            
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        p = []
        for l in self.list_layers:
            new_p = l.get_params()
            if new_p is not None :
                p.append(new_p)
        return np.concatenate(p)
    
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Z = np.copy(X)
        for l in self.list_layers:
            Z = l.forward(Z)
        return Z
    
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        b = []
        gs = grad_sortie
        for l in reversed(self.list_layers):
            gl, ge = l.backward(gs)
            
            if gl is not None:
                b.append(gl)
            gs = ge
        b.reverse()
        grad_local = np.concatenate(b)
        grad_entree = ge
        return grad_local,grad_entree
    
