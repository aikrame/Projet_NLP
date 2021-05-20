import numpy as np
class Generic() : 
    def __init__(self) :
        self.nb_params=None # Nombre de parametres de la couche
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
        return None
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=None
        return grad_local,grad_entree
    
class Project_and_concat() : 
    """
    The input is a vector x = (w_{t-1}, w_{t-2}, ..., w_{t-n+1})
    For example, for n=4 the input vector x can be
    (4, 2, 10)
    where 4, 2 and 10 are the indexes of the corresponding words.
    """
    def __init__(self, nb_features,dict_size) : # V*m ou m*V
        self.nb_features = nb_features
        self.dict_size = dict_size
        self.C = np.random.randn(dict_size,nb_features)
        self.nb_params = nb_features * dict_size # Nombre de parametres de la couche
        self.save_X = None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        pass
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        return np.ravel(self.C)
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X = np.copy(X)
        return np.ravel(np.concatenate(C[X, :]))
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=np.reshape(grad_sortie,(dict_size,nb_features))
        return grad_local,grad_entree
        
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
        self.nb_params=nb_entree*nb_sortie # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
        self.C=np.random.randn(nb_sortie,nb_entree)
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        self.A=np.reshape(params,(self.n_sortie,self.n_entree))
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        param=np.zeros(self.nb_params)
        param[:self.n_sortie*self.n_entree]=np.ravel(self.A)
        return param
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Y=self.A.dot(X)
        return Y
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_entree=np.transpose(self.A).dot(grad_sortie)
        ga=grad_sortie.dot(np.transpose(self.save_X))
        grad_local=np.zeros(self.nb_params)
        grad_local=np.ravel(ga)
        return grad_local,grad_entree   
    
def ilogit(z) :
    """Fonction calculant l'opération ilogit sur la matrice z
    Fonction utilisée dans la couche Ilogit_and_KL"""
    z=np.exp(z)
    somme=z.sum(axis=0)
    return z/somme
    
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
    def __init__(self,list_layers) :
        self.list_layers=list_layers
        param=0
        for layer in list_layers:
            param+=layer.nb_params
        self.nb_params=param # Nombre de parametres de la couche
        self.save_X=None # Parametre de sauvegarde des donnees
    def set_params(self,params) : 
        # Permet de modifier les parametres de la couche, en entree, prend un vecteur de la taille self.nb_params
        nbparam=0
        for layer in self.list_layers:
            layer.set_params(params[nbparam:nbparam+layer.nb_params])
            nbparam+=layer.nb_params
    def get_params(self) : 
        # Rend un vecteur de taille self.params qui contient les parametres de la couche
        params=np.zeros(self.nb_params)
        nbparam=0
        for layer in self.list_layers:
            params[nbparam:nbparam+layer.nb_params]=layer.get_params()
            nbparam+=layer.nb_params
        return params
    def forward(self,X) : 
        # calcul du forward, X est le vecteur des donnees d'entrees
        self.save_X=np.copy(X)
        Z=np.copy(X)
        for layers in self.list_layers:
            Z=layers.forward(Z)
        return Z
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche
        grad_local=np.zeros(self.nb_params)
        nbparam=0
        grad_entree=grad_sortie
        for layer in reversed(self.list_layers):
            grad=layer.backward(grad_entree)
            grad_entree=grad[1]
            grad_local[self.nb_params-layer.nb_params-nbparam:self.nb_params-nbparam]=grad[0]
            nbparam+=layer.nb_params
        return grad_local,grad_entree