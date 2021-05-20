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
    
class Arctan() : 
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
        return np.arctan(X)
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=1/(1+self.save_X**2)*grad_sortie
        return grad_local,grad_entree
    
class Sigmoid() : 
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
        self.save_X=np.copy(X)
        Z=1/(1+np.exp(-X))
        return Z
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        X=self.save_X
        grad_entree=(np.exp(-X)/((1+np.exp(-X))**2))*grad_sortie
        return grad_local,grad_entree
    
class RELU() : 
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
        self.save_X=np.copy(X)
        Z=np.max(0,X)
        return Z
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        X=self.save_X
        grad_entree=np.max(X/np.abs(X),0)*grad_sortie
        return grad_local,grad_entree

class ABS() : 
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
        self.save_X=np.copy(X)
        Z=np.abs(X)
        return Z
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        X=self.save_X
        grad_entree=X/np.abs(X)*grad_sortie
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
    
class Loss_L2() : 
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
        return np.linalg.norm(X-self.D)**2/2
    def backward(self,grad_sortie) :  
        # retropropagation du gradient sur la couche, 
        #grad_sortie est le vecteur du gradient en sortie
        #Cette fonction rend :
        #grad_local, un vecteur de taille self.nb_params qui contient le gradient par rapport aux parametres locaux
        #grad_entree, le gradient en entree de la couche 
        grad_local=None
        grad_entree=self.save_X-self.D
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