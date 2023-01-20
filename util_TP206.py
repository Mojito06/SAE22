from sounddevice import *
from pylab import *
import scipy.signal as sc       # pour utiliser les fonctions liées aux filtres 

def uniform_quantizer(s, niv, nmin, nmax):
    ''' quantification uniforme 
    sq, error = uniform_quantizer(s, niv, nmin, nmax):
    
    input : 
    - s : signal à quantifier
    - niv : nombre de niveaux de sortie
    - nmin : niveau minimal de sortie 
    - nmax : niveau maximal de sortie
    output : 
    - sq : signal quantifie
    - d: 
    '''
    sq = empty(len(s))
    d = (nmax-nmin)/(niv-1)
    for i in range(len(s)):
        if s[i]>=nmax: 
            sq[i]=nmax
        elif s[i]<=nmin:
            sq[i]=nmin
        else :
            if niv % 2 == 0 :
                sq[i] = d * np.round((s[i]-d/2)/d) + (d/2)   
            else : 
                sq[i] = d * np.round(s[i]/d)        
    return sq, d

def snr(s, sq):
    '''fonction snr : calcule le rapport signal à bruit (en dB) de quantification entre le 
    signal s et le signal quantifié sq.'''
    Ps=norm(s,2)**2/len(s)
    erreur=norm((s-sq),2)**2/len(s)
    return (10*log10(Ps/erreur)) 


def A_law(s,A): 
    '''fonction A_law - applique la fonction de la loi A sur le signal d'entrée s et rend
    le resultat sloi'''
    sloi = empty(len(s))
    for i in range(len(s)):
       if abs(s[i])<1/A:
        sloi[i]=sign(s[i])*A*abs(s[i])/(1+log(A))
       else:
        sloi[i]=sign(s[i])*(1+log(A*abs(s[i])))/(1+log(A))
    return sloi

def inverse_A_law(sloiQ,A):
    '''fonction inverse_A_law - applique la fonction inverse de la loi A sur le signal
    d'entrée sloiQ, et rend la fonction sqNU'''
    sqNU = empty(len(sloiQ))
    for i in range(len(sloiQ)):
        if abs(sloiQ[i]) < 1/(1+log(A)):
            sqNU[i] = sign(sloiQ[i]) * abs(sloiQ[i]) * (1+log(A)) / A
        else:
            sqNU[i] = sign(sloiQ[i]) * exp(abs(sloiQ[i])*(1+log(A))-1)/ A
    return sqNU


def plotSpectreAmplitudeNormalise(signalEntree, Fe_Hz):
    """trace le spectre d'amplitude NORMALISE du signal sig, qui est échantillonné à fe Hz."""
    longueur=len(signalEntree)
    specOrig=abs((1/longueur)*fftshift(fft(signalEntree)))
    freq = arange(-1/2,1/2,1/longueur) 
    plot(freq, specOrig,'-r')
    grid()
    xlabel('Frequence (Hertz)')
    ylabel('Amplitude')

def plotSpectreAmplitude(sig,fe):
    """trace le spectre d'amplitude du signal sig, qui est échantillonné à fe Hz."""
    longueur=len(sig)
    specOrig=abs((1/longueur)*fftshift(fft(sig)))
    freq = arange(-fe/2,fe/2,fe/longueur) 
    plot(freq, specOrig,'-r')
    grid()
    title("Spectre d'amplitude")
    xlim([-fe/2, fe/2])
    xlabel('Frequence (Hertz)')
    ylabel('Amplitude')
    
def filtrage(s, Fs, Fc, typ):
    """applique un filtre passe bas ou passe-haut de fréquence de coupure Fc sur le signal s
    - s : signal que l'on souhaite filtrer
    - Fs : fréquence d'échantillonnage du signal
    - Fc : fréquence de coupure du filtre
    - typ : 'low' ou 'high' , respectivement passe-bas ou passe-haut 
    - Exemple : sfiltre = filtrage(son, 44000, 2000,'low')
    - Cette instruction va filtrer le signal "son" (intialement échantillonné à 44000Hz) 
    à l'aide d'un passe-bas de fréquence de coupure 2000Hz. 
    Le signal filtré sera stocké dans la variable sfiltre.
    """
    ordre=8
    b, a = sc.butter(ordre, Fc/(Fs/2), typ)
    return sc.lfilter(b, a, s)


def puissancedBm(signal):
    """Fonction permet de calculer la puissance en dBm de signal.
    ===========================
    Parametre : 
    - signal : signal a transmettre
    ===========================
    """
    Pref = 0.001
    puissanceW = linalg.norm(signal,2)**2/size(signal)
    puissance = 10*math.log(puissanceW/Pref,10)
    return round(puissance,4)

import math
def channel(signal, Type, varargin_1,varargin_2,varargin_3=None) :
    """La fonction signalRecu = channel(signal, Type, varargin) permet de simuler l'attenuation de differents types de canaux transmission.
    Le nombre et la nature des parametres changent en fonction du type du canal choisi.
    ===========================
    Parametres d'entree :
    - signal : signal a transmettre
    - Type : type de canal : 'espacelibre', 'coaxial' et 'fibre'
    - varargin : 
    si Type = 'espacelibre'
     varargin_1 : longueur du canal (en km)
     varargin_2 : frequence porteuse (en MHz)
    si Type = 'coaxial'
     varargin_1 : longueur du canal (en km)
     varargin_2 : pertes d'insertion (en dB/100m)
    si Type = 'fibre'
     varargin_1 : longueur du canal (en km)
     varargin_2 : affaiblissement (en dB/km)
     varargin_3 : longueur d'onde pour la transmission (en nm)
    Parametres de sortie :
    - signalRecu : signal en sortie du canal
    """
    if Type == 'espacelibre' :
        alpha = 2
        pertes_dBtot = 32.45 + 20*math.log(varargin_2,10) + 10*alpha*math.log(varargin_1,10)
        
    elif Type == 'coaxial' : 
        distance_km = varargin_1
        pertes_dBper100metre = varargin_2
        pertes_dBtot = pertes_dBper100metre*10*distance_km
        
    elif Type == 'fibre' :
        distance_km = varargin_1
        pertes_dBperkm = varargin_2
        L0_nm = varargin_3
        c = 3*10**8
        Fp_Hz = c/L0_nm
        pertes_dBtot = pertes_dBperkm*distance_km
        
    gainG  = 10**(-pertes_dBtot/20)
    signalRecu = []
    for i in signal :
        signalRecu.append(i * gainG)
    return signalRecu
