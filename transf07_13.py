################## 
## IMPORTS et variables globales
from scipy.io import wavfile as wav
import scipy.signal as sgnl
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint
import sounddevice as sd
from time import time

FFTSIZE = 2048    # nombre d'échantillons 
SMPFREQ = 44100    # taux d'échantillonnage (44100 échantillons par seconde)
SUB = 16384    # nombre d'échantillons sur lesquels on mesure les amplitudes et les déphasages
AMP_REC = 0.2    # volume sonore
MARGE_REC = 30000
DELAI_ACTIVATION_CONTROLE_ACTIF = 1.0    # secondes
BUFFER = np.zeros(FFTSIZE)    # "tampon" utilisé pour le filtrage dans engine4()
s1_rec = np.array([])    # enregistrement du micro de réf
s2_rec = np.array([])    # enregistrement du micro d'erreur
s3_rec = np.array([])    # enregistrement de la source primaire
s4_rec = np.array([])    # enregistrement de la source secondaire

## Remarque : les fonctions utilisent les variables globales suivantes en écriture : BUFFER, a, rec, out, Hp, Hs, Hf

##
def transfer_sine(s1,s2,f,sampFreq): 
    ## calcule la fonction de transfert S2/S1 à la fréquence f
    A1 = (2**.5)*np.mean(s1.astype(np.float)**2)**.5 # amplitude de s1
    A2 = (2**.5)*np.mean(s2.astype(np.float)**2)**.5 # amplitude de s2
    r = A2/A1
    xcorr = sgnl.correlate(r*s1, s2) # corrélation croisée de s1 et s2
    i_max = np.argmax(xcorr)
    
    ## Interpolation parabolique, détermination de l'abscisse alpha du sommet
    x1,x2,x3=i_max-1,i_max,i_max+1
    y1,y2,y3=xcorr[x1],xcorr[x2],xcorr[x3]
    a = y1/((x1-x2)*(x1-x3)) + y2/((x2-x1)*(x2-x3)) + y3/((x3-x1)*(x3-x2))
    b = -(x2+x3)*y1/((x1-x2)*(x1-x3)) - (x1+x3)*y2/((x2-x1)*(x2-x3)) - (x1+x2)*y3/((x3-x1)*(x3-x2))
    alpha = -b/(2*a)
    ##
    
    delta_t = alpha-(len(s2)-1)
    ps = 2*np.pi*f*delta_t/sampFreq    # déphasage de s2 par rapport à s1
    return r,ps

def test_sd(): 
    ## test des périphériques audio
    t = np.linspace(0,2,num=88200,endpoint=False)
    s1d=0*t
    s1g = AMP_REC*sgnl.tukey(88200,alpha=0.1)*np.cos(2*np.pi*440*t)
    myrecording = sd.playrec(np.array([s1g,s1d]).transpose(),44100,channels=2, blocking=True)
    plt.plot(myrecording)
    plt.show()


def filter_OLA(s, h): 
    ## implémentation de la méthode Overlap-Add afin de filtrer s par le filtre h
    M = len(h)
    L = M
    N=M+L
    print("N=M+L=",N,"avec M=",M,"et L=",L)
    H = np.fft.rfft(h,n=N)
    Nx = len(s)
    a=0
    y=np.zeros(Nx+M)
    while a<Nx-L:
        b=a+L
        yt = np.fft.irfft( np.fft.rfft( s[a:b], n=N ) * H, n=N )
        c=a+N
        for i in range(N):
            y[a+i]+=yt[i]
        a+=L
    return y
    
def engine1(f,duree): 
    ## inteface audio, utilisée pour mesurer Hp (dans transferHp())
    ## émet le bruit primaire (sinusoïde de freq f), pas de bruit secondaire
    global BUFFER,a,s1_rec,s2_rec,s3_rec,s4_rec
    
    L = FFTSIZE    # taille d'un bloc
    BUFFER = np.zeros(2*L)
    a=0    # a est l'instant de début de chaque bloc (en échantillons)
    
    def callback(indata, outdata, frames, time, status):
        global BUFFER,a,s1_rec,s2_rec,s3_rec,s4_rec
        if status:
            print(status)
            
        s1 = np.array(indata).transpose()[0]
        s2 = np.array(indata).transpose()[1]
        s3 = np.zeros(L)
        s4 = np.zeros(L)
        
        
        t = np.linspace(a/SMPFREQ,(a+L)/SMPFREQ,num=L,endpoint=False)
        s3 = AMP_REC*np.cos(2*np.pi*f*t)
        a+=L
        
        s1_rec = np.concatenate([s1_rec,s1])
        s2_rec = np.concatenate([s2_rec,s2])
        s3_rec = np.concatenate([s3_rec,s3])
        s4_rec = np.concatenate([s4_rec,s4])
        
        outdata[:] = np.array([s3,s4]).transpose()
        
    with sd.Stream(samplerate=SMPFREQ, blocksize=L, latency='low',
                   channels=2, callback=callback):
        #print("in stream! (engine1)")
        sd.sleep(int(duree*1000))

def engine2(f,duree): 
    ## inteface audio, utilisée pour mesurer Hs (dans transferHs() et transferHf())
    ## émet le bruit secondaire (sinusoïde de freq f), pas de bruit primaire
    global BUFFER,a,s1_rec,s2_rec,s3_rec,s4_rec

    L = FFTSIZE    # taille d'un bloc
    BUFFER = np.zeros(2*L)
    a=0    # a est l'instant de début de chaque bloc (en échantillons)
    
    def callback(indata, outdata, frames, time, status):
        global BUFFER,a,s1_rec,s2_rec,s3_rec,s4_rec
        if status:
            print(status)
            
        s1 = np.array(indata).transpose()[0]
        s2 = np.array(indata).transpose()[1]
        s3 = np.zeros(L)
        s4 = np.zeros(L)
        
        
        t = np.linspace(a/SMPFREQ,(a+L)/SMPFREQ,num=L,endpoint=False)
        s4 = AMP_REC*np.cos(2*np.pi*f*t)
        a+=L
        
        s1_rec = np.concatenate([s1_rec,s1])
        s2_rec = np.concatenate([s2_rec,s2])
        s3_rec = np.concatenate([s3_rec,s3])
        s4_rec = np.concatenate([s4_rec,s4])
        
        outdata[:] = np.array([s3,s4]).transpose()
        
    with sd.Stream(samplerate=SMPFREQ, blocksize=L, latency='low',
                   channels=2, callback=callback):
        #print("in stream! (engine2)")
        sd.sleep(int(duree*1000))
        
        
def engine4(w,f, duree=None, mesure=False): 
    ## interface audio de la fonction principale du contrôle actif (controleactif())
    ## émet le bruit primaire (son composé de fréq fondamentale f), émet le bruit secondaire (qui est la signal du micro de réf filtré par w)
    global BUFFER,a,s1_rec,s2_rec,s3_rec,s4_rec,A1

    L=len(w)    # taille d'un bloc
    N=2*L
    W=np.fft.rfft(w,n=N)
    BUFFER = np.zeros(N)
    a=0    # a est l'instant de début de chaque bloc (en échantillons)
    A1=None

        
    def callback(indata, outdata, frames, time, status):
        global BUFFER,a,s1_rec,s2_rec,s3_rec,s4_rec,A1
        if status:
            print(status)
            
        s1 = np.array(indata).transpose()[0]
        s2 = np.array(indata).transpose()[1]
        s3 = np.zeros(L)
        s4 = np.zeros(L)
        
        ## après le délai d'activation du contrôle actif, s4 (bruit secondaire) est le signal s1 (réf) filtré par w
        if a>=DELAI_ACTIVATION_CONTROLE_ACTIF*SMPFREQ :
            if A1==None:A1 = (2**.5)*np.mean(s2.astype(np.float)**2)**.5 # amplitude avant contrôle actif
            yt = np.fft.irfft( np.fft.rfft( s1, n=N ) * W, n=N )
            y=np.zeros(N)
            for i in range(L):
                y[i]+=BUFFER[L+i]+yt[i]
            for i in range(L,N):
                y[i]+=yt[i]
            s4=y[0:L]
            BUFFER = y
        
        t = np.linspace(a/SMPFREQ,(a+L)/SMPFREQ,num=L,endpoint=False)
        s3 = AMP_REC*(0.5*np.cos(2*np.pi*f*t)+0.25*np.cos(4*np.pi*f*t)+0.17*np.cos(6*np.pi*f*t)+0.13*np.cos(8*np.pi*f*t))   # bruit primaire
        
        a+=L
        
        s1_rec = np.concatenate([s1_rec[-30*SMPFREQ:],s1])
        s2_rec = np.concatenate([s2_rec[-30*SMPFREQ:],s2])
        s3_rec = np.concatenate([s3_rec[-30*SMPFREQ:],s3])
        s4_rec = np.concatenate([s4_rec[-30*SMPFREQ:],s4])
        
        outdata[:] = np.array([s3,s4]).transpose()
        
    with sd.Stream(samplerate=SMPFREQ, blocksize=L, latency='low',
                   channels=2, callback=callback):
        if mesure:
            while True:
                if a>=DELAI_ACTIVATION_CONTROLE_ACTIF*SMPFREQ and A1!=None :
                    A2 = (2**.5)*np.mean(s2_rec[-L:].astype(np.float)**2)**.5
                    print("Gain instantané :",20*np.log10(A2/A1))
                sd.sleep(1000)
        elif duree==None:
            print('#' * 80)
            print('Appuyer sur Entrée pour quitter')
            print('#' * 80)
            input()
        else:
            sd.sleep(int(duree*1000))

def controleactif_sine(f, duree=None, mesure=False):    
    ## FONCTION PRINCIPALE DU CONTROLE ACTIF DANS LE CADRE D'UN BRUIT SINUSOIDAL PUR
    global BUFFER,a,s1_rec,s2_rec,s3_rec,s4_rec,r1,phi1,A1
    BUFFER = np.zeros(FFTSIZE*2)
    a=0
    
    ## mesure de Hp(f)
    clear_rec()
    nsamples = MARGE_REC+3*SUB
    engine1(f,nsamples/SMPFREQ)
    r_p, phi_p = transfer_sine(s1_rec[MARGE_REC:MARGE_REC+SUB],s2_rec[MARGE_REC:MARGE_REC+SUB],f,SMPFREQ)
    #print("Hp(f) OK")
    
    ## mesure de Hs(f)
    clear_rec()
    nsamples = MARGE_REC+3*SUB
    engine2(f,nsamples/SMPFREQ)
    r_s, phi_s = transfer_sine(s4_rec[MARGE_REC:MARGE_REC+SUB],s2_rec[MARGE_REC:MARGE_REC+SUB],f,SMPFREQ)
    #print("Hs(f) OK")
    
    '''## mesure de Hf(f)
    clear_rec()
    nsamples = MARGE_REC+3*SUB
    engine3(f,nsamples/SMPFREQ)
    r_f, phi_f = transfer_sine(s4_rec[MARGE_REC:MARGE_REC+SUB],s1_rec[MARGE_REC:MARGE_REC+SUB],f,SMPFREQ)
    #print("Hf(f) OK")'''
    # remarque : on ne mesure pas Hf ici : le retour du bruit secondaire dans le micro de réf n'est pas pris en compte
    
    ## calcul du gain et du déphasage (par rapport au signal de réf) à appliquer au contre-bruit
    r_f,phi_f=0,0
    z = r_f*np.exp(phi_f*1j) - (r_s/r_p)*np.exp((phi_s-phi_p)*1j)
    R = 1/np.abs(z)
    PHI = - np.angle(z)
    
    ## c'est parti
    L=FFTSIZE   # taille d'un bloc
    BUFFER = np.zeros(2*L)
    a=0
    clear_rec()
    r1,phi1,A1=None,None,None
    
    def callback(indata, outdata, frames, time, status):
        global BUFFER,a,s1_rec,s2_rec,s3_rec,s4_rec,r1,phi1,A1
        if status:
            print(status)
            
        s1 = np.array(indata).transpose()[0]
        s2 = np.array(indata).transpose()[1]
        s3 = np.zeros(L)
        s4 = np.zeros(L)
        
        
        t = np.linspace(a/SMPFREQ,(a+L)/SMPFREQ,num=L,endpoint=False)
        s3 = AMP_REC*np.cos(2*np.pi*f*t)
        if a>=DELAI_ACTIVATION_CONTROLE_ACTIF*SMPFREQ :
            if A1==None:A1 = (2**.5)*np.mean(s2.astype(np.float)**2)**.5 # amplitude avant contrôle actif
            if r1==None:r1,phi1 = transfer_sine(np.cos(2*np.pi*f*t), s1, f, SMPFREQ) # calcul de l'amplitude et de la phase de s1
            s4 = R*r1*np.cos(2*np.pi*f*t + phi1+PHI) # S4 = W*S1
        a+=L
        
        s1_rec = np.concatenate([s1_rec[-30*SMPFREQ:],s1])
        s2_rec = np.concatenate([s2_rec[-30*SMPFREQ:],s2])
        s3_rec = np.concatenate([s3_rec[-30*SMPFREQ:],s3])
        s4_rec = np.concatenate([s4_rec[-30*SMPFREQ:],s4])
        
        
        outdata[:] = np.array([s3,s4]).transpose()
        
    with sd.Stream(samplerate=SMPFREQ, blocksize=L, latency='low',
                   channels=2, callback=callback):
        if mesure:
            while True:
                if a>=DELAI_ACTIVATION_CONTROLE_ACTIF*SMPFREQ and A1!=None :
                    A2 = (2**.5)*np.mean(s2_rec[-L:].astype(np.float)**2)**.5
                    print("Gain instantané :",20*np.log10(A2/A1))
                sd.sleep(1000)
        elif duree==None:
            print('#' * 80)
            print('Appuyer sur Entrée pour quitter')
            print('#' * 80)
            input()
        else:
            sd.sleep(int(duree*1000))
        
def controleactif(f=220, duree=None, mesure=False):   
    ## FONCTION PRINCIPALE DU CONTROLE ACTIF POUR UN BRUIT COMPOSE
    try: W
    except NameError: 
        print("erreur : contrôle non initialisé (init_controleactif())")
        return

    
    w = np.fft.irfft(W)
    engine4(w,f, duree=duree, mesure=mesure)


def init_controleactif():   
    ## initialise Hp, Hs et Hf
    global Hp,Hs,Hf,W
    global s1_rec,s2_rec,s3_rec,s4_rec
    fmin,fmax=100,2000
    fftfreq = np.fft.rfftfreq(FFTSIZE,1/SMPFREQ)    # fréquences (Hz) de la FFT
    nsamples = MARGE_REC+3*SUB    # durée (échantillons) 

    ## mesure de Hp
    Hp = np.zeros(len(fftfreq), dtype=complex)
    for i in range(len(fftfreq)):
        f=fftfreq[i]
        if fmin<=f<=fmax:
            clear_rec()
            engine1(f,duree=nsamples/SMPFREQ)
            r,phi = transfer_sine(s1_rec[MARGE_REC:MARGE_REC+SUB],s2_rec[MARGE_REC:MARGE_REC+SUB],f,SMPFREQ)
            Hp[i] = r*np.exp(phi*1j)
    
    ## mesure de Hs et de Hf
    Hs = np.zeros(len(fftfreq), dtype=complex)
    Hf = np.zeros(len(fftfreq), dtype=complex)
    for i in range(len(fftfreq)):
        f=fftfreq[i]
        if fmin<=f<=fmax:
            clear_rec()
            engine2(f,duree=nsamples/SMPFREQ)
            
            r,phi = transfer_sine(s4_rec[MARGE_REC:MARGE_REC+SUB],s2_rec[MARGE_REC:MARGE_REC+SUB],f,SMPFREQ)
            Hs[i] = r*np.exp(phi*1j)
            
            r,phi = transfer_sine(s4_rec[MARGE_REC:MARGE_REC+SUB],s1_rec[MARGE_REC:MARGE_REC+SUB],f,SMPFREQ)
            Hf[i] = r*np.exp(phi*1j)
    
    ## calcul du filtre
    W = [1/(Hf[i]-Hs[i]/Hp[i]) if Hp[i]!=0 else 10**(-48/20) for i in range(len(Hp))] # W = 1/(Hf-Hs/Hp)    # 10**(-48/20) vaut -48dB
        
def bode(H):    
    ## diagramme de Bode de H
    F = np.fft.rfftfreq(FFTSIZE,1/SMPFREQ)
    G = 20*np.log10(np.abs(H))
    P = np.angle(H, deg=True)
    plt.subplot(211)
    plt.semilogx(F,G)
    plt.title("Gain (dB)")
    plt.grid()
    plt.subplot(212)
    plt.semilogx(F,P)
    plt.title("Phase (°)")
    plt.grid()
    plt.show()
    
def stats(n=10,f_list=[440]):
    ## mesure les gains obtenus pour des bruits primaires purs de différentes fréquences
    G_list = [ [] for f in f_list ]
    for j in range(len(f_list)):
        f=f_list[j]
        for i in range(n):
            clear_rec()
            controleactif_sine(f,duree=DELAI_ACTIVATION_CONTROLE_ACTIF+4*SUB/SMPFREQ)
            A1 = (2**.5)*np.mean(s2_rec[MARGE_REC:MARGE_REC+SUB].astype(np.float)**2)**.5 # amplitude avant contrôle actif
            A2 = (2**.5)*np.mean(s2_rec[-SUB:].astype(np.float)**2)**.5 # amplitude après contrôle actif
            G = 20*np.log10(A2/A1)
            G_list[j].append(G)
        print("Gain moyen pour f=",f,":",np.mean(G_list[j]))

def clear_rec():
    global s1_rec,s2_rec,s3_rec,s4_rec
    s1_rec = np.array([])
    s2_rec = np.array([])
    s3_rec = np.array([])
    s4_rec = np.array([])
    
## EXEMPLE D'UTILISATION :
#init_controleactif()
#bode(W)
#controleactif(f=440,mesure=True)
#stats(n=6,f_list=[262,330,440,523,659,880,1046.5,1318.5,1760])