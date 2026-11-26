import numpy as np, scipy.fft as sf, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS=122.88e6; NFFT=4096; NA=2604; CP=288; TS=NFFT+CP; M=128
K=np.r_[np.arange(-NA//2,0),np.arange(1,NA//2+1)]%NFFT
GEN=24            # genuine echo delay (samples)
SNR_DB=0          # genuine-echo SNR per time sample
HIST=4            # symbols of history kept ahead of the processed block
LAGS=(0,1,2,3)
THR=-np.log(1e-3/(NFFT*M*3))   # cell threshold: ~1e-3 false alarms per CPI over lags 1..3

def tx(rng,repeat=False):
    n=M+HIST+1
    q=lambda s:(rng.choice([-1.,1.],s)+1j*rng.choice([-1.,1.],s)).astype(np.complex64)/np.float32(np.sqrt(2))
    A=np.tile(q((1,NA)),(n,1)) if repeat else q((n,NA))
    G=np.zeros((n,NFFT),np.complex64); G[:,K]=A
    x=sf.ifft(G,axis=1)*np.float32(NFFT/np.sqrt(NA))
    return G,np.concatenate([x[:,-CP:],x],axis=1).ravel()

def rx(rng,s,D,jsr_db):
    start=HIST*TS; L=M*TS
    cn=lambda:((rng.standard_normal(L)+1j*rng.standard_normal(L))/np.sqrt(2)).astype(np.complex64)
    ph=lambda:np.complex64(np.exp(2j*np.pi*rng.random()))
    y=ph()*s[start-GEN:start-GEN+L] + np.float32(10**(-SNR_DB/20))*cn()
    if D is not None: y=y+np.float32(10**(jsr_db/20))*ph()*s[start-D:start-D+L]
    return sf.fft(y.reshape(M,TS)[:,CP:],axis=1)

def rdmap(Y,G,lag,win):
    H=np.zeros_like(Y); H[:,K]=Y[:,K]*np.conj(G[HIST-lag:HIST-lag+M][:,K])*win
    return np.abs(sf.fft(sf.ifft(H,axis=1),axis=0))**2

def flagged(Y,G):
    hit=False
    for l in (1,2,3):
        P=rdmap(Y,G,l,np.ones(NA,np.float32))
        mean=np.median(P)/np.log(2)
        if P.max()>THR*mean: hit=True; break
    return hit

rng=np.random.default_rng(7)
# --- sweep: fraction of CPIs in which a stale-reference detection appears
delays_us=[0.5,1.5,2.2,2.4,2.5,2.7,3.0,3.9,6,12,36,75]
TR=100; res={}
for jsr in (0,-20):
    res[jsr]=[]
    for d in delays_us:
        D=int(round(d*1e-6*FS)); c=0
        for _ in range(TR):
            G,s=tx(rng); c+=flagged(rx(rng,s,D,jsr),G)
        res[jsr].append(c/TR); print(jsr,d,c/TR,flush=True)
fa=0; NFA=1000
for _ in range(NFA):
    G,s=tx(rng); fa+=flagged(rx(rng,s,None,0),G)
print('false flags',fa,NFA)
json.dump(dict(delays_us=delays_us,res=res,fa=fa,nfa=NFA,trials=TR),open('results.json','w'))

# --- example profiles: replay two symbols (+ 96 samples) old
D=2*TS+96; win=np.hanning(NA).astype(np.float32)
def prof(Y,G,l):
    return rdmap(Y,G,l,win)[0,:200]
G,s=tx(rng,repeat=True); a=prof(rx(rng,s,D,0),G,0)
G,s=tx(rng); Y=rx(rng,s,D,0); b=prof(Y,G,0); c=prof(Y,G,2); ref=b.max()
rm=np.arange(200)/FS*3e8/2
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],'font.size':7,'axes.linewidth':.6})
fig,ax=plt.subplots(2,1,figsize=(3.45,2.75),gridspec_kw=dict(height_ratios=[1,1],hspace=.72))
for v,lab,st in ((a,'Repeated symbol','k:'),(b,'Fresh data, lag 0','k-'),(c,'Fresh data, lag 2','r--')):
    ax[0].plot(rm,10*np.log10(v/ref+1e-12),st,lw=.9,label=lab)
ax[0].set(ylim=(-70,6),xlim=(0,240),xlabel='Equivalent monostatic range (m)',ylabel='Power (dB)')
ax[0].legend(fontsize=6,ncol=3,loc='lower center',bbox_to_anchor=(.5,.99),frameon=False,handlelength=1.8,columnspacing=.8)
ax[0].annotate('(a)  echo',(GEN/FS*1.5e8,0),(41,-8),fontsize=6); ax[0].annotate('replay',(96/FS*1.5e8,0),(129,-8),fontsize=6)
for jsr,st,lab in ((0,'ko-','Replay = echo power'),(-20,'rs--','Replay 20 dB weaker')):
    ax[1].semilogx(delays_us,res[jsr],st,ms=3,lw=.9,label=lab)
ax[1].axvline(CP/FS*1e6,color='0.5',lw=.8,ls=':'); ax[1].text(2.15,.62,'(b)  CP',ha='right',fontsize=6)
ax[1].set(xlabel='Replay delay after transmission (µs)',ylabel='Fraction of CPIs flagged',ylim=(-.05,1.08))
ax[1].set_xticks([.5,1,2.34,5,10,36,75]); ax[1].set_xticklabels(['0.5','1','2.34','5','10','36','75']); ax[1].minorticks_off()
ax[1].legend(fontsize=6,loc='center right',frameon=False)
fig.savefig('fig_sim.png',dpi=450,bbox_inches='tight',pad_inches=.02)
